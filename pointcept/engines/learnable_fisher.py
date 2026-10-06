import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableCovarianceFisher(nn.Module):
    """Support-conditioned low-rank deformation of covariance Fisher geometry.

    Core formulation:
        G_c^LF = G_c^{F/2} (I + alpha A_c A_c^T) G_c^{F/2},
        G_c^F  = (Sigma_c + eps I)^(-1),
        A_c    = h_theta(mu_c, Lambda_c, R_c).

    For efficiency, G_c^F is represented by a rank-m eigenspace plus an
    isotropic residual precision. The learned deformation acts on the resulting
    approximate G_c^{F/2}, avoiding dense D x D transforms for every query.

    Optionally, the support covariance can be expanded into a local ellipsoidal
    support region before inversion. The retained/tangent directions and the
    residual subspace are expanded anisotropically in closed form, which is
    equivalent to integrating zero-mean Gaussian perturbations around observed
    support features without Monte-Carlo sampling.
    """

    def __init__(
        self,
        feat_dim=512,
        spectral_rank=32,
        low_rank=8,
        hidden_dim=128,
        alpha=0.25,
        eps=1e-3,
        shrinkage=0.1,
        relation_topk=3,
        init_scale=1e-3,
        match_temperature=1.0,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.spectral_rank = min(int(spectral_rank), self.feat_dim)
        self.low_rank = int(low_rank)
        self.hidden_dim = int(hidden_dim)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.shrinkage = float(shrinkage)
        self.relation_topk = int(relation_topk)
        self.init_scale = float(init_scale)
        self.match_temperature = float(match_temperature)
        self.metric = "learnable_cov_fisher"
        self.support_stats = None

        # Stage-3 support-region exploration is opt-in so the exact Stage-1
        # behaviour remains available through the original trainer/config.
        self.region_exploration = False
        self.region_tangent_scale = 0.0
        self.region_residual_scale = 0.0

        self.mu_encoder = nn.Sequential(
            nn.Linear(self.feat_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.spectrum_encoder = nn.Sequential(
            nn.Linear(self.spectral_rank + 1, self.hidden_dim),
            nn.GELU(),
        )
        self.relation_encoder = nn.Sequential(
            nn.Linear(2 * self.relation_topk, self.hidden_dim),
            nn.GELU(),
        )
        self.generator = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.feat_dim * self.low_rank),
        )

        # A=0 is a stationary point of A A^T. Tiny non-zero initialization
        # keeps the initial geometry practically fixed Fisher but trainable.
        nn.init.normal_(self.generator[-1].weight, mean=0.0, std=self.init_scale)
        nn.init.normal_(self.generator[-1].bias, mean=0.0, std=self.init_scale)

    @property
    def needs_support_statistics(self):
        return False

    def reset_support_statistics(self, *args, **kwargs):
        return None

    def update_support_statistics(self, *args, **kwargs):
        return None

    def finalize_support_statistics(self):
        return None

    def configure_region_exploration(
        self,
        enabled=True,
        tangent_scale=0.5,
        residual_scale=0.1,
    ):
        """Configure closed-form anisotropic exploration of support regions.

        Let the empirical covariance eigensystem be (U, lambda). Around each
        observed support feature, consider zero-mean local perturbations whose
        standard deviations are ``tangent_scale * sqrt(lambda)`` in the retained
        eigenspace and ``residual_scale * sqrt(residual_var)`` in the orthogonal
        residual subspace. Marginalizing those perturbations adds their covariance,
        therefore the explored variances are exactly

            lambda_tilde = (1 + tangent_scale^2) * lambda,
            var_perp_tilde = (1 + residual_scale^2) * var_perp.

        No synthetic points are materialized and no extra stochasticity is added.
        """
        tangent_scale = float(tangent_scale)
        residual_scale = float(residual_scale)
        if tangent_scale < 0.0 or residual_scale < 0.0:
            raise ValueError("Region exploration scales must be non-negative")
        self.region_exploration = bool(enabled)
        self.region_tangent_scale = tangent_scale
        self.region_residual_scale = residual_scale
        return self

    def prepare_support_from_moments(self, sums, outers, counts):
        """Estimate mu_c, a stable covariance spectrum, and R_c."""
        if sums.ndim != 2 or sums.shape[1] != self.feat_dim:
            raise ValueError(f"Expected sums [C,{self.feat_dim}]")
        C, D = sums.shape
        means, bases, evals, residual_precs = [], [], [], []
        raw_evals, raw_residual_vars = [], []
        eye = torch.eye(D, device=sums.device, dtype=sums.dtype)

        with torch.no_grad():
            for c in range(C):
                n = int(counts[c].item())
                if n <= 0:
                    raise RuntimeError(f"No support points for class index {c}")
                nf = counts[c].to(sums.dtype)
                mu = sums[c] / nf
                if n <= 1:
                    cov = eye.clone()
                else:
                    cov = (outers[c] - nf * torch.outer(mu, mu)) / (nf - 1.0)
                    cov = 0.5 * (cov + cov.transpose(0, 1))
                    avg_var = torch.trace(cov).clamp_min(1e-8) / D
                    cov = (
                        (1.0 - self.shrinkage) * cov
                        + self.shrinkage * avg_var * eye
                    )

                lam_all, U_all = torch.linalg.eigh(cov.float())
                lam_all = lam_all.clamp_min(0.0)
                m = self.spectral_rank
                lam = lam_all[-m:]
                U = U_all[:, -m:]

                if m < D:
                    residual_var = lam_all[:-m].mean().clamp_min(0.0)
                else:
                    residual_var = lam_all.new_zeros(())

                # Stage-3: interpret each observed support feature as the centre
                # of a small local ellipsoidal region. We integrate that region
                # analytically by adding the perturbation covariance. Expansion
                # is deliberately stronger along observed retained/tangent
                # directions than in the residual subspace.
                raw_lam = lam
                raw_residual_var = residual_var
                if self.region_exploration:
                    tangent_factor = 1.0 + self.region_tangent_scale ** 2
                    residual_factor = 1.0 + self.region_residual_scale ** 2
                    lam = lam * tangent_factor
                    residual_var = residual_var * residual_factor

                residual_prec = 1.0 / (residual_var + self.eps)

                means.append(mu.float())
                bases.append(U.float())
                evals.append(lam.float())
                residual_precs.append(residual_prec.float())
                raw_evals.append(raw_lam.float())
                raw_residual_vars.append(raw_residual_var.float())

            means = torch.stack(means, dim=0)
            bases = torch.stack(bases, dim=0)
            evals = torch.stack(evals, dim=0)
            residual_precs = torch.stack(residual_precs, dim=0)
            raw_evals = torch.stack(raw_evals, dim=0)
            raw_residual_vars = torch.stack(raw_residual_vars, dim=0)
            relations = self._inter_class_relations(means)

        stats = {
            "means": means.detach(),
            "bases": bases.detach(),
            "evals": evals.detach(),
            "residual_precs": residual_precs.detach(),
            "relations": relations.detach(),
            # Raw spectra are retained only for diagnostics/ablation analysis;
            # distances and the deformation generator use the explored spectra.
            "raw_evals": raw_evals.detach(),
            "raw_residual_vars": raw_residual_vars.detach(),
        }
        self.support_stats = stats
        return stats

    def _inter_class_relations(self, means):
        """R_c: top-k strongest competing classes by prototype cosine."""
        C = means.shape[0]
        k = self.relation_topk
        if C <= 1:
            return means.new_zeros((C, 2 * k))

        norm_means = F.normalize(means, dim=-1)
        cosine = norm_means @ norm_means.transpose(0, 1)
        euclidean = torch.cdist(means, means, p=2)
        eye = torch.eye(C, device=means.device, dtype=torch.bool)
        cosine = cosine.masked_fill(eye, -float("inf"))
        euclidean = euclidean.masked_fill(eye, float("inf"))

        kk = min(k, C - 1)
        competitor_idx = cosine.topk(kk, largest=True, dim=-1).indices
        cos_rel = torch.gather(cosine, 1, competitor_idx)
        l2_rel = torch.gather(euclidean, 1, competitor_idx)
        if kk < k:
            cos_rel = F.pad(cos_rel, (0, k - kk))
            l2_rel = F.pad(l2_rel, (0, k - kk))
        return torch.cat([cos_rel, torch.log1p(l2_rel.clamp_min(0.0))], dim=-1)

    def _stats_on(self, support_stats, device):
        return {key: value.to(device) for key, value in support_stats.items()}

    def generate_deformation(self, support_stats):
        """Generate feature-space A_c=h_theta(mu_c,Lambda_c,R_c), [C,D,r]."""
        mu = support_stats["means"]
        lam = support_stats["evals"]
        residual_prec = support_stats["residual_precs"]
        rel = support_stats["relations"]

        mu_desc = self.mu_encoder(F.normalize(mu, dim=-1))
        residual_var = (1.0 / residual_prec.clamp_min(1e-12)) - self.eps
        spectrum = torch.cat(
            [
                torch.log(lam + self.eps),
                torch.log(residual_var[:, None].clamp_min(0.0) + self.eps),
            ],
            dim=-1,
        )
        spec_desc = self.spectrum_encoder(spectrum)
        rel_desc = self.relation_encoder(rel)
        descriptor = torch.cat([mu_desc, spec_desc, rel_desc], dim=-1)
        A = self.generator(descriptor)
        return A.view(mu.shape[0], self.feat_dim, self.low_rank)

    def distances(self, query, support_stats=None):
        """Compute raw support-conditioned Fisher-Riemannian distances [N,C]."""
        if support_stats is None:
            support_stats = self.support_stats
        if support_stats is None:
            raise RuntimeError("Learnable Fisher requested before support statistics were fit")
        if query.ndim == 1:
            query = query.unsqueeze(0)

        q = query.float()
        stats = self._stats_on(support_stats, q.device)
        means = stats["means"]
        bases = stats["bases"]
        evals = stats["evals"]
        residual_precs = stats["residual_precs"]
        A = self.generate_deformation(stats)

        all_dist = []
        for c in range(means.shape[0]):
            delta = q - means[c]
            U = bases[c]
            lam = evals[c]
            rho = residual_precs[c]

            coeff = delta @ U
            whitened = coeff / torch.sqrt(lam + self.eps)
            retained = whitened.square().sum(dim=-1)

            delta_norm = delta.square().sum(dim=-1)
            projected_norm = coeff.square().sum(dim=-1)
            residual_norm = (delta_norm - projected_norm).clamp_min(0.0)
            fixed_fisher = retained + rho * residual_norm

            # Efficiently evaluate y A where
            # y = delta G_F^{1/2}
            #   = whitened U^T + sqrt(rho) delta (I-UU^T).
            # This includes both retained and isotropic residual subspaces
            # without forming a dense D x D Fisher square root.
            A_sub = U.transpose(0, 1) @ A[c]
            retained_A = whitened @ A_sub
            residual_A = delta @ A[c] - coeff @ A_sub
            fisher_half_A = retained_A + torch.sqrt(rho) * residual_A
            learned_residual = fisher_half_A.square().sum(dim=-1)

            all_dist.append(fixed_fisher + self.alpha * learned_residual)

        return torch.stack(all_dist, dim=1)

    def normalized_distances(self, query, support_stats=None):
        """Return mean Fisher energy per feature dimension.

        The quadratic form accumulates energy over D feature dimensions. Dividing
        by D makes its scale substantially less sensitive to feature dimension
        and prevents unnecessarily large logits during metric learning.
        """
        return self.distances(query, support_stats) / float(self.feat_dim)

    def fixed_fisher_distances(self, query, support_stats=None):
        if support_stats is None:
            support_stats = self.support_stats
        if support_stats is None:
            raise RuntimeError("Fixed Fisher requested before support statistics were fit")
        if query.ndim == 1:
            query = query.unsqueeze(0)

        q = query.float()
        stats = self._stats_on(support_stats, q.device)
        means = stats["means"]
        bases = stats["bases"]
        evals = stats["evals"]
        residual_precs = stats["residual_precs"]

        all_dist = []
        for c in range(means.shape[0]):
            delta = q - means[c]
            U = bases[c]
            lam = evals[c]
            coeff = delta @ U
            retained = (coeff.square() / (lam + self.eps)).sum(dim=-1)
            residual_norm = (
                delta.square().sum(dim=-1) - coeff.square().sum(dim=-1)
            ).clamp_min(0.0)
            all_dist.append(retained + residual_precs[c] * residual_norm)
        return torch.stack(all_dist, dim=1)

    def logits(self, query, support_stats=None, temperature=None):
        if temperature is None:
            temperature = self.match_temperature
        return -self.normalized_distances(query, support_stats) / max(
            float(temperature), 1e-6
        )

    def probabilities(self, query, support_stats=None, temperature=None):
        return torch.softmax(self.logits(query, support_stats, temperature), dim=-1)

    def similarity(self, query, prototypes=None, class_indices=None):
        """Compatibility API for GFS-VL calibration using class posteriors."""
        del prototypes
        probs = self.probabilities(query)
        if class_indices is not None:
            idx = torch.as_tensor(class_indices, device=probs.device, dtype=torch.long)
            probs = probs.index_select(1, idx)
        return probs

    def deformation_regularization(self, support_stats=None):
        if support_stats is None:
            support_stats = self.support_stats
        if support_stats is None:
            raise RuntimeError("Regularization requested before support statistics were fit")
        device = next(self.parameters()).device
        stats = self._stats_on(support_stats, device)
        return self.generate_deformation(stats).square().mean()

    def diagnostics(self, support_stats=None):
        if support_stats is None:
            support_stats = self.support_stats
        with torch.no_grad():
            device = next(self.parameters()).device
            stats = self._stats_on(support_stats, device)
            A = self.generate_deformation(stats)
            diag = {
                "a_rms": float(A.square().mean().sqrt().cpu()),
                "a_max": float(A.abs().max().cpu()),
                "alpha": self.alpha,
                "spectral_rank": self.spectral_rank,
                "low_rank": self.low_rank,
                "region_exploration": self.region_exploration,
                "region_tangent_scale": self.region_tangent_scale,
                "region_residual_scale": self.region_residual_scale,
            }
            if "raw_evals" in stats:
                raw = stats["raw_evals"].clamp_min(1e-12)
                diag["region_retained_variance_ratio"] = float(
                    (stats["evals"] / raw).mean().cpu()
                )
            return diag
