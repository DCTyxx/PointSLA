import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableCovarianceFisher(nn.Module):
    """Support-conditioned low-rank deformation of covariance Fisher geometry.

    Core formulation:
        G_c^LF = G_c^{F/2} (I + alpha A_c A_c^T) G_c^{F/2}
        G_c^F  = (Sigma_c + eps I)^{-1}
        A_c    = h_theta(mu_c, Lambda_c, R_c)

    The support episode determines mu_c, Sigma_c, Lambda_c and R_c. Only
    h_theta is learned. A_c lives in the original feature coordinates, while
    G_c^{F/2} is applied spectrally.
    """

    def __init__(
        self,
        feat_dim=512,
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

        self.mu_encoder = nn.Sequential(
            nn.Linear(self.feat_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.spectrum_encoder = nn.Sequential(
            nn.Linear(self.feat_dim, self.hidden_dim),
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

        # A=0 is a stationary point of A A^T, so exact zero initialization
        # cannot learn. A tiny non-zero initialization keeps the initial metric
        # practically identical to fixed Fisher while preserving gradients.
        nn.init.normal_(self.generator[-1].weight, mean=0.0, std=self.init_scale)
        nn.init.normal_(self.generator[-1].bias, mean=0.0, std=self.init_scale)

    @property
    def needs_support_statistics(self):
        # Statistics are prepared explicitly from the registration episode.
        return False

    def reset_support_statistics(self, *args, **kwargs):
        return None

    def update_support_statistics(self, *args, **kwargs):
        return None

    def finalize_support_statistics(self):
        return None

    def prepare_support_from_moments(self, sums, outers, counts):
        """Create frozen support statistics from per-class first/second moments."""
        if sums.ndim != 2 or sums.shape[1] != self.feat_dim:
            raise ValueError(f"Expected sums [C,{self.feat_dim}]")
        C, D = sums.shape
        means, bases, evals = [], [], []
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
                lam, U = torch.linalg.eigh(cov.float())
                means.append(mu.float())
                bases.append(U.float())
                evals.append(lam.clamp_min(0.0).float())

            means = torch.stack(means, dim=0)
            bases = torch.stack(bases, dim=0)
            evals = torch.stack(evals, dim=0)
            relations = self._inter_class_relations(means)

        stats = {
            "means": means.detach(),
            "bases": bases.detach(),
            "evals": evals.detach(),
            "relations": relations.detach(),
        }
        self.support_stats = stats
        return stats

    def _inter_class_relations(self, means):
        """R_c: top-k strongest prototype competitors by cosine similarity."""
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
        """A_c = h_theta(mu_c, Lambda_c, R_c), shape [C,D,r]."""
        mu = support_stats["means"]
        lam = support_stats["evals"]
        rel = support_stats["relations"]
        mu_desc = self.mu_encoder(F.normalize(mu, dim=-1))
        spec_desc = self.spectrum_encoder(torch.log(lam + self.eps))
        rel_desc = self.relation_encoder(rel)
        descriptor = torch.cat([mu_desc, spec_desc, rel_desc], dim=-1)
        A = self.generator(descriptor)
        return A.view(mu.shape[0], self.feat_dim, self.low_rank)

    def _fisher_half_apply(self, delta, basis, evals):
        """Apply (Sigma+eps I)^(-1/2) to row vectors delta."""
        coeff = delta @ basis
        coeff = coeff / torch.sqrt(evals + self.eps)
        return coeff @ basis.transpose(0, 1)

    def distances(self, query, support_stats=None):
        """Return d_c^2=(q-mu_c)^T G_c^LF (q-mu_c), shape [N,C]."""
        if support_stats is None:
            support_stats = self.support_stats
        if support_stats is None:
            raise RuntimeError("Learnable Fisher requested before support statistics were fit")
        if query.ndim == 1:
            query = query.unsqueeze(0)
        q = query.float()
        stats = self._stats_on(support_stats, q.device)
        means, bases, evals = stats["means"], stats["bases"], stats["evals"]
        A = self.generate_deformation(stats)

        all_dist = []
        for c in range(means.shape[0]):
            delta = q - means[c]
            y = self._fisher_half_apply(delta, bases[c], evals[c])
            fixed_fisher = y.square().sum(dim=-1)
            learned_residual = (y @ A[c]).square().sum(dim=-1)
            all_dist.append(fixed_fisher + self.alpha * learned_residual)
        return torch.stack(all_dist, dim=1)

    def fixed_fisher_distances(self, query, support_stats=None):
        if support_stats is None:
            support_stats = self.support_stats
        if support_stats is None:
            raise RuntimeError("Fixed Fisher requested before support statistics were fit")
        if query.ndim == 1:
            query = query.unsqueeze(0)
        q = query.float()
        stats = self._stats_on(support_stats, q.device)
        means, bases, evals = stats["means"], stats["bases"], stats["evals"]
        all_dist = []
        for c in range(means.shape[0]):
            delta = q - means[c]
            y = self._fisher_half_apply(delta, bases[c], evals[c])
            all_dist.append(y.square().sum(dim=-1))
        return torch.stack(all_dist, dim=1)

    def logits(self, query, support_stats=None, temperature=None):
        if temperature is None:
            temperature = self.match_temperature
        return -self.distances(query, support_stats) / max(float(temperature), 1e-6)

    def probabilities(self, query, support_stats=None, temperature=None):
        return torch.softmax(self.logits(query, support_stats, temperature), dim=-1)

    def similarity(self, query, prototypes=None, class_indices=None):
        """Compatibility API for GFS-VL calibration.

        Similarity is the posterior over support-conditioned Fisher distances.
        `prototypes` is intentionally ignored because mu_c and Sigma_c are
        already defined by the support episode.
        """
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
        A = self.generate_deformation(stats)
        return A.square().mean()

    def diagnostics(self, support_stats=None):
        if support_stats is None:
            support_stats = self.support_stats
        with torch.no_grad():
            device = next(self.parameters()).device
            stats = self._stats_on(support_stats, device)
            A = self.generate_deformation(stats)
            return {
                "a_rms": float(A.square().mean().sqrt().cpu()),
                "a_max": float(A.abs().max().cpu()),
                "alpha": self.alpha,
                "low_rank": self.low_rank,
            }
