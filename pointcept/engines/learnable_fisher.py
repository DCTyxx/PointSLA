import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableCovarianceFisher(nn.Module):
    """Support-conditioned learnable Fisher-Riemannian metric.

    Implements
        G_c^LF = G_c^{F/2} (I + alpha A_c A_c^T) G_c^{F/2},
    where
        G_c^F = (Sigma_c + eps I)^{-1},
        A_c = h_theta(mu_c, Lambda_c, R_c).

    Computation is performed in the retained covariance eigenspace to avoid
    constructing dense D x D metric tensors.
    """

    def __init__(
        self,
        feat_dim=512,
        rank=32,
        low_rank=8,
        hidden_dim=128,
        alpha=0.25,
        eps=1e-3,
        shrinkage=0.1,
        relation_topk=3,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.rank = int(rank)
        self.low_rank = int(low_rank)
        self.hidden_dim = int(hidden_dim)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.shrinkage = float(shrinkage)
        self.relation_topk = int(relation_topk)

        descriptor_dim = 2 * self.rank + 2 * self.relation_topk
        self.generator = nn.Sequential(
            nn.Linear(descriptor_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.rank * self.low_rank),
        )
        # A_c = 0 at initialization, so the method starts from fixed Fisher.
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)

    @staticmethod
    def _covariance(x):
        if x.shape[0] <= 1:
            dim = x.shape[1]
            return torch.eye(dim, device=x.device, dtype=x.dtype)
        centered = x - x.mean(dim=0, keepdim=True)
        return centered.transpose(0, 1) @ centered / float(x.shape[0] - 1)

    def _shrink_covariance(self, cov):
        dim = cov.shape[0]
        cov = 0.5 * (cov + cov.transpose(0, 1))
        avg_var = torch.trace(cov).clamp_min(1e-8) / dim
        eye = torch.eye(dim, device=cov.device, dtype=cov.dtype)
        return (1.0 - self.shrinkage) * cov + self.shrinkage * avg_var * eye

    def fit_support(self, class_features):
        """Estimate mu_c, Lambda_c, R_c and generate A_c for an episode."""
        if len(class_features) == 0:
            raise ValueError("class_features must contain at least one class")

        means, bases, evals, projected_means = [], [], [], []
        for x in class_features:
            if x.ndim != 2 or x.shape[1] != self.feat_dim:
                raise ValueError(
                    f"Expected support features [N,{self.feat_dim}], got {tuple(x.shape)}"
                )
            x = x.float()
            mu = x.mean(dim=0)
            cov = self._shrink_covariance(self._covariance(x))
            eigvals, eigvecs = torch.linalg.eigh(cov)
            r = min(self.rank, eigvals.numel())
            lam = eigvals[-r:].clamp_min(self.eps)
            U = eigvecs[:, -r:]
            if r < self.rank:
                pad_r = self.rank - r
                U = F.pad(U, (0, pad_r))
                lam = F.pad(lam, (0, pad_r), value=self.eps)
            means.append(mu)
            bases.append(U)
            evals.append(lam)
            projected_means.append(mu @ U)

        means = torch.stack(means, dim=0)
        bases = torch.stack(bases, dim=0)
        evals = torch.stack(evals, dim=0)
        projected_means = torch.stack(projected_means, dim=0)
        relations = self._inter_class_relations(means)
        descriptor = torch.cat(
            [projected_means, torch.log(evals + self.eps), relations], dim=-1
        )
        A = self.generator(descriptor).view(
            len(class_features), self.rank, self.low_rank
        )
        return {
            "means": means,
            "bases": bases,
            "evals": evals,
            "A": A,
            "relations": relations,
        }

    def _inter_class_relations(self, means):
        """R_c: top-k nearest competing-class cosine and Euclidean relations."""
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
        nearest_idx = euclidean.topk(kk, largest=False, dim=-1).indices
        cos_rel = torch.gather(cosine, 1, nearest_idx)
        l2_rel = torch.gather(euclidean, 1, nearest_idx)
        if kk < k:
            cos_rel = F.pad(cos_rel, (0, k - kk))
            l2_rel = F.pad(l2_rel, (0, k - kk))
        return torch.cat([cos_rel, torch.log1p(l2_rel.clamp_min(0.0))], dim=-1)

    def distances(self, query, stats):
        """Compute d_c^2=(q-mu_c)^T G_c^LF (q-mu_c)."""
        q = query.float()
        means = stats["means"]
        bases = stats["bases"]
        evals = stats["evals"]
        A = stats["A"]
        all_dist = []
        for c in range(means.shape[0]):
            delta = q - means[c]
            U = bases[c]
            lam = evals[c]
            Ac = A[c]

            # Fisher-normalized coordinate x = G_F^{1/2}(q-mu_c)
            # represented in the retained covariance eigenspace.
            x = (delta @ U) / torch.sqrt(lam + self.eps)
            base = x.square().sum(dim=-1)
            low_rank = (x @ Ac).square().sum(dim=-1)
            dist = base + self.alpha * low_rank

            # Weak positive ridge on discarded directions.
            projected = (delta @ U) @ U.transpose(0, 1)
            residual = delta - projected
            dist = dist + self.eps * residual.square().sum(dim=-1)
            all_dist.append(dist)
        return torch.stack(all_dist, dim=1)

    def logits(self, query, stats, temperature=1.0):
        return -self.distances(query, stats) / max(float(temperature), 1e-6)

    def deformation_regularization(self, stats):
        return stats["A"].square().mean()

    def metric_diagnostics(self, stats):
        with torch.no_grad():
            A = stats["A"]
            return {
                "a_rms": A.square().mean().sqrt().item(),
                "a_max": A.abs().max().item(),
                "alpha": self.alpha,
                "rank": self.rank,
                "low_rank": self.low_rank,
            }
