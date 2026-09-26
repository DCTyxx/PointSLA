import torch
import torch.nn.functional as F


class SemanticMetricMatcher:
    """Unified similarity backend for GFS-VL pseudo-label calibration.

    Supported metrics:
      - cosine: original cosine similarity.
      - pca_mahalanobis: a global support-set PCA precision metric.
      - local_pca: class-conditioned support-set PCA precision metrics.
      - fisher_riemannian: a point-dependent Fisher pullback metric induced
        by the frozen text classifier.

    All variants return a metric-induced cosine similarity in [-1, 1] so the
    original GFS-VL thresholds remain interpretable across ablations.
    """

    VALID_METRICS = {
        "cosine",
        "pca_mahalanobis",
        "local_pca",
        "fisher_riemannian",
    }

    def __init__(
        self,
        metric="cosine",
        text_embed=None,
        pca_rank=32,
        pca_shrinkage=0.1,
        metric_ridge=1e-3,
        fisher_temperature=1.0,
    ):
        if metric not in self.VALID_METRICS:
            raise ValueError(
                f"Unknown semantic metric '{metric}'. "
                f"Expected one of {sorted(self.VALID_METRICS)}."
            )
        self.metric = metric
        self.pca_rank = int(pca_rank)
        self.pca_shrinkage = float(pca_shrinkage)
        self.metric_ridge = float(metric_ridge)
        self.fisher_temperature = float(fisher_temperature)

        self.text_embed = None
        if text_embed is not None:
            self.text_embed = text_embed.detach().float()

        self._stats_ready = False
        self._global_sum = None
        self._global_outer = None
        self._global_count = None
        self._class_sum = None
        self._class_outer = None
        self._class_count = None

        self.global_basis = None
        self.global_precision = None
        self.local_basis = None
        self.local_precision = None

    @property
    def needs_support_statistics(self):
        return self.metric in {"pca_mahalanobis", "local_pca"}

    @torch.no_grad()
    def reset_support_statistics(self, num_classes, feat_dim, device):
        if not self.needs_support_statistics:
            return
        dtype = torch.float32
        self._global_sum = torch.zeros(feat_dim, device=device, dtype=dtype)
        self._global_outer = torch.zeros(
            feat_dim, feat_dim, device=device, dtype=dtype
        )
        self._global_count = torch.zeros((), device=device, dtype=torch.long)
        self._class_sum = torch.zeros(
            num_classes, feat_dim, device=device, dtype=dtype
        )
        self._class_outer = torch.zeros(
            num_classes, feat_dim, feat_dim, device=device, dtype=dtype
        )
        self._class_count = torch.zeros(
            num_classes, device=device, dtype=torch.long
        )
        self._stats_ready = False

    @torch.no_grad()
    def update_support_statistics(self, class_index, features):
        if not self.needs_support_statistics or features.numel() == 0:
            return
        x = features.detach().float()
        self._global_sum += x.sum(dim=0)
        self._global_outer += x.transpose(0, 1) @ x
        self._global_count += x.shape[0]

        self._class_sum[class_index] += x.sum(dim=0)
        self._class_outer[class_index] += x.transpose(0, 1) @ x
        self._class_count[class_index] += x.shape[0]

    def _covariance_from_stats(self, sum_x, outer_x, count):
        feat_dim = sum_x.numel()
        if int(count.item()) <= 1:
            return torch.eye(feat_dim, device=sum_x.device, dtype=sum_x.dtype)

        count_f = count.to(sum_x.dtype)
        mean = sum_x / count_f
        cov = (outer_x - count_f * torch.outer(mean, mean)) / (count_f - 1.0)
        cov = 0.5 * (cov + cov.transpose(0, 1))

        avg_var = torch.trace(cov).clamp_min(1e-8) / feat_dim
        eye = torch.eye(feat_dim, device=cov.device, dtype=cov.dtype)
        cov = (
            (1.0 - self.pca_shrinkage) * cov
            + self.pca_shrinkage * avg_var * eye
        )
        return cov

    def _factorize_covariance(self, cov):
        evals, evecs = torch.linalg.eigh(cov.float())
        rank = min(self.pca_rank, cov.shape[0])
        evals = evals[-rank:].clamp_min(self.metric_ridge)
        basis = evecs[:, -rank:]
        precision = evals.reciprocal()
        return basis, precision

    @torch.no_grad()
    def finalize_support_statistics(self):
        if not self.needs_support_statistics:
            return
        if self._global_count is None or int(self._global_count.item()) == 0:
            raise RuntimeError("No support features were collected for PCA metric.")

        global_cov = self._covariance_from_stats(
            self._global_sum, self._global_outer, self._global_count
        )
        self.global_basis, self.global_precision = self._factorize_covariance(
            global_cov
        )

        if self.metric == "local_pca":
            bases = []
            precisions = []
            for idx in range(self._class_sum.shape[0]):
                if int(self._class_count[idx].item()) <= 1:
                    basis = self.global_basis
                    precision = self.global_precision
                else:
                    cov = self._covariance_from_stats(
                        self._class_sum[idx],
                        self._class_outer[idx],
                        self._class_count[idx],
                    )
                    basis, precision = self._factorize_covariance(cov)
                bases.append(basis)
                precisions.append(precision)
            self.local_basis = torch.stack(bases, dim=0)
            self.local_precision = torch.stack(precisions, dim=0)

        self._stats_ready = True

    def _cosine(self, query, prototypes):
        q = F.normalize(query.float(), dim=-1)
        p = F.normalize(prototypes.float(), dim=-1)
        return q @ p.transpose(0, 1)

    def _global_pca_similarity(self, query, prototypes):
        if not self._stats_ready:
            raise RuntimeError("PCA metric requested before support statistics were fit.")
        q = query.float()
        p = prototypes.float()
        q_proj = q @ self.global_basis
        p_proj = p @ self.global_basis

        weighted_q = q_proj * self.global_precision
        inner = weighted_q @ p_proj.transpose(0, 1)
        inner = inner + self.metric_ridge * (q @ p.transpose(0, 1))

        q_norm = (weighted_q * q_proj).sum(dim=-1)
        q_norm = q_norm + self.metric_ridge * q.square().sum(dim=-1)

        weighted_p = p_proj * self.global_precision
        p_norm = (weighted_p * p_proj).sum(dim=-1)
        p_norm = p_norm + self.metric_ridge * p.square().sum(dim=-1)

        denom = torch.sqrt(
            q_norm.clamp_min(1e-12)[:, None]
            * p_norm.clamp_min(1e-12)[None, :]
        )
        return (inner / denom).clamp(-1.0, 1.0)

    def _local_pca_similarity(self, query, prototypes, class_indices):
        if not self._stats_ready:
            raise RuntimeError("Local PCA requested before support statistics were fit.")
        if class_indices is None:
            raise ValueError("local_pca requires prototype class indices.")

        q = query.float()
        p = prototypes.float()
        scores = []
        for proto_idx, class_idx in enumerate(class_indices):
            class_idx = int(class_idx)
            basis = self.local_basis[class_idx]
            precision = self.local_precision[class_idx]

            q_proj = q @ basis
            p_vec = p[proto_idx]
            p_proj = p_vec @ basis

            inner = (q_proj * precision) @ p_proj
            inner = inner + self.metric_ridge * (q @ p_vec)

            q_norm = (q_proj.square() * precision).sum(dim=-1)
            q_norm = q_norm + self.metric_ridge * q.square().sum(dim=-1)

            p_norm = (p_proj.square() * precision).sum()
            p_norm = p_norm + self.metric_ridge * p_vec.square().sum()

            denom = torch.sqrt(
                q_norm.clamp_min(1e-12) * p_norm.clamp_min(1e-12)
            )
            scores.append((inner / denom).clamp(-1.0, 1.0))
        return torch.stack(scores, dim=1)

    def _fisher_similarity(self, query, prototypes):
        if self.text_embed is None:
            raise RuntimeError(
                "fisher_riemannian requires frozen text classifier embeddings."
            )

        q = query.float()
        prototypes = prototypes.float()
        text = self.text_embed.to(q.device, dtype=q.dtype)

        q_logits = q @ text.transpose(0, 1)
        p_logits = prototypes @ text.transpose(0, 1)
        prob = torch.softmax(
            q_logits / max(self.fisher_temperature, 1e-6), dim=-1
        )

        eq = (prob * q_logits).sum(dim=-1)
        ep = prob @ p_logits.transpose(0, 1)
        eqp = (prob * q_logits) @ p_logits.transpose(0, 1)
        fisher_inner = eqp - eq[:, None] * ep

        q_var = (prob * q_logits.square()).sum(dim=-1) - eq.square()
        p_var = prob @ p_logits.square().transpose(0, 1) - ep.square()

        euclidean_inner = q @ prototypes.transpose(0, 1)
        inner = fisher_inner + self.metric_ridge * euclidean_inner

        q_norm = q_var + self.metric_ridge * q.square().sum(dim=-1)
        p_norm = p_var + self.metric_ridge * prototypes.square().sum(
            dim=-1
        )[None, :]

        denom = torch.sqrt(
            q_norm.clamp_min(1e-12)[:, None] * p_norm.clamp_min(1e-12)
        )
        return (inner / denom).clamp(-1.0, 1.0)

    def similarity(self, query, prototypes, class_indices=None):
        """Return pairwise similarities with shape [N_query, N_proto]."""
        if query.ndim == 1:
            query = query.unsqueeze(0)
        if prototypes.ndim == 1:
            prototypes = prototypes.unsqueeze(0)

        if self.metric == "cosine":
            return self._cosine(query, prototypes)
        if self.metric == "pca_mahalanobis":
            return self._global_pca_similarity(query, prototypes)
        if self.metric == "local_pca":
            return self._local_pca_similarity(
                query, prototypes, class_indices=class_indices
            )
        if self.metric == "fisher_riemannian":
            return self._fisher_similarity(query, prototypes)
        raise AssertionError(f"Unhandled semantic metric: {self.metric}")
