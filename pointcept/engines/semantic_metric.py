import torch
import torch.nn as nn
import torch.nn.functional as F


class FisherCompetitionCalibrator(nn.Module):
    """Shared residual calibrator for candidate-conditioned Fisher competition."""

    INPUT_DIM = 6

    def __init__(self, hidden_dim=16, bottleneck_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return self.net(features).squeeze(-1)


class SemanticMetricMatcher(nn.Module):
    """Unified similarity backend for GFS-VL pseudo-label calibration.

    Supported metrics:
      - cosine: original cosine similarity.
      - pca_mahalanobis: global support-set PCA precision metric.
      - local_pca: class-conditioned support-set PCA precision metrics.
      - fisher_riemannian: original query-conditioned Fisher pullback metric.
      - fisher_support: fixed support-conditioned Fisher geometry.
      - learnable_fisher: support-conditioned Fisher geometry with learned
        residual calibration of class competition.
    """

    VALID_METRICS = {
        "cosine",
        "pca_mahalanobis",
        "local_pca",
        "fisher_riemannian",
        "fisher_support",
        "learnable_fisher",
    }

    def __init__(
        self,
        metric="cosine",
        text_embed=None,
        pca_rank=32,
        pca_shrinkage=0.1,
        metric_ridge=1e-3,
        fisher_temperature=1.0,
        fisher_support_gamma=0.2,
        fisher_residual_scale=0.5,
        fisher_calibrator_hidden_dim=16,
        fisher_calibrator_bottleneck_dim=8,
        fisher_metric_temperature=0.1,
        fisher_margin=0.1,
        fisher_margin_loss_weight=0.5,
        fisher_preserve_loss_weight=0.1,
    ):
        super().__init__()
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
        self.fisher_support_gamma = float(fisher_support_gamma)
        self.fisher_residual_scale = float(fisher_residual_scale)
        self.fisher_metric_temperature = float(fisher_metric_temperature)
        self.fisher_margin = float(fisher_margin)
        self.fisher_margin_loss_weight = float(fisher_margin_loss_weight)
        self.fisher_preserve_loss_weight = float(fisher_preserve_loss_weight)

        if text_embed is None:
            self.register_buffer("text_embed", None, persistent=False)
        else:
            self.register_buffer(
                "text_embed", text_embed.detach().float(), persistent=False
            )

        self.calibrator = None
        if self.metric == "learnable_fisher":
            self.calibrator = FisherCompetitionCalibrator(
                hidden_dim=int(fisher_calibrator_hidden_dim),
                bottleneck_dim=int(fisher_calibrator_bottleneck_dim),
            )

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

    @property
    def is_learnable_fisher(self):
        return self.metric == "learnable_fisher"

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

    def _require_text_embed(self):
        if self.text_embed is None:
            raise RuntimeError(
                f"{self.metric} requires frozen text classifier embeddings."
            )

    def _fisher_similarity(self, query, prototypes):
        """Original fixed Fisher baseline; intentionally kept unchanged."""
        self._require_text_embed()
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

    def _resolve_text_class_indices(self, p_logits, text_class_indices):
        num_proto = p_logits.shape[0]
        if text_class_indices is None:
            return p_logits.argmax(dim=-1)
        indices = torch.as_tensor(
            text_class_indices, device=p_logits.device, dtype=torch.long
        )
        if indices.ndim != 1 or indices.numel() != num_proto:
            raise ValueError(
                "text_class_indices must contain one global text-class index "
                "per prototype."
            )
        if torch.any(indices < 0) or torch.any(indices >= p_logits.shape[1]):
            raise ValueError("text_class_indices contains an out-of-range index.")
        return indices

    def _candidate_fisher_similarity_from_prob(
        self, query, prototypes, q_logits, p_logits, prob
    ):
        q = query.float()
        p = prototypes.float()
        ql = q_logits[:, None, :]
        pl = p_logits[None, :, :]
        eq = (prob * ql).sum(dim=-1)
        ep = (prob * pl).sum(dim=-1)
        eqp = (prob * ql * pl).sum(dim=-1)
        fisher_inner = eqp - eq * ep
        q_var = (prob * ql.square()).sum(dim=-1) - eq.square()
        p_var = (prob * pl.square()).sum(dim=-1) - ep.square()
        inner = fisher_inner + self.metric_ridge * (q @ p.transpose(0, 1))
        q_norm = q_var + self.metric_ridge * q.square().sum(dim=-1)[:, None]
        p_norm = p_var + self.metric_ridge * p.square().sum(dim=-1)[None, :]
        denom = torch.sqrt(
            q_norm.clamp_min(1e-12) * p_norm.clamp_min(1e-12)
        )
        return (inner / denom).clamp(-1.0, 1.0)

    def _support_fisher_state(
        self,
        query,
        prototypes,
        text_class_indices=None,
        learnable=False,
    ):
        self._require_text_embed()
        q = query.float()
        p = prototypes.float()
        text = self.text_embed.to(q.device, dtype=q.dtype)
        q_logits = q @ text.transpose(0, 1)
        p_logits = p @ text.transpose(0, 1)
        q_all = q_logits[:, None, :]
        p_all = p_logits[None, :, :]
        conditioned_logits = q_all + self.fisher_support_gamma * p_all
        residual = None

        if learnable:
            if self.calibrator is None:
                raise RuntimeError(
                    "learnable_fisher calibrator is not initialized."
                )
            class_idx = self._resolve_text_class_indices(
                p_logits, text_class_indices
            )
            q_c = q_logits[:, class_idx]
            a_cc = p_logits.gather(1, class_idx[:, None]).squeeze(1)
            q_j = q_all.expand(-1, p.shape[0], -1)
            a_cj = p_all.expand(q.shape[0], -1, -1)
            q_c_exp = q_c[:, :, None].expand_as(q_j)
            a_cc_exp = a_cc[None, :, None].expand_as(q_j)
            features = torch.stack(
                [
                    q_j,
                    q_c_exp,
                    a_cj,
                    a_cc_exp,
                    q_c_exp - q_j,
                    a_cc_exp - a_cj,
                ],
                dim=-1,
            )
            residual = self.fisher_residual_scale * torch.tanh(
                self.calibrator(features)
            )
            conditioned_logits = conditioned_logits + residual

        prob = torch.softmax(
            conditioned_logits / max(self.fisher_temperature, 1e-6), dim=-1
        )
        base_prob = torch.softmax(
            q_logits / max(self.fisher_temperature, 1e-6), dim=-1
        )
        return q_logits, p_logits, prob, base_prob, residual

    def _support_fisher_similarity(
        self, query, prototypes, text_class_indices=None
    ):
        if abs(self.fisher_support_gamma) < 1e-12:
            return self._fisher_similarity(query, prototypes)
        q_logits, p_logits, prob, _, _ = self._support_fisher_state(
            query,
            prototypes,
            text_class_indices=text_class_indices,
            learnable=False,
        )
        return self._candidate_fisher_similarity_from_prob(
            query, prototypes, q_logits, p_logits, prob
        )

    def _learnable_fisher_similarity(
        self,
        query,
        prototypes,
        text_class_indices=None,
        return_details=False,
    ):
        q_logits, p_logits, prob, base_prob, residual = (
            self._support_fisher_state(
                query,
                prototypes,
                text_class_indices=text_class_indices,
                learnable=True,
            )
        )
        scores = self._candidate_fisher_similarity_from_prob(
            query, prototypes, q_logits, p_logits, prob
        )
        if return_details:
            return scores, {
                "prob": prob,
                "base_prob": base_prob,
                "residual": residual,
            }
        return scores

    def learnable_fisher_loss(
        self,
        query,
        prototypes,
        targets,
        text_class_indices,
    ):
        """Step-4 objective; episodic sampling/optimizer belong to Step 5."""
        if self.metric != "learnable_fisher":
            raise RuntimeError(
                "learnable_fisher_loss requires metric='learnable_fisher'."
            )
        if query.ndim == 1:
            query = query.unsqueeze(0)
        if prototypes.ndim == 1:
            prototypes = prototypes.unsqueeze(0)
        targets = targets.to(query.device, dtype=torch.long)
        scores, details = self._learnable_fisher_similarity(
            query,
            prototypes,
            text_class_indices=text_class_indices,
            return_details=True,
        )
        cls_loss = F.cross_entropy(
            scores / max(self.fisher_metric_temperature, 1e-6), targets
        )
        pos = scores.gather(1, targets[:, None]).squeeze(1)
        wrong_scores = scores.masked_fill(
            F.one_hot(targets, num_classes=scores.shape[1]).bool(),
            float("-inf"),
        )
        top_wrong = wrong_scores.max(dim=1).values
        margin_loss = F.relu(
            self.fisher_margin - pos + top_wrong
        ).mean()

        learned_prob = details["prob"].clamp_min(1e-8)
        base_prob = details["base_prob"][:, None, :].expand_as(
            learned_prob
        ).clamp_min(1e-8)
        preserve_loss = (
            learned_prob
            * (learned_prob.log() - base_prob.log())
        ).sum(dim=-1).mean()

        total_loss = (
            cls_loss
            + self.fisher_margin_loss_weight * margin_loss
            + self.fisher_preserve_loss_weight * preserve_loss
        )
        residual = details["residual"]
        return {
            "loss": total_loss,
            "cls_loss": cls_loss,
            "margin_loss": margin_loss,
            "preserve_loss": preserve_loss,
            "mean_residual_abs": residual.abs().mean(),
            "mean_gt_topwrong_margin": (pos - top_wrong).mean(),
            "scores": scores,
        }

    def similarity(
        self,
        query,
        prototypes,
        class_indices=None,
        text_class_indices=None,
    ):
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
        if self.metric == "fisher_support":
            return self._support_fisher_similarity(
                query,
                prototypes,
                text_class_indices=text_class_indices,
            )
        if self.metric == "learnable_fisher":
            return self._learnable_fisher_similarity(
                query,
                prototypes,
                text_class_indices=text_class_indices,
            )
        raise AssertionError(f"Unhandled semantic metric: {self.metric}")
