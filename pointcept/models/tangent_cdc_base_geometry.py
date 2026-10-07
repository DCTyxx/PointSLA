"""Tangent-supervised CDC for Step-1 base geometry learning.

This diagnostic variant keeps PTv3 frozen and replaces isotropic Gaussian
metric matching with direct supervision from same-class local feature tangents
estimated on the current Base-only training batch. A discrimination hinge
encourages the CDC subspace to explain intra-class residual energy more than
inter-class residual energy. The semantic projector is still trained for
Base-only CLIP alignment, but text-direction regularization is deliberately
excluded from the Tangent-CDC objective.
"""

import torch
import torch.nn.functional as F

from pointcept.models.utils.structure import Point
from .builder import MODELS
from .cdc_base_geometry import CDCBaseGeometrySegmentor


@MODELS.register_module()
class TangentCDCBaseGeometrySegmentor(CDCBaseGeometrySegmentor):
    """Frozen-PTv3 Tangent-CDC diagnostic model."""

    def __init__(self, tangent=None, **kwargs):
        super().__init__(**kwargs)
        tangent = {} if tangent is None else dict(tangent)

        self.tangent_knn = int(tangent.get("knn", 32))
        self.tangent_inter_knn = int(tangent.get("inter_knn", 32))
        self.tangent_max_bank_points = int(tangent.get("max_bank_points", 4096))
        self.tangent_max_anchors = int(tangent.get("max_anchors", 256))
        self.tangent_loss_weight = float(tangent.get("loss_weight", 1.0))
        self.tangent_disc_weight = float(tangent.get("disc_loss_weight", 0.1))
        self.tangent_disc_margin = float(tangent.get("disc_margin", 0.05))
        self.tangent_projection_eps = float(tangent.get("projection_eps", 1e-4))

        if not self.freeze_backbone:
            raise ValueError("Tangent-CDC Step 1 requires a frozen PTv3 backbone.")
        if self.tangent_knn < self.cdc_rank:
            raise ValueError(
                "Tangent-CDC requires tangent.knn >= CDC rank, got "
                f"knn={self.tangent_knn}, rank={self.cdc_rank}."
            )
        if self.tangent_max_anchors <= 0:
            raise ValueError("tangent.max_anchors must be positive.")
        if self.tangent_max_bank_points <= self.tangent_knn:
            raise ValueError(
                "tangent.max_bank_points must exceed tangent.knn."
            )

    @staticmethod
    def _balanced_unique_indices(labels, max_points):
        """Class-balanced subsampling without replacement."""
        valid = torch.nonzero(labels >= 0, as_tuple=False).flatten()
        if valid.numel() == 0 or max_points <= 0 or valid.numel() <= max_points:
            return valid

        valid_labels = labels[valid]
        classes = torch.unique(valid_labels)
        per_class = max(1, max_points // max(1, classes.numel()))
        selected = []
        for class_id in classes:
            idx = valid[valid_labels == class_id]
            if idx.numel() > per_class:
                perm = torch.randperm(idx.numel(), device=idx.device)[:per_class]
                idx = idx[perm]
            selected.append(idx)
        return torch.cat(selected, dim=0)

    @staticmethod
    def _balanced_tangent_anchors(labels, max_anchors, min_class_points):
        """Choose anchors only from classes with enough same-class neighbors."""
        classes = torch.unique(labels)
        eligible = []
        for class_id in classes:
            idx = torch.nonzero(labels == class_id, as_tuple=False).flatten()
            if idx.numel() >= min_class_points:
                eligible.append(idx)

        if len(eligible) == 0:
            return labels.new_empty((0,), dtype=torch.long)

        per_class = max(1, max_anchors // len(eligible))
        selected = []
        for idx in eligible:
            if idx.numel() > per_class:
                perm = torch.randperm(idx.numel(), device=idx.device)[:per_class]
                idx = idx[perm]
            selected.append(idx)

        selected = torch.cat(selected, dim=0)
        if selected.numel() > max_anchors:
            perm = torch.randperm(selected.numel(), device=selected.device)[
                :max_anchors
            ]
            selected = selected[perm]
        return selected

    @staticmethod
    def _projected_energy_ratio(factor, gram, residual):
        """Fraction of residual energy explained by span(factor)."""
        # factor: [A, D, r], residual: [A, K, D]
        coeff = torch.einsum("adr,akd->ark", factor, residual)
        solved = torch.linalg.solve(gram, coeff)
        projected_energy = (coeff * solved).sum(dim=(1, 2))
        total_energy = residual.square().sum(dim=(1, 2)).clamp_min(1e-12)
        return projected_energy / total_energy

    def _tangent_objective(self, clean, labels):
        """Directly supervise the CDC span with same-class empirical tangents."""
        bank_idx = self._balanced_unique_indices(
            labels, self.tangent_max_bank_points
        )
        bank = clean[bank_idx]
        bank_labels = labels[bank_idx]

        anchor_idx = self._balanced_tangent_anchors(
            bank_labels,
            self.tangent_max_anchors,
            self.tangent_knn + 1,
        )
        if anchor_idx.numel() == 0:
            raise RuntimeError(
                "Tangent-CDC found no class with enough training points for "
                f"knn={self.tangent_knn}. Increase max_bank_points/batch size "
                "or reduce tangent.knn/rank."
            )

        anchors = bank[anchor_idx]
        anchor_labels = bank_labels[anchor_idx]
        num_anchors = anchors.shape[0]

        # KNN/SVD define a detached empirical training target. PTv3 is frozen,
        # so the only gradient path in this objective is through the CDC head.
        with torch.no_grad():
            dist = torch.cdist(anchors, bank, p=2).square()
            same = anchor_labels[:, None].eq(bank_labels[None, :])
            row = torch.arange(num_anchors, device=clean.device)
            same[row, anchor_idx] = False

            intra_dist = dist.masked_fill(~same, float("inf"))
            nearest_intra = torch.topk(
                intra_dist,
                k=self.tangent_knn,
                dim=1,
                largest=False,
            ).indices
            intra_residual = bank[nearest_intra] - anchors[:, None, :]

            # Empirical local tangent Q_emp from same-class KNN residuals.
            _, _, vh = torch.linalg.svd(
                intra_residual.detach(), full_matrices=False
            )
            q_emp = vh[:, : self.cdc_rank, :].transpose(1, 2).contiguous()

            different = ~anchor_labels[:, None].eq(bank_labels[None, :])
            min_different = int(different.sum(dim=1).min().item())
            if min_different <= 0:
                raise RuntimeError(
                    "Tangent-CDC discrimination loss requires at least two "
                    "base classes in the current training batch."
                )
            k_inter = min(self.tangent_inter_knn, min_different)
            inter_dist = dist.masked_fill(~different, float("inf"))
            nearest_inter = torch.topk(
                inter_dist,
                k=k_inter,
                dim=1,
                largest=False,
            ).indices
            inter_residual = bank[nearest_inter] - anchors[:, None, :]

        # Evaluate CDC at the same fixed bandwidth used by held-out geometry
        # diagnostics, so training/evaluation differ by data split, not scale.
        relative_h = torch.full(
            (num_anchors,),
            self.cdc_probe_h,
            device=clean.device,
            dtype=clean.dtype,
        )
        feature_rms = anchors.square().mean(dim=-1).add(1e-8).sqrt()
        factor = self.cdc_head(anchors, relative_h * feature_rms)

        # Differentiable subspace projector:
        #   P_U = U (U^T U + eps I)^(-1) U^T.
        # This makes the objective depend on the CDC span rather than a
        # particular basis rotation and avoids differentiating through SVD/QR of U.
        gram = torch.einsum("adr,adq->arq", factor, factor)
        eye = torch.eye(
            self.cdc_rank, device=clean.device, dtype=clean.dtype
        ).unsqueeze(0)
        gram = gram + self.tangent_projection_eps * eye

        # TAS-style training objective:
        # 1 - (1/r) tr(Q_emp^T P_U Q_emp).
        u_t_q = torch.einsum("adr,adq->arq", factor, q_emp)
        solved_q = torch.linalg.solve(gram, u_t_q)
        alignment = (u_t_q * solved_q).sum(dim=(1, 2)) / self.cdc_rank
        alignment = alignment.clamp(min=0.0, max=1.0)
        tangent_loss = 1.0 - alignment.mean()

        # Geometry discrimination hinge:
        # max(0, margin - EVR_intra + EVR_inter).
        intra_evr = self._projected_energy_ratio(
            factor, gram, intra_residual.detach()
        )
        inter_evr = self._projected_energy_ratio(
            factor, gram, inter_residual.detach()
        )
        gap = intra_evr - inter_evr
        disc_loss = F.relu(self.tangent_disc_margin - gap).mean()

        return (
            tangent_loss,
            disc_loss,
            alignment.mean().detach(),
            gap.mean().detach(),
            factor,
        )

    def forward(self, input_dict):
        if not bool(self._text_ready.item()):
            raise RuntimeError(
                "Frozen text embeddings are not initialized. "
                "Use CDCBaseGeometryTrainer for Step-1 training."
            )

        # PTv3 and the original segmentation head remain frozen throughout.
        with torch.no_grad():
            point = self.backbone(Point(input_dict))
            feature = point.feat if isinstance(point, Point) else point

        labels = input_dict["segment"].long()
        valid = labels >= 0
        if not valid.any():
            raise RuntimeError(
                "The current batch contains no valid base-class points."
            )
        if int(labels[valid].max().item()) >= self.num_classes:
            raise RuntimeError(
                "Tangent-CDC Step-1 label leakage detected: encountered a "
                "non-base class label."
            )

        # Keep all geometry operations in fp32. The frozen-r24 config also turns
        # off outer AMP because eval-mode spconv autotuning is unstable in fp16
        # on the current server build.
        with torch.cuda.amp.autocast(enabled=False):
            clean = feature.detach().float()
            labels = labels.to(device=clean.device)

            (
                tangent_loss,
                disc_loss,
                train_tas,
                train_gdg,
                factor,
            ) = self._tangent_objective(clean, labels)

            # The projector is learned only so held-out SVR has a stable Base-only
            # CLIP semantic coordinate system. It does not constrain CDC here.
            text_idx = self._balanced_unique_indices(
                labels, self.text_max_points
            )
            text_align_loss, text_acc = self._semantic_alignment_loss(
                clean[text_idx], labels[text_idx]
            )

            loss = (
                self.tangent_loss_weight * tangent_loss
                + self.tangent_disc_weight * disc_loss
                + self.text_align_weight * text_align_loss
            )
            factor_energy = (
                factor.square().sum(dim=(1, 2)) / self.feature_dim
            ).mean()

        return {
            "loss": loss,
            "tangent_loss": tangent_loss.detach(),
            "discrimination_loss": disc_loss.detach(),
            "train_tas": train_tas,
            "train_gdg": train_gdg,
            "text_align_loss": text_align_loss.detach(),
            "text_acc": text_acc.detach(),
            "cdc_factor_energy": factor_energy.detach(),
        }
