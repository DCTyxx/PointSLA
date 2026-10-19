"""Merged Stage-1 + Stage-2 Phase-2 geometry learning.

This module removes the separate Base-only geometry pretraining stage. It starts
from an already task-adapted GFS-VL registration checkpoint, freezes PTv3 and the
Base/Novel segmentation heads, and trains the geometry stack from scratch in the
final task feature space:

- CDC tangent geometry: Base GT + trusted K-shot Novel GT;
- semantic projector: Base GT only;
- conservative semantic safe-boundary / radius head: Base anchors only.

Thus Novel labels can reshape the local tangent field, while language-based
semantic boundary estimation never uses Novel labels or Novel text prototypes.
"""

import torch
import torch.nn as nn

from pointcept.models.utils.structure import Point
from .builder import MODELS
from .stage2_phase2_cdc_adapt import Stage2Phase2CDCAdaptSegmentor


@MODELS.register_module()
class Stage2MergedGeometrySegmentor(Stage2Phase2CDCAdaptSegmentor):
    """Frozen task representation + jointly learned geometry/semantic boundary."""

    def __init__(self, **kwargs):
        # Build the same hierarchy used by Phase 2 / Phase 3 so the resulting
        # checkpoint remains strictly compatible with Stage2Phase3CDCAugSegmentor.
        super().__init__(**kwargs)

        # The task-adapted representation/classifiers are fixed.
        self.backbone.requires_grad_(False)
        self.seg_head.requires_grad_(False)
        self.novel_seg_head.requires_grad_(False)

        # Stage-1 and old Phase-2 are merged here: all geometry-side modules are
        # trained directly in the final task feature space.
        self.cdc_head.requires_grad_(True)
        self.semantic_projector.requires_grad_(True)
        self.safe_radius_head.requires_grad_(True)

        self.freeze_backbone = True
        self.freeze_seg_head = True
        self._assert_merged_trainability()

    def _assert_merged_trainability(self):
        frozen_modules = (
            ("backbone", self.backbone),
            ("seg_head", self.seg_head),
            ("novel_seg_head", self.novel_seg_head),
        )
        for name, module in frozen_modules:
            if any(p.requires_grad for p in module.parameters()):
                raise RuntimeError(f"{name} must be frozen in merged geometry stage")

        trainable_modules = (
            ("cdc_head", self.cdc_head),
            ("semantic_projector", self.semantic_projector),
            ("safe_radius_head", self.safe_radius_head),
        )
        for name, module in trainable_modules:
            if not all(p.requires_grad for p in module.parameters()):
                raise RuntimeError(f"every {name} parameter must be trainable")

        allowed = ("cdc_head.", "semantic_projector.", "safe_radius_head.")
        bad = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.startswith(allowed)
        ]
        if bad:
            raise RuntimeError(
                "Only CDC/projector/radius parameters may train in merged stage; "
                f"got {bad}"
            )

    def train(self, mode=True):
        # Bypass Phase-2 train(), which freezes projector/radius.
        nn.Module.train(self, mode)
        self.backbone.eval()
        self.seg_head.eval()
        self.novel_seg_head.eval()
        self.cdc_head.train(mode)
        self.semantic_projector.train(mode)
        self.safe_radius_head.train(mode)
        self._assert_merged_trainability()
        return self

    def _merged_tangent_objective(self, clean, labels):
        """QR Tangent-CDC on trusted Base + Novel GT, with split diagnostics."""
        bank_idx = self._balanced_unique_indices(labels, self.tangent_max_bank_points)
        bank = clean[bank_idx]
        bank_labels = labels[bank_idx]

        anchor_idx = self._balanced_tangent_anchors(
            bank_labels,
            self.tangent_max_anchors,
            self.tangent_knn + 1,
        )
        if anchor_idx.numel() == 0:
            raise RuntimeError(
                "Merged geometry stage found no class with enough trusted points "
                f"for tangent KNN; knn={self.tangent_knn}."
            )

        anchors = bank[anchor_idx]
        anchor_labels = bank_labels[anchor_idx]
        num_anchors = anchors.shape[0]

        with torch.no_grad():
            dist = torch.cdist(anchors, bank, p=2).square()
            same = anchor_labels[:, None].eq(bank_labels[None, :])
            row = torch.arange(num_anchors, device=clean.device)
            same[row, anchor_idx] = False
            nearest_intra = torch.topk(
                dist.masked_fill(~same, float("inf")),
                k=self.tangent_knn,
                dim=1,
                largest=False,
            ).indices
            intra_residual = bank[nearest_intra] - anchors[:, None, :]

            _, _, vh = torch.linalg.svd(
                intra_residual.detach(), full_matrices=False
            )
            q_emp = vh[:, : self.cdc_rank, :].transpose(1, 2).contiguous()

            different = ~anchor_labels[:, None].eq(bank_labels[None, :])
            min_different = int(different.sum(dim=1).min().item())
            if min_different <= 0:
                raise RuntimeError(
                    "Merged geometry discrimination requires at least two trusted "
                    "classes in the current feature bank."
                )
            k_inter = min(self.tangent_inter_knn, min_different)
            nearest_inter = torch.topk(
                dist.masked_fill(~different, float("inf")),
                k=k_inter,
                dim=1,
                largest=False,
            ).indices
            inter_residual = bank[nearest_inter] - anchors[:, None, :]

        relative_h = torch.full(
            (num_anchors,),
            self.cdc_probe_h,
            device=clean.device,
            dtype=clean.dtype,
        )
        feature_rms = anchors.square().mean(dim=-1).add(1e-8).sqrt()
        factor = self.cdc_head(anchors, relative_h * feature_rms)
        q = self._orthonormalize_factor(factor)

        overlap = torch.einsum("adr,adq->arq", q, q_emp)
        alignment = overlap.square().sum(dim=(1, 2)) / self.cdc_rank
        tangent_loss = 1.0 - alignment.mean()

        intra_evr = self._orthonormal_projected_energy_ratio(
            q, intra_residual.detach()
        )
        inter_evr = self._orthonormal_projected_energy_ratio(
            q, inter_residual.detach()
        )
        gap = intra_evr - inter_evr
        disc_loss = torch.relu(self.tangent_disc_margin - gap).mean()

        base_mask = anchor_labels < self.num_base_classes
        novel_mask = ~base_mask

        return dict(
            tangent_loss=tangent_loss,
            disc_loss=disc_loss,
            train_tas=alignment.mean().detach(),
            train_gdg=gap.mean().detach(),
            base_tas=self._masked_mean_or_zero(alignment.detach(), base_mask),
            base_gdg=self._masked_mean_or_zero(gap.detach(), base_mask),
            novel_tas=self._masked_mean_or_zero(alignment.detach(), novel_mask),
            novel_gdg=self._masked_mean_or_zero(gap.detach(), novel_mask),
            num_base_anchors=base_mask.sum().to(dtype=clean.dtype),
            num_novel_anchors=novel_mask.sum().to(dtype=clean.dtype),
            factor=factor,
            q=q,
            anchors=anchors,
            anchor_labels=anchor_labels,
        )

    @staticmethod
    def _zero_radius_metrics(zero):
        return dict(
            radius_loss=zero,
            regression_loss=zero,
            overshoot_loss=zero,
            predicted_svr=zero,
            oracle_svr=zero,
            mean_safe_radius=zero,
            oracle_mean_safe_radius=zero,
            radius_mae=zero,
            radius_ratio=zero,
        )

    def forward(self, input_dict):
        if not bool(self._text_ready.item()):
            raise RuntimeError(
                "Base-only frozen text prototypes are not initialized. Use "
                "MergedStage1Phase2GeometryTrainer."
            )

        # PTv3 is already task-adapted by the official GFS-VL registration
        # checkpoint and must remain fixed while geometry is learned.
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                point = self.backbone(Point(input_dict))
                feature = point.feat if isinstance(point, Point) else point

        labels = input_dict["segment"].long().to(feature.device)
        valid = labels >= 0
        if not bool(valid.any().item()):
            raise RuntimeError("Merged geometry batch contains no trusted GT points")

        total_classes = self.num_base_classes + self.num_novel_classes
        if int(labels[valid].max().item()) >= total_classes:
            raise RuntimeError(
                "Trusted label exceeds Base+Novel class range in merged stage: "
                f"max={int(labels[valid].max().item())}, total={total_classes}"
            )

        with torch.cuda.amp.autocast(enabled=False):
            clean = feature.detach().float()[valid]
            clean_labels = labels[valid]

            # Geometry uses all trusted GT: Base + inserted K-shot Novel.
            geometry = self._merged_tangent_objective(clean, clean_labels)

            # Language alignment is Base-only because only Base text prototypes
            # are injected. Novel language features never enter this stage.
            base_point_mask = clean_labels < self.num_base_classes
            if not bool(base_point_mask.any().item()):
                raise RuntimeError("Merged geometry batch contains no Base GT points")
            base_clean = clean[base_point_mask]
            base_labels = clean_labels[base_point_mask]
            text_idx = self._balanced_unique_indices(base_labels, self.text_max_points)
            text_align_loss, text_acc = self._semantic_alignment_loss(
                base_clean[text_idx], base_labels[text_idx]
            )

            # Conservative safe-boundary supervision is STRICTLY Base-only.
            # Novel anchors participate in tangent learning above but are removed
            # before any CLIP semantic boundary or radius-head computation.
            base_anchor_mask = geometry["anchor_labels"] < self.num_base_classes
            radius_active = self.current_epoch >= self.safe_radius_warmup_epochs
            if radius_active and bool(base_anchor_mask.any().item()):
                radius = self._boundary_radius_objective(
                    geometry["anchors"][base_anchor_mask],
                    geometry["anchor_labels"][base_anchor_mask],
                    geometry["q"][base_anchor_mask],
                )
                radius_loss = radius["radius_loss"]
            else:
                zero = clean.new_zeros(())
                radius = self._zero_radius_metrics(zero)
                # Keep all radius parameters visible to DDP during warm-up or a
                # rare crop with no eligible Base tangent anchor.
                radius_loss = next(self.safe_radius_head.parameters()).sum() * 0.0

            loss = (
                self.tangent_loss_weight * geometry["tangent_loss"]
                + self.tangent_disc_weight * geometry["disc_loss"]
                + self.text_align_weight * text_align_loss
                + radius_loss
            )

            factor_energy = (
                geometry["factor"].square().sum(dim=(1, 2)) / self.feature_dim
            ).mean()
            qtq = torch.einsum("adr,adq->arq", geometry["q"], geometry["q"])
            eye = torch.eye(
                self.cdc_rank, device=clean.device, dtype=clean.dtype
            )[None]
            basis_orth_error = (qtq - eye).square().mean()

        return {
            "loss": loss,
            "tangent_loss": geometry["tangent_loss"].detach(),
            "discrimination_loss": geometry["disc_loss"].detach(),
            "train_tas": geometry["train_tas"],
            "train_gdg": geometry["train_gdg"],
            "base_tas": geometry["base_tas"],
            "base_gdg": geometry["base_gdg"],
            "novel_tas": geometry["novel_tas"],
            "novel_gdg": geometry["novel_gdg"],
            "num_base_anchors": geometry["num_base_anchors"],
            "num_novel_anchors": geometry["num_novel_anchors"],
            "base_boundary_anchors": base_anchor_mask.sum().to(dtype=clean.dtype),
            "text_align_loss": text_align_loss.detach(),
            "text_acc": text_acc.detach(),
            "radius_boundary_loss": radius["radius_loss"].detach(),
            "radius_regression_loss": radius["regression_loss"].detach(),
            "radius_overshoot_loss": radius["overshoot_loss"].detach(),
            "predicted_svr": radius["predicted_svr"].detach(),
            "oracle_svr": radius["oracle_svr"].detach(),
            "mean_safe_radius": radius["mean_safe_radius"].detach(),
            "oracle_mean_safe_radius": radius["oracle_mean_safe_radius"].detach(),
            "radius_mae": radius["radius_mae"].detach(),
            "radius_ratio": radius["radius_ratio"].detach(),
            "radius_active": clean.new_tensor(float(radius_active)),
            "basis_orth_error": basis_orth_error.detach(),
            "cdc_factor_energy": factor_energy.detach(),
        }
