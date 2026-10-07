"""Joint Step-1 CDC base geometry learning.

This variant keeps the original GFS-VL base segmentation task as an anchor,
warms up CDC/text alignment with a frozen PTv3 backbone, then unfreezes PTv3
with a smaller learning rate. Only base-class labels/text names are allowed.
"""

import math

import torch
import torch.nn as nn

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS
from .cdc_base_geometry import CDCBaseGeometrySegmentor


@MODELS.register_module()
class CDCBaseGeometryJointSegmentor(CDCBaseGeometrySegmentor):
    """CDC base pretraining with freeze-then-unfreeze PTv3 optimization."""

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        freeze_seg_head=True,
        backbone_freeze_epochs=5,
        seg_loss_weight=1.0,
        cdc=None,
        text_constraint=None,
    ):
        # ``freeze_backbone`` here means permanently frozen. For the intended
        # Step-1 schedule leave it False and use backbone_freeze_epochs > 0.
        super().__init__(
            num_classes=num_classes,
            backbone_out_channels=backbone_out_channels,
            backbone=backbone,
            criteria=criteria,
            freeze_backbone=False,
            freeze_seg_head=freeze_seg_head,
            cdc=cdc,
            text_constraint=text_constraint,
        )
        self.permanent_freeze_backbone = bool(freeze_backbone)
        self.backbone_freeze_epochs = int(backbone_freeze_epochs)
        self.seg_loss_weight = float(seg_loss_weight)
        self.criteria = build_criteria(criteria)

        self._backbone_is_frozen = True
        self._set_backbone_frozen(True)

    def _set_backbone_frozen(self, frozen):
        frozen = bool(frozen)
        self._backbone_is_frozen = frozen
        self.backbone.requires_grad_(not frozen)
        if frozen:
            self.backbone.eval()
        elif self.training:
            self.backbone.train()

    def set_training_epoch(self, epoch):
        self.current_epoch = int(epoch)
        should_freeze = self.permanent_freeze_backbone or (
            self.current_epoch < self.backbone_freeze_epochs
        )
        if should_freeze != self._backbone_is_frozen:
            self._set_backbone_frozen(should_freeze)

    def train(self, mode=True):
        # Bypass the parent permanent-freeze policy; this class controls PTv3
        # dynamically via ``_backbone_is_frozen``.
        nn.Module.train(self, mode)
        if self._backbone_is_frozen:
            self.backbone.eval()
        if self.freeze_seg_head:
            self.seg_head.eval()
        return self

    def _check_base_only_labels(self, labels):
        valid = labels >= 0
        if valid.any():
            max_label = int(labels[valid].max().item())
            min_label = int(labels[valid].min().item())
            if min_label < 0 or max_label >= self.num_classes:
                raise RuntimeError(
                    "Step-1 label leakage detected: valid labels must be in "
                    f"[0, {self.num_classes - 1}], got [{min_label}, {max_label}]."
                )

    def forward(self, input_dict):
        if not bool(self._text_ready.item()):
            raise RuntimeError(
                "Frozen base-class text embeddings are not initialized. "
                "Use CDCBaseGeometryTrainer for Step-1 training."
            )

        labels = input_dict["segment"].long()
        self._check_base_only_labels(labels)

        if self._backbone_is_frozen:
            with torch.no_grad():
                point = self.backbone(Point(input_dict))
        else:
            point = self.backbone(Point(input_dict))
        feature = point.feat if isinstance(point, Point) else point

        indices = self._balanced_indices(labels, self.cdc_max_points)
        if indices.numel() == 0:
            raise RuntimeError("The current batch contains no valid base-class points.")

        # Keep CDC and semantic geometry calculations in fp32 for stability.
        with torch.cuda.amp.autocast(enabled=False):
            feature_fp32 = feature.float()

            # The original base segmentation objective anchors PTv3 after it is
            # unfrozen. The base classifier itself remains frozen.
            seg_logits = self.seg_head(feature_fp32)
            seg_loss = self.criteria(seg_logits, labels)

            clean = feature_fp32[indices]
            sampled_labels = labels[indices]

            relative_h = self._sample_relative_h(
                clean.shape[0], clean.device, clean.dtype
            )
            feature_rms = clean.square().mean(dim=-1).add(1e-8).sqrt()
            h_abs = relative_h * feature_rms
            noise = torch.randn_like(clean) * h_abs.unsqueeze(-1)
            noisy = clean + noise
            diff = -noise / (
                math.sqrt(2.0) * h_abs.unsqueeze(-1).clamp_min(1e-8)
            )

            factor = self.cdc_head(noisy, h_abs)
            cdc_loss = self._low_rank_metric_matching_loss(
                factor, diff, self.cdc_tikhonov
            )
            text_align_loss, text_acc = self._semantic_alignment_loss(
                clean, sampled_labels
            )

            if self.current_epoch >= self.text_warmup_epochs:
                (
                    text_direction_loss,
                    text_virtual_margin,
                    text_virtual_valid_rate,
                ) = self._semantic_direction_loss(clean, sampled_labels)
                direction_weight = self.text_direction_weight
            else:
                text_direction_loss = clean.new_zeros(())
                text_virtual_margin = clean.new_zeros(())
                text_virtual_valid_rate = clean.new_zeros(())
                direction_weight = 0.0

            loss = (
                self.seg_loss_weight * seg_loss
                + self.cdc_loss_weight * cdc_loss
                + self.text_align_weight * text_align_loss
                + direction_weight * text_direction_loss
            )
            factor_energy = (
                factor.square().sum(dim=(1, 2)) / self.feature_dim
            ).mean()

        return {
            "loss": loss,
            "seg_loss": seg_loss.detach(),
            "cdc_loss": cdc_loss.detach(),
            "text_align_loss": text_align_loss.detach(),
            "text_direction_loss": text_direction_loss.detach(),
            "text_acc": text_acc.detach(),
            "text_virtual_margin": text_virtual_margin.detach(),
            "text_virtual_valid_rate": text_virtual_valid_rate.detach(),
            "cdc_factor_energy": factor_energy.detach(),
            "relative_h": relative_h.mean().detach(),
            "backbone_frozen": clean.new_tensor(float(self._backbone_is_frozen)),
        }
