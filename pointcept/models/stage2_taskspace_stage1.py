"""Task-space Stage-1 geometry learning after official GFS-VL adaptation.

This stage reintroduces the original Base-only Stage-1 BEFORE the original
Stage-2 Phase-2 CDC adaptation.  It starts from one official registration
checkpoint (already task-adapted by GFS-VL), freezes the task representation and
both classifiers, and learns the complete geometry stack using Base data only:

- PTv3 backbone: frozen;
- Base segmentation head: frozen and retained from the official checkpoint;
- Novel segmentation head: frozen and retained only for checkpoint continuity;
- CDC tangent geometry: trainable, Base-only;
- semantic projector: trainable, Base-only;
- conservative safe-radius head: trainable, Base-only.

The resulting checkpoint has the same parameter hierarchy as Stage-2 Phase-2,
so the existing strict Phase-1 -> Phase-2 loader can restore it exactly before
CDC is fine-tuned with Base + K-shot Novel GT.
"""

import torch.nn as nn

from .builder import MODELS
from .conservative_boundary_semantic_tangent_cdc_base_geometry import (
    ConservativeBoundarySemanticTangentCDCBaseGeometrySegmentor,
)


@MODELS.register_module()
class Stage2TaskSpaceStage1Segmentor(
    ConservativeBoundarySemanticTangentCDCBaseGeometrySegmentor
):
    """Base-only Stage-1 on the frozen task-adapted GFS-VL feature space."""

    def __init__(
        self,
        num_base_classes=None,
        num_novel_classes=0,
        num_classes=None,
        backbone_out_channels=64,
        **kwargs,
    ):
        # Original Stage-1 configs provide ``num_classes``; registration configs
        # use ``num_base_classes``.  Accept both so this model can directly inherit
        # the validated Stage-1 hyperparameters/dataset protocol.
        if num_base_classes is None:
            if num_classes is None:
                raise ValueError(
                    "Stage2TaskSpaceStage1Segmentor requires num_base_classes "
                    "or num_classes."
                )
            num_base_classes = int(num_classes)
        num_base_classes = int(num_base_classes)
        num_novel_classes = int(num_novel_classes)
        if num_novel_classes <= 0:
            raise ValueError("num_novel_classes must be positive")

        super().__init__(
            num_classes=num_base_classes,
            backbone_out_channels=backbone_out_channels,
            freeze_backbone=True,
            freeze_seg_head=True,
            **kwargs,
        )

        self.num_base_classes = num_base_classes
        self.num_novel_classes = num_novel_classes

        # Preserve the official Novel classifier in the state dict even though
        # Stage-1 itself never uses Novel labels or Novel logits.  This makes the
        # Stage-1 checkpoint strictly compatible with the existing Phase-2 model.
        self.novel_seg_head = nn.Linear(
            backbone_out_channels,
            self.num_novel_classes,
            bias=False,
        )

        # Task representation/classifiers are frozen throughout this Base-only
        # geometry stage.  The entire geometry stack is learned from scratch.
        self.backbone.requires_grad_(False)
        self.seg_head.requires_grad_(False)
        self.novel_seg_head.requires_grad_(False)
        self.cdc_head.requires_grad_(True)
        self.semantic_projector.requires_grad_(True)
        self.safe_radius_head.requires_grad_(True)

        self.freeze_backbone = True
        self.freeze_seg_head = True
        self._assert_task_stage1_trainability()

    def _assert_task_stage1_trainability(self):
        frozen_modules = (
            ("backbone", self.backbone),
            ("seg_head", self.seg_head),
            ("novel_seg_head", self.novel_seg_head),
        )
        for name, module in frozen_modules:
            if any(p.requires_grad for p in module.parameters()):
                raise RuntimeError(f"{name} must be frozen in task-space Stage-1")

        trainable_modules = (
            ("cdc_head", self.cdc_head),
            ("semantic_projector", self.semantic_projector),
            ("safe_radius_head", self.safe_radius_head),
        )
        for name, module in trainable_modules:
            if not all(p.requires_grad for p in module.parameters()):
                raise RuntimeError(
                    f"every {name} parameter must be trainable in task-space Stage-1"
                )

        allowed = ("cdc_head.", "semantic_projector.", "safe_radius_head.")
        bad = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.startswith(allowed)
        ]
        if bad:
            raise RuntimeError(
                "Only CDC/projector/radius may train in task-space Stage-1; "
                f"got {bad}"
            )

    def train(self, mode=True):
        # Keep the official task representation/classifiers in eval mode while
        # reproducing the original Stage-1 optimization on Base data only.
        nn.Module.train(self, mode)
        self.backbone.eval()
        self.seg_head.eval()
        self.novel_seg_head.eval()
        self.cdc_head.train(mode)
        self.semantic_projector.train(mode)
        self.safe_radius_head.train(mode)
        self._assert_task_stage1_trainability()
        return self
