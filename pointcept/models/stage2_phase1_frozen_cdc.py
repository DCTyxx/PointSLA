"""Stage-2 Phase-1 GFS-VL adaptation with a frozen Stage-1 CDC prior.

The model keeps the complete Stage-1 geometry stack in the state dict so the
best Stage-1 checkpoint can be loaded without discarding CDC-related modules,
while restoring the original GFS-VL registration behavior for task adaptation:

- PTv3 backbone: trainable;
- base segmentation head: trainable;
- novel segmentation head: trainable;
- CDC / semantic projector / safe-radius head: frozen and inactive in the loss.

Phase 1 therefore changes the task feature space with the original GFS-VL
objective while preserving the Stage-1 CDC parameters exactly for Phase 2.
"""

import torch
import torch.nn as nn

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS
from .conservative_boundary_semantic_tangent_cdc_base_geometry import (
    ConservativeBoundarySemanticTangentCDCBaseGeometrySegmentor,
)


@MODELS.register_module()
class Stage2Phase1FrozenCDCRegisSegmentor(
    ConservativeBoundarySemanticTangentCDCBaseGeometrySegmentor
):
    """Original GFS-VL registration model plus a frozen Stage-1 geometry stack."""

    def __init__(
        self,
        num_base_classes,
        num_novel_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        cdc=None,
        tangent=None,
        text_constraint=None,
        safe_radius=None,
        **kwargs,
    ):
        # Build the exact Stage-1 module hierarchy first so the Stage-1 best
        # checkpoint can load backbone, base head and all geometry-side weights
        # by their original parameter names.
        super().__init__(
            num_classes=num_base_classes,
            backbone_out_channels=backbone_out_channels,
            backbone=backbone,
            criteria=None,
            freeze_backbone=True,
            freeze_seg_head=True,
            cdc=cdc,
            tangent=tangent,
            text_constraint=text_constraint,
            safe_radius=safe_radius,
            **kwargs,
        )

        self.num_base_classes = int(num_base_classes)
        self.num_novel_classes = int(num_novel_classes)
        self.criteria = build_criteria(criteria)

        # Match the original GFS-VL registration head definition exactly.
        self.novel_seg_head = nn.Linear(
            backbone_out_channels, self.num_novel_classes, bias=False
        )

        # Phase 1 reverses the Stage-1 optimization roles: PTv3/base head adapt
        # to the K-shot task, while the learned geometry prior is preserved.
        self.freeze_backbone = False
        self.freeze_seg_head = False
        self.backbone.requires_grad_(True)
        self.seg_head.requires_grad_(True)

        self._freeze_geometry_stack()
        self._assert_phase1_trainability()

    def _geometry_modules(self):
        return (
            ("cdc_head", self.cdc_head),
            ("semantic_projector", self.semantic_projector),
            ("safe_radius_head", self.safe_radius_head),
        )

    def _freeze_geometry_stack(self):
        for _, module in self._geometry_modules():
            module.requires_grad_(False)
            module.eval()

    def _assert_phase1_trainability(self):
        if not any(p.requires_grad for p in self.backbone.parameters()):
            raise RuntimeError("Stage-2 Phase-1 requires a trainable PTv3 backbone")
        if not any(p.requires_grad for p in self.seg_head.parameters()):
            raise RuntimeError("Stage-2 Phase-1 requires a trainable base head")
        if not any(p.requires_grad for p in self.novel_seg_head.parameters()):
            raise RuntimeError("Stage-2 Phase-1 requires a trainable novel head")
        for name, module in self._geometry_modules():
            if any(p.requires_grad for p in module.parameters()):
                raise RuntimeError(f"{name} must remain frozen during Phase 1")

    def train(self, mode=True):
        # Do not use the Stage-1 train() behavior, which intentionally kept PTv3
        # and the base head in eval mode. Phase 1 follows original GFS-VL.
        nn.Module.train(self, mode)
        if mode:
            self.backbone.train()
            self.seg_head.train()
            self.novel_seg_head.train()
        self._freeze_geometry_stack()
        return self

    def getFeatures(self, input_dict):
        point = self.backbone(Point(input_dict))
        return point.feat if isinstance(point, Point) else point

    def forward(self, input_dict):
        # Phase 1 deliberately does not execute CDC or any geometry loss.
        point = self.backbone(Point(input_dict))
        feat = point.feat if isinstance(point, Point) else point

        base_seg_logits = self.seg_head(feat)
        novel_seg_logits = self.novel_seg_head(feat)
        seg_logits = torch.cat([base_seg_logits, novel_seg_logits], dim=1)

        if self.training:
            return dict(seg_logits=seg_logits)
        if "segment" in input_dict:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        return dict(seg_logits=seg_logits)
