"""Stage-2 Phase-3: CDC-guided Novel feature augmentation.

Phase 3 converts the task-adapted geometry learned in Phase 2 into a direct
segmentation improvement signal.  The representation and geometry are fixed:

- task-adapted PTv3 backbone: frozen;
- Phase-2 adapted CDC: frozen;
- base segmentation head: frozen;
- novel segmentation head: trainable.

For every trusted Novel feature z, CDC predicts an orthonormal tangent basis Q.
M online virtual features are sampled as

    z_tilde = z + rho * ||z|| * Q a,

where a is a unit Gaussian coefficient vector in the CDC tangent coordinates.
The trainable Novel head is optimized by real Base+Novel CE plus virtual Novel
CE through the complete Base+Novel classifier.  Thus real Base samples provide
implicit protection against Novel-logit intrusion while CDC expands the local
Novel decision region.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.utils.structure import Point
from .builder import MODELS
from .stage2_phase2_cdc_adapt import Stage2Phase2CDCAdaptSegmentor


@MODELS.register_module()
class Stage2Phase3CDCAugSegmentor(Stage2Phase2CDCAdaptSegmentor):
    """Frozen PTv3/CDC/base head + CDC-augmented Novel-head refinement."""

    def __init__(
        self,
        virtual_samples=4,
        virtual_radius=0.025,
        virtual_loss_weight=0.5,
        **kwargs,
    ):
        # Build exactly the Phase-2 hierarchy first so every Phase-2 checkpoint
        # can be restored with strict=True and no parameter-name translation.
        super().__init__(**kwargs)

        self.virtual_samples = int(virtual_samples)
        self.virtual_radius = float(virtual_radius)
        self.virtual_loss_weight = float(virtual_loss_weight)
        if self.virtual_samples < 1:
            raise ValueError("virtual_samples must be >= 1 for CDC augmentation")
        if self.virtual_radius <= 0:
            raise ValueError("virtual_radius must be > 0")
        if self.virtual_loss_weight < 0:
            raise ValueError("virtual_loss_weight must be >= 0")

        # Phase 3 freezes the task representation, geometry and Base classifier.
        self.backbone.requires_grad_(False)
        self.seg_head.requires_grad_(False)
        self.cdc_head.requires_grad_(False)
        self.semantic_projector.requires_grad_(False)
        self.safe_radius_head.requires_grad_(False)

        # Only the Novel classifier is refined by real + virtual Novel support.
        self.novel_seg_head.requires_grad_(True)

        self.freeze_backbone = True
        self.freeze_seg_head = True
        self._assert_phase3_trainability()

    def _assert_phase3_trainability(self):
        frozen_modules = (
            ("backbone", self.backbone),
            ("seg_head", self.seg_head),
            ("cdc_head", self.cdc_head),
            ("semantic_projector", self.semantic_projector),
            ("safe_radius_head", self.safe_radius_head),
        )
        for name, module in frozen_modules:
            if any(p.requires_grad for p in module.parameters()):
                raise RuntimeError(f"{name} must be frozen during Stage-2 Phase-3")

        if not any(p.requires_grad for p in self.novel_seg_head.parameters()):
            raise RuntimeError("Novel segmentation head must be trainable in Phase 3")
        if not all(p.requires_grad for p in self.novel_seg_head.parameters()):
            raise RuntimeError("Every Novel-head parameter must be trainable in Phase 3")

        bad_trainable = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
            and not name.startswith("novel_seg_head.")
        ]
        if bad_trainable:
            raise RuntimeError(
                "Only novel_seg_head may be trainable in Stage-2 Phase-3; got "
                f"{bad_trainable}"
            )

    def train(self, mode=True):
        # Bypass Phase-2 train(), which intentionally makes CDC trainable.
        nn.Module.train(self, mode)
        self.backbone.eval()
        self.seg_head.eval()
        self.cdc_head.eval()
        self.semantic_projector.eval()
        self.safe_radius_head.eval()
        self.novel_seg_head.train(mode)
        self._assert_phase3_trainability()
        return self

    def _full_logits(self, feature):
        base_logits = self.seg_head(feature)
        novel_logits = self.novel_seg_head(feature)
        return torch.cat([base_logits, novel_logits], dim=1)

    @torch.no_grad()
    def _cdc_tangent_basis(self, novel_feature):
        """Return the Phase-2 CDC orthonormal tangent basis for Novel features."""
        n = novel_feature.shape[0]
        relative_h = torch.full(
            (n,),
            self.cdc_probe_h,
            device=novel_feature.device,
            dtype=novel_feature.dtype,
        )
        feature_rms = novel_feature.square().mean(dim=-1).add(1e-8).sqrt()
        factor = self.cdc_head(novel_feature, relative_h * feature_rms)
        return self._orthonormalize_factor(factor)

    @torch.no_grad()
    def _sample_virtual_novel_features(self, novel_feature):
        """Online one-step CDC tangent augmentation.

        For each real Novel point, sample ``virtual_samples`` unit directions in
        the learned rank-r tangent subspace and move by a relative radius
        ``virtual_radius * ||z||``.
        """
        q = self._cdc_tangent_basis(novel_feature)  # [N, D, r]
        coeff = torch.randn(
            novel_feature.shape[0],
            self.virtual_samples,
            self.cdc_rank,
            device=novel_feature.device,
            dtype=novel_feature.dtype,
        )
        coeff = F.normalize(coeff, p=2, dim=-1, eps=1e-8)
        direction = torch.einsum("ndr,nmr->nmd", q, coeff)
        direction = F.normalize(direction, p=2, dim=-1, eps=1e-8)

        feature_norm = novel_feature.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        step = self.virtual_radius * feature_norm[:, None, :] * direction
        virtual = novel_feature[:, None, :] + step
        return virtual.reshape(-1, novel_feature.shape[-1]), step

    def forward(self, input_dict):
        # PTv3 is fixed in Phase 3.  Keep geometry-side computation in fp32 to
        # match Phase 2 and avoid introducing AMP-dependent tangent changes.
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                point = self.backbone(Point(input_dict))
                feature = point.feat if isinstance(point, Point) else point
                feature = feature.detach().float()

        # Standard inference path: CDC is not needed at test time.
        if not self.training:
            seg_logits = self._full_logits(feature)
            if "segment" in input_dict:
                labels = input_dict["segment"].long().to(seg_logits.device)
                loss = self.criteria(seg_logits, labels)
                return dict(loss=loss, seg_logits=seg_logits)
            return dict(seg_logits=seg_logits)

        labels = input_dict["segment"].long().to(feature.device)
        valid = labels >= 0
        if not bool(valid.any().item()):
            raise RuntimeError("Phase-3 batch contains no trusted GT points")

        # Explicitly remove unknown/background points before both real CE and CDC
        # augmentation.  Only Base GT + inserted Novel GT supervise Phase 3.
        real_feature = feature[valid]
        real_labels = labels[valid]
        total_classes = self.num_base_classes + self.num_novel_classes
        if int(real_labels.max().item()) >= total_classes:
            raise RuntimeError(
                "Phase-3 trusted label exceeds Base+Novel classifier range: "
                f"max_label={int(real_labels.max().item())}, "
                f"num_classes={total_classes}"
            )

        real_logits = self._full_logits(real_feature)
        real_loss = self.criteria(real_logits, real_labels)

        novel_mask = real_labels >= self.num_base_classes
        num_novel = novel_mask.sum()
        if bool(novel_mask.any().item()):
            novel_feature = real_feature[novel_mask]
            novel_labels = real_labels[novel_mask]

            virtual_feature, step = self._sample_virtual_novel_features(
                novel_feature
            )
            virtual_labels = novel_labels.repeat_interleave(self.virtual_samples)
            virtual_logits = self._full_logits(virtual_feature)
            virtual_loss = self.criteria(virtual_logits, virtual_labels)

            relative_step = (
                step.norm(dim=-1)
                / novel_feature.norm(dim=-1).clamp_min(1e-8)[:, None]
            ).mean()
            num_virtual = virtual_labels.numel()
        else:
            # Robust fallback for a crop containing no Novel point.  Real Base CE
            # still protects the classifier; later batches can provide Novel aug.
            virtual_loss = real_loss.new_zeros(())
            relative_step = real_loss.new_zeros(())
            num_virtual = 0

        loss = real_loss + self.virtual_loss_weight * virtual_loss

        return {
            "loss": loss,
            "real_loss": real_loss.detach(),
            "virtual_loss": virtual_loss.detach(),
            "num_novel_points": num_novel.to(dtype=real_loss.dtype),
            "num_virtual_points": real_loss.new_tensor(float(num_virtual)),
            "virtual_step_ratio": relative_step.detach(),
        }
