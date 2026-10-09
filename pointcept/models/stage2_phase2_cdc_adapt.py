"""Stage-2 Phase-2: adapt CDC to each task-adapted GFS-VL feature space.

Phase 1 updates PTv3/base/novel segmentation heads while preserving the Stage-1
CDC prior.  Phase 2 reverses the optimization roles:

- task-adapted PTv3 backbone: frozen;
- base and novel segmentation heads: frozen;
- CDC geometry head: trainable;
- semantic projector and safe-radius head: frozen.

Only trusted labels already present in ``input_dict['segment']`` are used.  With
``ScanNetDataset_REGISTrain`` these are Base GT plus inserted K-shot Novel GT;
background/unknown points remain -1 and are ignored by tangent construction.
No 3D-VLM pseudo labels are used in this phase.
"""

import torch
import torch.nn as nn

from pointcept.models.utils.structure import Point
from .builder import MODELS
from .stage2_phase1_frozen_cdc import Stage2Phase1FrozenCDCRegisSegmentor


@MODELS.register_module()
class Stage2Phase2CDCAdaptSegmentor(Stage2Phase1FrozenCDCRegisSegmentor):
    """Frozen task backbone + trainable CDC tangent adaptation."""

    def __init__(self, **kwargs):
        # Build the exact Phase-1 hierarchy first.  This preserves parameter names
        # so every regis-specific Phase-1 checkpoint can be loaded strictly.
        super().__init__(**kwargs)

        self.freeze_backbone = True
        self.freeze_seg_head = True

        # Phase 2: task representation and classifiers are fixed.
        self.backbone.requires_grad_(False)
        self.seg_head.requires_grad_(False)
        self.novel_seg_head.requires_grad_(False)

        # CDC alone adapts to the fixed task-specific 64-D feature space.
        self.cdc_head.requires_grad_(True)

        # These Stage-1 semantic-side modules are deliberately held fixed in the
        # first Phase-2 implementation.  They are not executed by the loss below.
        self.semantic_projector.requires_grad_(False)
        self.safe_radius_head.requires_grad_(False)

        self._assert_phase2_trainability()

    def _assert_phase2_trainability(self):
        frozen_modules = (
            ("backbone", self.backbone),
            ("seg_head", self.seg_head),
            ("novel_seg_head", self.novel_seg_head),
            ("semantic_projector", self.semantic_projector),
            ("safe_radius_head", self.safe_radius_head),
        )
        for name, module in frozen_modules:
            if any(p.requires_grad for p in module.parameters()):
                raise RuntimeError(f"{name} must be frozen during Stage-2 Phase-2")

        if not any(p.requires_grad for p in self.cdc_head.parameters()):
            raise RuntimeError("CDC must be trainable during Stage-2 Phase-2")
        if not all(p.requires_grad for p in self.cdc_head.parameters()):
            raise RuntimeError("Every CDC parameter must be trainable in Phase 2")

        bad_trainable = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.startswith("cdc_head.")
        ]
        if bad_trainable:
            raise RuntimeError(
                "Only cdc_head may be trainable in Stage-2 Phase-2; got "
                f"{bad_trainable}"
            )

    def train(self, mode=True):
        # Bypass Phase-1 train(), which intentionally freezes CDC.  Set recursive
        # mode first, then restore the intended optimization/eval state explicitly.
        nn.Module.train(self, mode)
        self.backbone.eval()
        self.seg_head.eval()
        self.novel_seg_head.eval()
        self.semantic_projector.eval()
        self.safe_radius_head.eval()
        self.cdc_head.train(mode)
        self._assert_phase2_trainability()
        return self

    @staticmethod
    def _masked_mean_or_zero(value, mask):
        if bool(mask.any().item()):
            return value[mask].mean()
        return value.new_zeros(())

    def _phase2_tangent_objective(self, clean, labels):
        """QR Tangent-CDC objective with Base/Novel diagnostics.

        This is the same geometric objective used by Stage 1, evaluated in the
        *current regis-specific, task-adapted* frozen PTv3 feature space.
        """
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
                "Phase-2 CDC found no class with enough GT points for tangent KNN; "
                f"knn={self.tangent_knn}."
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
                    "Phase-2 CDC discrimination requires at least two GT classes "
                    "in the current feature bank."
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
        novel_mask = anchor_labels >= self.num_base_classes

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
        )

    def forward(self, input_dict):
        # The task-adapted representation is fixed during CDC adaptation.
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                point = self.backbone(Point(input_dict))
                feature = point.feat if isinstance(point, Point) else point

        labels = input_dict["segment"].long().to(feature.device)
        valid = labels >= 0
        if not bool(valid.any().item()):
            raise RuntimeError("Phase-2 batch contains no trusted GT points")

        with torch.cuda.amp.autocast(enabled=False):
            clean = feature.detach().float()
            geometry = self._phase2_tangent_objective(clean, labels)
            loss = (
                self.tangent_loss_weight * geometry["tangent_loss"]
                + self.tangent_disc_weight * geometry["disc_loss"]
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
            "basis_orth_error": basis_orth_error.detach(),
            "cdc_factor_energy": factor_energy.detach(),
        }
