"""Trainer for merged Stage-1 + Stage-2 Phase-2 geometry learning.

Unlike the original Stage-1 trainer, training uses REGISTrain data containing
Base GT + inserted K-shot Novel GT.  However, the CLIP semantic prior and every
safe-boundary diagnostic remain strictly Base-only via ``data.names`` and the
held-out ``data.val`` BaseTrain split.
"""

from .train import TRAINERS
from .conservative_boundary_semantic_tangent_cdc_train import (
    ConservativeBoundarySemanticTangentCDCTrainer,
)


@TRAINERS.register_module("MergedStage1Phase2GeometryTrainer")
class MergedStage1Phase2GeometryTrainer(
    ConservativeBoundarySemanticTangentCDCTrainer
):
    """Task-space geometry training with Base-only semantic boundaries."""

    @staticmethod
    def _assert_base_only_protocol(cfg):
        # This method intentionally replaces the original Stage-1 Base-only
        # protocol check. Geometry training is allowed to see trusted Novel GT,
        # but text prototypes / semantic boundary evaluation must remain Base-only.
        train_type = str(cfg.data.train.type)
        if "REGISTrain" not in train_type:
            raise RuntimeError(
                "Merged geometry training must use a REGISTrain dataset with "
                f"trusted Base + K-shot Novel GT, got '{train_type}'."
            )

        if not hasattr(cfg.data, "num_bases"):
            raise RuntimeError("Merged geometry config must define data.num_bases")
        num_bases = int(cfg.data.num_bases)
        base_names = list(cfg.data.names)
        if len(base_names) != num_bases:
            raise RuntimeError(
                "Merged geometry semantic protocol requires data.names to contain "
                f"BASE classes only: got {len(base_names)} names, expected {num_bases}."
            )

        # The inherited CLIP builder reads data.num_classes/data.names. Force the
        # semantic vocabulary to be Base-only even though geometry sees Novel GT.
        if int(cfg.data.num_classes) != num_bases:
            raise RuntimeError(
                "Merged geometry requires data.num_classes == data.num_bases for "
                "Base-only text prototype construction."
            )

        val_type = str(cfg.data.val.type)
        if "BASETrain" not in val_type:
            raise RuntimeError(
                "Safe-boundary evaluation must use a held-out Base-only dataset; "
                f"got cfg.data.val.type='{val_type}'."
            )

    def _build_and_inject_text_prior(self):
        # Phase-1 configs did not need a text-encoder config because Stage-1
        # checkpoints already contained text buffers. The merged stage learns the
        # semantic projector from scratch, so provide the repository's standard
        # GFS-VL CLIP config when the inherited task config does not define one.
        if not hasattr(self.cfg, "cdc_text_cfg_file"):
            self.cfg.cdc_text_cfg_file = (
                "./pointcept/models/PLA/tools/cfgs/scannet200_models/zs/"
                "spconv_clip_caption_openscene.yaml"
            )

        # Reuse the original leakage-guarded CLIP construction. Since configs set
        # data.names to Base classes only, no Novel text prototype can be encoded.
        super()._build_and_inject_text_prior()
        self.logger.info(
            "=> Merged geometry protocol: tangent loss uses trusted Base+Novel GT; "
            "semantic alignment and safe-boundary supervision are BASE-ONLY."
        )
