"""Runtime checks for Stage-2 Phase-1 checkpoint loading and freezing."""

import os
from collections import OrderedDict

import torch

import pointcept.utils.comm as comm
from .builder import HOOKS
from .default import HookBase


@HOOKS.register_module()
class Stage2Phase1CheckpointLoader(HookBase):
    """Load the complete Stage-1 best model and fail on unexpected mismatches.

    The only parameter absent from the Stage-1 checkpoint is the new
    ``novel_seg_head`` introduced for GFS-VL registration. All Stage-1 PTv3,
    base-head and geometry-side parameters must load exactly.
    """

    def before_train(self):
        path = self.trainer.cfg.backbone_weight
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"Stage-1 best checkpoint not found: {path}")

        self.trainer.logger.info(f"Loading complete Stage-1 checkpoint: {path}")
        checkpoint = torch.load(
            path,
            map_location=lambda storage, loc: storage.cuda(),
        )

        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            if key.startswith("module."):
                key = key[7:]
            weight[key] = value

        model = (
            self.trainer.model.module
            if comm.get_world_size() > 1
            else self.trainer.model
        )
        load_info = model.load_state_dict(weight, strict=False)

        bad_missing = [
            key
            for key in load_info.missing_keys
            if not key.startswith("novel_seg_head.")
        ]
        unexpected = list(load_info.unexpected_keys)
        if bad_missing or unexpected:
            raise RuntimeError(
                "Incomplete Stage-1 checkpoint load: "
                f"bad_missing={bad_missing}, unexpected={unexpected}"
            )

        novel_missing = [
            key
            for key in load_info.missing_keys
            if key.startswith("novel_seg_head.")
        ]
        if not novel_missing:
            raise RuntimeError(
                "Expected the newly initialized novel_seg_head to be absent "
                "from the Stage-1 checkpoint"
            )

        if hasattr(model, "_assert_phase1_trainability"):
            model._assert_phase1_trainability()

        trainable_backbone = sum(
            p.numel() for p in model.backbone.parameters() if p.requires_grad
        )
        frozen_cdc = sum(
            p.numel() for p in model.cdc_head.parameters() if not p.requires_grad
        )
        total_cdc = sum(p.numel() for p in model.cdc_head.parameters())

        if trainable_backbone <= 0:
            raise RuntimeError("PTv3 has no trainable parameters in Phase 1")
        if frozen_cdc != total_cdc:
            raise RuntimeError("CDC is not completely frozen in Phase 1")

        self.trainer.logger.info(
            "Stage-2 Phase-1 checkpoint verified: all Stage-1 weights loaded; "
            f"new_only={novel_missing}; PTv3_trainable_params={trainable_backbone}; "
            f"CDC_frozen_params={frozen_cdc}/{total_cdc}"
        )
