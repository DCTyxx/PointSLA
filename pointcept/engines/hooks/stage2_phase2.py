"""Runtime checks for Stage-2 Phase-2 checkpoint loading and freezing."""

import os
from collections import OrderedDict

import torch

import pointcept.utils.comm as comm
from .builder import HOOKS
from .default import HookBase


@HOOKS.register_module()
class Stage2Phase2CheckpointLoader(HookBase):
    """Strictly restore one regis-specific Phase-1 checkpoint.

    Phase-2 must start from the exact task-adapted feature space associated with
    the same registration split.  Therefore every model key (PTv3, base/novel
    heads, CDC, semantic projector, safe-radius head) must load exactly.
    """

    def before_train(self):
        path = self.trainer.cfg.phase1_weight
        regis_name = getattr(self.trainer.cfg, "phase2_regis_name", "unknown")
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                f"Phase-1 checkpoint for {regis_name} not found: {path}"
            )

        self.trainer.logger.info(
            f"Loading Stage-2 Phase-1 checkpoint for {regis_name}: {path}"
        )
        checkpoint = torch.load(
            path,
            map_location=lambda storage, loc: storage.cuda(),
        )
        if "state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint has no state_dict: {path}")

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
        load_info = model.load_state_dict(weight, strict=True)
        if load_info.missing_keys or load_info.unexpected_keys:
            raise RuntimeError(
                "Strict Phase-1 -> Phase-2 load unexpectedly reported mismatches: "
                f"missing={load_info.missing_keys}, "
                f"unexpected={load_info.unexpected_keys}"
            )

        if not hasattr(model, "_assert_phase2_trainability"):
            raise RuntimeError(
                "Phase-2 model does not expose _assert_phase2_trainability()"
            )
        model._assert_phase2_trainability()

        trainable_cdc = sum(
            p.numel() for p in model.cdc_head.parameters() if p.requires_grad
        )
        total_cdc = sum(p.numel() for p in model.cdc_head.parameters())
        frozen_backbone = sum(
            p.numel() for p in model.backbone.parameters() if not p.requires_grad
        )
        total_backbone = sum(p.numel() for p in model.backbone.parameters())

        if trainable_cdc != total_cdc:
            raise RuntimeError("CDC is not fully trainable in Phase 2")
        if frozen_backbone != total_backbone:
            raise RuntimeError("PTv3 is not fully frozen in Phase 2")

        epoch = checkpoint.get("epoch", "unknown")
        self.trainer.logger.info(
            "Stage-2 Phase-2 checkpoint verified: "
            f"regis={regis_name}; source_epoch={epoch}; "
            f"strict_load=True; CDC_trainable={trainable_cdc}/{total_cdc}; "
            f"PTv3_frozen={frozen_backbone}/{total_backbone}"
        )
