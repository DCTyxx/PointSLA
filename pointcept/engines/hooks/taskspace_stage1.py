"""Checkpoint loading for task-space Base-only Stage-1."""

import os
from collections import OrderedDict

import torch

import pointcept.utils.comm as comm
from .builder import HOOKS
from .default import HookBase


@HOOKS.register_module()
class TaskSpaceStage1CheckpointLoader(HookBase):
    """Load official task-adapted PTv3/Base/Novel heads, keep geometry fresh.

    Only ``backbone.*``, ``seg_head.*`` and ``novel_seg_head.*`` are restored
    from the official GFS-VL registration checkpoint.  CDC/projector/radius are
    intentionally left at their fresh initialization and are then learned from
    Base-only data using the original Stage-1 objective.
    """

    TASK_PREFIXES = ("backbone.", "seg_head.", "novel_seg_head.")

    def before_train(self):
        path = self.trainer.cfg.task_weight
        regis_name = getattr(
            self.trainer.cfg, "task_stage1_regis_name", "unknown"
        )
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                f"Official task checkpoint for {regis_name} not found: {path}"
            )

        self.trainer.logger.info(
            f"Loading official GFS-VL task checkpoint for task-space Stage-1 "
            f"[{regis_name}]: {path}"
        )
        checkpoint = torch.load(
            path,
            map_location=lambda storage, loc: storage.cuda(),
        )
        raw_state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(raw_state, dict):
            raise TypeError(f"Unsupported checkpoint format: {path}")

        source = OrderedDict()
        for key, value in raw_state.items():
            if key.startswith("module."):
                key = key[7:]
            if key.startswith(self.TASK_PREFIXES):
                source[key] = value

        model = (
            self.trainer.model.module
            if comm.get_world_size() > 1
            else self.trainer.model
        )
        model_state = model.state_dict()
        expected_task_keys = {
            key for key in model_state if key.startswith(self.TASK_PREFIXES)
        }
        source_task_keys = set(source.keys())

        missing = sorted(expected_task_keys - source_task_keys)
        unexpected = sorted(source_task_keys - expected_task_keys)
        if missing or unexpected:
            raise RuntimeError(
                "Official GFS-VL checkpoint is incompatible with task-space "
                "Stage-1: "
                f"missing_task_keys={missing[:20]}, "
                f"unexpected_task_keys={unexpected[:20]}"
            )

        shape_errors = []
        for key in sorted(expected_task_keys):
            if tuple(source[key].shape) != tuple(model_state[key].shape):
                shape_errors.append(
                    (key, tuple(source[key].shape), tuple(model_state[key].shape))
                )
        if shape_errors:
            raise RuntimeError(
                "Official task checkpoint shape mismatch: "
                + "; ".join(
                    f"{k}: ckpt={a}, model={b}"
                    for k, a, b in shape_errors[:20]
                )
            )

        load_info = model.load_state_dict(source, strict=False)
        bad_missing = [
            key
            for key in load_info.missing_keys
            if key.startswith(self.TASK_PREFIXES)
        ]
        if bad_missing or load_info.unexpected_keys:
            raise RuntimeError(
                "Partial official task load failed: "
                f"bad_missing={bad_missing}, "
                f"unexpected={load_info.unexpected_keys}"
            )

        if not hasattr(model, "_assert_task_stage1_trainability"):
            raise RuntimeError(
                "Task-space Stage-1 model does not expose "
                "_assert_task_stage1_trainability()"
            )
        model._assert_task_stage1_trainability()

        frozen_task = sum(
            p.numel()
            for name, p in model.named_parameters()
            if name.startswith(self.TASK_PREFIXES) and not p.requires_grad
        )
        total_task = sum(
            p.numel()
            for name, p in model.named_parameters()
            if name.startswith(self.TASK_PREFIXES)
        )
        trainable_geometry = sum(
            p.numel()
            for name, p in model.named_parameters()
            if p.requires_grad
            and name.startswith(
                ("cdc_head.", "semantic_projector.", "safe_radius_head.")
            )
        )

        if frozen_task != total_task:
            raise RuntimeError(
                "Official task modules are not fully frozen in task-space Stage-1"
            )

        self.trainer.logger.info(
            "Task-space Stage-1 checkpoint verified: "
            f"regis={regis_name}; loaded_task_keys={len(expected_task_keys)}; "
            f"task_frozen={frozen_task}/{total_task}; "
            f"fresh_geometry_trainable_params={trainable_geometry}."
        )
        self.trainer.logger.info(
            "Task-space Stage-1 provenance: official PTv3/Base/Novel heads are "
            "frozen; CDC/projector/radius start fresh and train on Base only."
        )
