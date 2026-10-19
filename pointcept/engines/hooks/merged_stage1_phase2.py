"""Hooks for merged Stage-1 + Stage-2 Phase-2 geometry learning."""

import json
import os
from collections import OrderedDict

import torch

import pointcept.utils.comm as comm
from .builder import HOOKS
from .default import HookBase


@HOOKS.register_module()
class MergedStage1Phase2CheckpointLoader(HookBase):
    """Load only the official GFS-VL task modules from a registration checkpoint.

    The official checkpoint contains the task-adapted PTv3, Base head and Novel
    head.  CDC/projector/radius parameters are intentionally NOT restored: they
    start from the current model initialization and are learned from scratch in
    the merged geometry stage.
    """

    TASK_PREFIXES = ("backbone.", "seg_head.", "novel_seg_head.")

    def before_train(self):
        path = self.trainer.cfg.task_weight
        regis_name = getattr(self.trainer.cfg, "merged_regis_name", "unknown")
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                f"Official task checkpoint for {regis_name} not found: {path}"
            )

        self.trainer.logger.info(
            f"Loading official GFS-VL task checkpoint for {regis_name}: {path}"
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
                "Official GFS-VL task checkpoint is incompatible with merged model: "
                f"missing_task_keys={missing[:20]}, unexpected_task_keys={unexpected[:20]}"
            )

        shape_errors = []
        for key in sorted(expected_task_keys):
            if tuple(source[key].shape) != tuple(model_state[key].shape):
                shape_errors.append(
                    (key, tuple(source[key].shape), tuple(model_state[key].shape))
                )
        if shape_errors:
            raise RuntimeError(
                "Task checkpoint shape mismatch: " + "; ".join(
                    f"{k}: ckpt={a}, model={b}" for k, a, b in shape_errors[:20]
                )
            )

        load_info = model.load_state_dict(source, strict=False)
        # Every reported missing key must belong to the freshly initialized
        # geometry/text stack (plus buffers). Task modules were checked above.
        bad_missing = [
            key
            for key in load_info.missing_keys
            if key.startswith(self.TASK_PREFIXES)
        ]
        if bad_missing or load_info.unexpected_keys:
            raise RuntimeError(
                "Partial official checkpoint load failed: "
                f"bad_missing={bad_missing}, unexpected={load_info.unexpected_keys}"
            )

        if not hasattr(model, "_assert_merged_trainability"):
            raise RuntimeError(
                "Merged geometry model does not expose _assert_merged_trainability()"
            )
        model._assert_merged_trainability()

        trainable_geometry = sum(
            p.numel()
            for name, p in model.named_parameters()
            if p.requires_grad
            and name.startswith(("cdc_head.", "semantic_projector.", "safe_radius_head."))
        )
        frozen_task = sum(
            p.numel()
            for name, p in model.named_parameters()
            if (name.startswith(self.TASK_PREFIXES) and not p.requires_grad)
        )
        total_task = sum(
            p.numel()
            for name, p in model.named_parameters()
            if name.startswith(self.TASK_PREFIXES)
        )

        self.trainer.logger.info(
            "Merged Stage1+Phase2 checkpoint verified: "
            f"regis={regis_name}; loaded_task_keys={len(expected_task_keys)}; "
            f"task_frozen={frozen_task}/{total_task}; "
            f"fresh_geometry_trainable_params={trainable_geometry}."
        )
        self.trainer.logger.info(
            "Geometry provenance: CDC/projector/radius were NOT loaded from the "
            "official checkpoint and are learned in the task-adapted feature space."
        )


@HOOKS.register_module()
class MergedStage1Phase2MetricSummary(HookBase):
    """Persist merged-stage epoch metrics for provenance and ablations."""

    METRICS = (
        "loss",
        "tangent_loss",
        "discrimination_loss",
        "train_tas",
        "train_gdg",
        "base_tas",
        "base_gdg",
        "novel_tas",
        "novel_gdg",
        "num_base_anchors",
        "num_novel_anchors",
        "base_boundary_anchors",
        "text_align_loss",
        "text_acc",
        "radius_boundary_loss",
        "radius_regression_loss",
        "radius_overshoot_loss",
        "predicted_svr",
        "oracle_svr",
        "mean_safe_radius",
        "oracle_mean_safe_radius",
        "radius_mae",
        "radius_ratio",
        "radius_active",
        "basis_orth_error",
        "cdc_factor_energy",
    )

    def __init__(self):
        self.records = []

    def _collect(self):
        record = {
            "regis": str(
                getattr(self.trainer.cfg, "merged_regis_name", "unknown")
            ),
            "epoch": int(self.trainer.epoch + 1),
        }
        for key in self.METRICS:
            try:
                record[key] = float(self.trainer.storage.history(key).avg)
            except Exception:
                continue
        return record

    def _write(self, filename, payload):
        if not comm.is_main_process():
            return
        path = os.path.join(self.trainer.cfg.save_path, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def after_epoch(self):
        record = self._collect()
        self.records.append(record)
        self._write("merged_geometry_metrics_history.json", self.records)
        self._write("merged_geometry_metrics_final.json", record)

        if comm.is_main_process():
            tracked = ", ".join(
                f"{key}={record[key]:.6f}"
                for key in (
                    "base_tas",
                    "novel_tas",
                    "base_gdg",
                    "novel_gdg",
                    "text_acc",
                    "mean_safe_radius",
                )
                if key in record
            )
            self.trainer.logger.info(
                f"Merged geometry metrics [{record['regis']}] epoch "
                f"{record['epoch']}: {tracked}"
            )
