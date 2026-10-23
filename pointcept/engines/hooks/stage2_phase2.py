"""Runtime checks and structured statistics for Stage-2 Phase-2."""

import json
import os
from collections import OrderedDict

import torch
import torch.distributed as dist

import pointcept.utils.comm as comm
from .builder import HOOKS
from .default import HookBase


@HOOKS.register_module()
class Stage2Phase2CheckpointLoader(HookBase):
    """Strictly restore one regis-specific Phase-1 checkpoint.

    Phase-2 must start from the exact task-adapted feature space associated with
    the same registration split. Therefore every model key (PTv3, base/novel
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


@HOOKS.register_module()
class Stage2Phase2MetricSummary(HookBase):
    """Save exact anchor-weighted Phase-2 LSL metrics in JSON.

    The five paper-facing LSL diagnostics are aggregated separately for
    Overall/Base/Novel anchors:

    1. TAS (higher is better)
    2. Intra-EVR (higher is better)
    3. Inter-EVR (lower is better)
    4. GDG = Intra-EVR - Inter-EVR (higher is better)
    5. Mean Principal Angle in degrees (lower is better)

    Aggregation is anchor-weighted rather than a simple mean over batches. In
    distributed training, per-rank metric sums/counts are all-reduced before
    epoch accumulation, so the reported values cover all GPUs exactly.
    """

    BASIC_METRICS = (
        "loss",
        "tangent_loss",
        "discrimination_loss",
        "basis_orth_error",
        "cdc_factor_energy",
    )

    LSL_FIELDS = (
        "tas",
        "intra_evr",
        "inter_evr",
        "gdg",
        "principal_angle_deg",
    )

    def __init__(self):
        self.records = []
        self.lsl_records = []
        self._epoch_lsl = None

    def before_epoch(self):
        self._epoch_lsl = {
            group: {
                "count": 0.0,
                **{field: 0.0 for field in self.LSL_FIELDS},
            }
            for group in ("overall", "base", "novel")
        }

    @staticmethod
    def _metric_key(group, field):
        if group == "overall":
            prefix = "train"
        else:
            prefix = group
        return f"{prefix}_{field}"

    @staticmethod
    def _get_count(output, group):
        base_count = output["num_base_anchors"].detach().float()
        novel_count = output["num_novel_anchors"].detach().float()
        if group == "base":
            return base_count
        if group == "novel":
            return novel_count
        return base_count + novel_count

    def after_step(self):
        output = self.trainer.comm_info.get("model_output_dict", None)
        if output is None:
            return

        required = ["num_base_anchors", "num_novel_anchors"]
        required += [
            self._metric_key(group, field)
            for group in ("overall", "base", "novel")
            for field in self.LSL_FIELDS
        ]
        missing = [key for key in required if key not in output]
        if missing:
            raise KeyError(
                "Phase-2 LSL summary is missing model outputs: "
                f"{missing}"
            )

        for group in ("overall", "base", "novel"):
            count = self._get_count(output, group)
            packed = [
                output[self._metric_key(group, field)].detach().float() * count
                for field in self.LSL_FIELDS
            ]
            packed.append(count)
            packed = torch.stack(packed)

            if comm.get_world_size() > 1:
                if not dist.is_available() or not dist.is_initialized():
                    raise RuntimeError(
                        "Distributed Phase-2 LSL aggregation requested but "
                        "torch.distributed is not initialized"
                    )
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)

            packed = packed.cpu()
            global_count = float(packed[-1].item())
            self._epoch_lsl[group]["count"] += global_count
            for idx, field in enumerate(self.LSL_FIELDS):
                self._epoch_lsl[group][field] += float(packed[idx].item())

    def _collect_epoch_metrics(self):
        record = {
            "regis": str(
                getattr(self.trainer.cfg, "phase2_regis_name", "unknown")
            ),
            "epoch": int(self.trainer.epoch + 1),
        }

        for key in self.BASIC_METRICS:
            try:
                record[key] = float(self.trainer.storage.history(key).avg)
            except Exception:
                continue

        if self._epoch_lsl is None:
            raise RuntimeError("Phase-2 LSL epoch accumulator was not initialized")

        for group in ("overall", "base", "novel"):
            count = self._epoch_lsl[group]["count"]
            record[f"num_{group}_anchors"] = count
            for field in self.LSL_FIELDS:
                value = (
                    self._epoch_lsl[group][field] / count
                    if count > 0.0
                    else 0.0
                )
                record[f"{group}_{field}"] = value

        # Backward-compatible aliases expected by old result parsers.
        record["train_tas"] = record["overall_tas"]
        record["train_gdg"] = record["overall_gdg"]
        record["base_tas"] = record["base_tas"]
        record["base_gdg"] = record["base_gdg"]
        record["novel_tas"] = record["novel_tas"]
        record["novel_gdg"] = record["novel_gdg"]
        record["num_base_anchors"] = record["num_base_anchors"]
        record["num_novel_anchors"] = record["num_novel_anchors"]
        return record

    def _to_lsl_payload(self, record):
        groups = {}
        for group in ("overall", "base", "novel"):
            groups[group] = {
                "TAS": record[f"{group}_tas"],
                "Intra-EVR": record[f"{group}_intra_evr"],
                "Inter-EVR": record[f"{group}_inter_evr"],
                "GDG": record[f"{group}_gdg"],
                "Mean-Principal-Angle-Deg": record[
                    f"{group}_principal_angle_deg"
                ],
                "num_anchors": record[f"num_{group}_anchors"],
            }
        return {
            "regis": record["regis"],
            "epoch": record["epoch"],
            "aggregation": "anchor_weighted_all_gpu",
            "groups": groups,
        }

    def _write_json(self, filename, payload):
        if not comm.is_main_process():
            return
        path = os.path.join(self.trainer.cfg.save_path, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def after_epoch(self):
        record = self._collect_epoch_metrics()
        lsl_payload = self._to_lsl_payload(record)

        self.records.append(record)
        self.lsl_records.append(lsl_payload)
        self._write_json("phase2_metrics_history.json", self.records)
        self._write_json("phase2_metrics_final.json", record)
        self._write_json("phase2_lsl_metrics_history.json", self.lsl_records)
        self._write_json("phase2_lsl_metrics_final.json", lsl_payload)

        if comm.is_main_process():
            for group in ("overall", "base", "novel"):
                self.trainer.logger.info(
                    "Stage2-Phase2 LSL metrics "
                    f"[{record['regis']}] epoch {record['epoch']} | "
                    f"{group.upper()} | "
                    f"TAS: {record[f'{group}_tas']:.6f} | "
                    f"Intra-EVR: {record[f'{group}_intra_evr']:.6f} | "
                    f"Inter-EVR: {record[f'{group}_inter_evr']:.6f} | "
                    f"GDG: {record[f'{group}_gdg']:.6f} | "
                    "MeanPrincipalAngle(deg): "
                    f"{record[f'{group}_principal_angle_deg']:.6f} | "
                    f"anchors: {int(record[f'num_{group}_anchors'])}"
                )
