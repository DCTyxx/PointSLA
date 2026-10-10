"""Runtime loading and GFS metrics for Stage-2 Phase-3."""

import json
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import pointops2.pointops as pointops

import pointcept.utils.comm as comm
from pointcept.utils.misc import intersection_and_union_gpu
from pointcept.engines.test import TESTERS

from .builder import HOOKS
from .default import HookBase


@HOOKS.register_module()
class Stage2Phase3CheckpointLoader(HookBase):
    """Strictly restore the matching regis-specific Phase-2 checkpoint."""

    def before_train(self):
        path = self.trainer.cfg.phase2_weight
        regis_name = getattr(self.trainer.cfg, "phase3_regis_name", "unknown")
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                f"Phase-2 checkpoint for {regis_name} not found: {path}"
            )

        self.trainer.logger.info(
            f"Loading Stage-2 Phase-2 checkpoint for {regis_name}: {path}"
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
                "Strict Phase-2 -> Phase-3 load unexpectedly reported mismatches: "
                f"missing={load_info.missing_keys}, "
                f"unexpected={load_info.unexpected_keys}"
            )

        if not hasattr(model, "_assert_phase3_trainability"):
            raise RuntimeError(
                "Phase-3 model does not expose _assert_phase3_trainability()"
            )
        model._assert_phase3_trainability()

        trainable_novel = sum(
            p.numel() for p in model.novel_seg_head.parameters() if p.requires_grad
        )
        total_novel = sum(p.numel() for p in model.novel_seg_head.parameters())
        frozen_backbone = sum(
            p.numel() for p in model.backbone.parameters() if not p.requires_grad
        )
        total_backbone = sum(p.numel() for p in model.backbone.parameters())
        frozen_cdc = sum(
            p.numel() for p in model.cdc_head.parameters() if not p.requires_grad
        )
        total_cdc = sum(p.numel() for p in model.cdc_head.parameters())
        frozen_base = sum(
            p.numel() for p in model.seg_head.parameters() if not p.requires_grad
        )
        total_base = sum(p.numel() for p in model.seg_head.parameters())

        if trainable_novel != total_novel:
            raise RuntimeError("Novel head is not fully trainable in Phase 3")
        if frozen_backbone != total_backbone:
            raise RuntimeError("PTv3 is not fully frozen in Phase 3")
        if frozen_cdc != total_cdc:
            raise RuntimeError("CDC is not fully frozen in Phase 3")
        if frozen_base != total_base:
            raise RuntimeError("Base head is not fully frozen in Phase 3")

        epoch = checkpoint.get("epoch", "unknown")
        self.trainer.logger.info(
            "Stage-2 Phase-3 checkpoint verified: "
            f"regis={regis_name}; source_epoch={epoch}; strict_load=True; "
            f"NovelHead_trainable={trainable_novel}/{total_novel}; "
            f"PTv3_frozen={frozen_backbone}/{total_backbone}; "
            f"CDC_frozen={frozen_cdc}/{total_cdc}; "
            f"BaseHead_frozen={frozen_base}/{total_base}"
        )


@HOOKS.register_module()
class Stage2Phase3GFSEvaluator(HookBase):
    """Epoch-wise Base/Novel/HM evaluator used for best-HM selection.

    This is the lightweight validation path.  The precise fragment-based GFS
    tester is run once on the selected best checkpoint after training.
    """

    def __init__(self):
        self.records = []

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def _write_history(self):
        if not comm.is_main_process():
            return
        path = os.path.join(
            self.trainer.cfg.save_path, "phase3_val_history.json"
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, sort_keys=True)

    def eval(self):
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Phase-3 GFS Validation >>>>>>>>>>>>>>>>"
        )
        self.trainer.model.eval()

        total_classes = int(self.trainer.cfg.data.num_base_novels)
        num_base = int(self.trainer.cfg.data.num_bases)
        ignore_index = int(self.trainer.cfg.data.ignore_index)

        intersection_total = torch.zeros(
            total_classes, dtype=torch.long, device="cuda"
        )
        union_total = torch.zeros_like(intersection_total)
        target_total = torch.zeros_like(intersection_total)
        loss_sum = torch.zeros((), dtype=torch.float64, device="cuda")
        batch_count = torch.zeros((), dtype=torch.float64, device="cuda")

        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            logits = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = logits.max(1)[1]
            segment = input_dict["segment"]

            if "origin_coord" in input_dict:
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                pred = pred[idx.flatten().long()]
                segment = input_dict["origin_segment"]

            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                total_classes,
                ignore_index,
            )
            intersection_total += intersection.to(dtype=torch.long)
            union_total += union.to(dtype=torch.long)
            target_total += target.to(dtype=torch.long)
            loss_sum += loss.detach().double()
            batch_count += 1.0

            self.trainer.logger.info(
                "Phase3 Val: [{}/{}] Loss {:.4f}".format(
                    i + 1, len(self.trainer.val_loader), loss.item()
                )
            )

        if comm.get_world_size() > 1:
            dist.all_reduce(intersection_total)
            dist.all_reduce(union_total)
            dist.all_reduce(target_total)
            dist.all_reduce(loss_sum)
            dist.all_reduce(batch_count)

        intersection = intersection_total.cpu().numpy().astype(np.float64)
        union = union_total.cpu().numpy().astype(np.float64)
        target = target_total.cpu().numpy().astype(np.float64)
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)

        m_iou = float(np.mean(iou_class))
        base_miou = float(np.mean(iou_class[:num_base]))
        novel_miou = float(np.mean(iou_class[num_base:]))
        hm_miou = float(
            2.0 * base_miou * novel_miou
            / (base_miou + novel_miou + 1e-10)
        )
        m_acc = float(np.mean(acc_class))
        all_acc = float(intersection.sum() / (target.sum() + 1e-10))
        loss_avg = float((loss_sum / batch_count.clamp_min(1.0)).item())

        record = {
            "regis": str(
                getattr(self.trainer.cfg, "phase3_regis_name", "unknown")
            ),
            "epoch": int(self.trainer.epoch + 1),
            "loss": loss_avg,
            "mIoU": m_iou,
            "base_mIoU": base_miou,
            "novel_mIoU": novel_miou,
            "hm_mIoU": hm_miou,
            "mAcc": m_acc,
            "allAcc": all_acc,
        }
        self.records.append(record)
        self._write_history()

        self.trainer.logger.info(
            "Phase3 Val result: mIoU/Base/Novel/HM/mAcc/allAcc "
            "{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(
                m_iou, base_miou, novel_miou, hm_miou, m_acc, all_acc
            )
        )
        if self.trainer.writer is not None and comm.is_main_process():
            epoch = self.trainer.epoch + 1
            self.trainer.writer.add_scalar("val/loss", loss_avg, epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, epoch)
            self.trainer.writer.add_scalar("val/base_mIoU", base_miou, epoch)
            self.trainer.writer.add_scalar("val/novel_mIoU", novel_miou, epoch)
            self.trainer.writer.add_scalar("val/hm_mIoU", hm_miou, epoch)

        # CheckpointSaver consumes this immediately after this hook.  HM is the
        # generalized few-shot selection metric, not Novel mIoU alone.
        self.trainer.comm_info["current_metric_value"] = hm_miou
        self.trainer.comm_info["current_metric_name"] = "HM"
        self.trainer.logger.info(
            "<<<<<<<<<<<<<<<< End Phase-3 GFS Validation <<<<<<<<<<<<<<<<"
        )


@HOOKS.register_module()
class Stage2Phase3PreciseEvaluator(HookBase):
    """Run the repository's fragment-based GFS tester on the selected model.

    The resulting Base/Novel/HM values are written to JSON so the five regis
    runs can be aggregated automatically by the Phase-3 launcher.
    """

    def __init__(self, test_last=False):
        self.test_last = bool(test_last)

    def after_train(self):
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Phase-3 Precise Evaluation >>>>>>>>>>>>>>>>"
        )
        torch.cuda.empty_cache()
        cfg = self.trainer.cfg
        tester = TESTERS.build(
            dict(type=cfg.test.type, cfg=cfg, model=self.trainer.model)
        )

        selected = "last" if self.test_last else "best"
        if self.test_last:
            self.trainer.logger.info("=> Testing Phase-3 model_last ...")
        else:
            best_path = os.path.join(cfg.save_path, "model", "model_best.pth")
            if not os.path.isfile(best_path):
                raise FileNotFoundError(
                    f"Phase-3 best checkpoint not found: {best_path}"
                )
            self.trainer.logger.info(
                f"=> Testing Phase-3 model_best selected by validation HM: {best_path}"
            )
            checkpoint = torch.load(
                best_path,
                map_location=lambda storage, loc: storage.cuda(),
            )
            tester.model.load_state_dict(checkpoint["state_dict"], strict=True)

        result = tester.test()
        if comm.is_main_process():
            if result is None:
                raise RuntimeError("Phase-3 precise GFS tester returned no result")
            m_iou, base_iou, novel_iou, hm_iou, iou_list = result
            record = {
                "regis": str(getattr(cfg, "phase3_regis_name", "unknown")),
                "selected_checkpoint": selected,
                "mIoU": float(m_iou),
                "base_mIoU": float(base_iou),
                "novel_mIoU": float(novel_iou),
                "hm_mIoU": float(hm_iou),
                "class_iou": np.asarray(iou_list).astype(float).tolist(),
            }
            path = os.path.join(cfg.save_path, "phase3_metrics_final.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, sort_keys=True)
            self.trainer.logger.info(
                "Stage2-Phase3 precise metrics [{}]: mIoU={:.6f}, "
                "Base={:.6f}, Novel={:.6f}, HM={:.6f}".format(
                    record["regis"],
                    record["mIoU"],
                    record["base_mIoU"],
                    record["novel_mIoU"],
                    record["hm_mIoU"],
                )
            )
        self.trainer.logger.info(
            "<<<<<<<<<<<<<<<< End Phase-3 Precise Evaluation <<<<<<<<<<<<<<<<"
        )
