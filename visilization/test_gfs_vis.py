#!/usr/bin/env python3
"""Safe GFS semantic-segmentation evaluator for qualitative visualization.

This evaluator intentionally saves only full-resolution GT/prediction arrays per
scene. Raw XYZ/RGB are reloaded later by build_showcase.py from the dataset
root, so the tester never assumes that top-level test samples contain `coord`
or `color` after fragment construction/inverse mapping.
"""

import json
import os
import sys
import time

# Allow this file to be executed directly as
#   python visilization/test_gfs_vis.py ...
# from any working directory without requiring an editable installation of
# pointcept.  The repository root is the parent of the visilization directory.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

import pointcept.utils.comm as comm
from pointcept.datasets import collate_fn
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.launch import launch
from pointcept.engines.test import GFSSemSegTester
from pointcept.utils.logger import get_root_logger
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    make_dirs,
)


class SafeGFSSemSegTester(GFSSemSegTester):
    """GFSSemSegTester variant that saves GT/pred only for visualization."""

    def one_regis_test(self, **kwargs):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Safe GFS Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        comm.synchronize()
        record = {}

        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")

            pred = torch.zeros(
                (segment.size, self.cfg.data.num_base_novels)
            ).cuda()

            for i in range(len(fragment_list)):
                input_dict = collate_fn(fragment_list[i : i + 1])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)

                idx_part = input_dict["index"]
                with torch.no_grad():
                    pred_part = self.model(input_dict, **kwargs)["seg_logits"]
                    pred_part = F.softmax(pred_part, -1)
                    if self.cfg.empty_cache:
                        torch.cuda.empty_cache()
                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs:be], :] += pred_part[bs:be]
                        bs = be

                logger.info(
                    "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                        idx + 1,
                        len(self.test_loader),
                        data_name=data_name,
                        batch_idx=i,
                        batch_num=len(fragment_list),
                    )
                )

            pred = pred.max(1)[1].data.cpu().numpy()
            if "origin_segment" in data_dict:
                if "inverse" not in data_dict:
                    raise KeyError(
                        f"origin_segment exists but inverse is missing for {data_name}"
                    )
                pred = pred[data_dict["inverse"]]
                segment = data_dict["origin_segment"]

            segment = np.asarray(segment).reshape(-1)
            pred = np.asarray(pred).reshape(-1)
            if pred.shape != segment.shape:
                raise RuntimeError(
                    f"Full-resolution prediction/GT mismatch for {data_name}: "
                    f"pred={pred.shape}, gt={segment.shape}"
                )

            intersection, union, target = intersection_and_union(
                pred,
                segment,
                self.cfg.data.num_base_novels,
                self.cfg.data.ignore_index,
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection,
                union=union,
                target=target,
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask]) if np.any(mask) else 0.0
            acc = sum(intersection) / (sum(target) + 1e-10)
            m_iou = np.mean(
                intersection_meter.sum / (union_meter.sum + 1e-10)
            )
            m_acc = np.mean(
                intersection_meter.sum / (target_meter.sum + 1e-10)
            )

            if self.cfg.vis is not None:
                scene_dir = os.path.join(self.cfg.vis, data_name)
                os.makedirs(scene_dir, exist_ok=True)
                # Do NOT save query.npy here. Coordinates/colors are not a
                # reliable top-level test field after fragment/inverse handling.
                np.save(
                    os.path.join(scene_dir, "query_label.npy"),
                    segment.astype(np.int64, copy=False),
                )
                np.save(
                    os.path.join(scene_dir, "query_pred.npy"),
                    pred.astype(np.int64, copy=False),
                )

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} Batch {:.3f} ({:.3f}) "
                "Accuracy {:.4f} ({:.4f}) mIoU {:.4f} ({:.4f})".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time.val,
                    batch_time.avg,
                    acc,
                    m_acc,
                    iou,
                    m_iou,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if not comm.is_main_process():
            return None

        record = {}
        for r in record_sync:
            record.update(r)

        intersection = np.sum(
            [meters["intersection"] for meters in record.values()], axis=0
        )
        union = np.sum(
            [meters["union"] for meters in record.values()], axis=0
        )
        target = np.sum(
            [meters["target"] for meters in record.values()], axis=0
        )

        iou_class = intersection / (union + 1e-10)
        accuracy_class = intersection / (target + 1e-10)
        m_iou = float(np.mean(iou_class))
        m_acc = float(np.mean(accuracy_class))
        all_acc = float(sum(intersection) / (sum(target) + 1e-10))

        base_iou_list = iou_class[: self.cfg.data.num_bases]
        novel_iou_list = iou_class[self.cfg.data.num_bases :]
        base_iou = float(np.mean(base_iou_list))
        novel_iou = float(np.mean(novel_iou_list))
        hm_iou = float(
            2 * base_iou * novel_iou / (base_iou + novel_iou + 1e-10)
        )

        class_names = (
            self.test_loader.dataset.base_class_names
            + self.test_loader.dataset.novel_class_names
        )
        logger.info(
            "Val result: mIoU/Base/Novel/HM/mAcc/allAcc "
            "{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(
                m_iou, base_iou, novel_iou, hm_iou, m_acc, all_acc
            )
        )
        for i, name in enumerate(class_names):
            logger.info(
                "Class_{} - {} Result: iou/accuracy {:.4f}/{:.4f}".format(
                    i, name, iou_class[i], accuracy_class[i]
                )
            )

        metric_record = {
            "mIoU": m_iou,
            "base_mIoU": base_iou,
            "novel_mIoU": novel_iou,
            "hm_mIoU": hm_iou,
            "mAcc": m_acc,
            "allAcc": all_acc,
            "class_iou": np.asarray(iou_class).astype(float).tolist(),
            "num_scenes": len(record),
        }
        metric_path = os.path.join(self.cfg.save_path, "evaluation_metrics.json")
        with open(metric_path, "w", encoding="utf-8") as f:
            json.dump(metric_record, f, indent=2, sort_keys=True)
        logger.info(f"Saved evaluation metrics: {metric_path}")
        logger.info("<<<<<<<<<<<<<<<< End Safe GFS Evaluation <<<<<<<<<<<<<<<<")

        return m_iou, base_iou, novel_iou, hm_iou, iou_class


def main_worker(cfg):
    cfg = default_setup(cfg)
    tester = SafeGFSSemSegTester(cfg=cfg)
    tester.test()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
