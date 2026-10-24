"""Best-so-far feature cache for Stage-2 Phase-3.

Unlike Stage2Phase3BestFeaturePairExporter, which exports one full train/val
snapshot for every registration, this hook compares the current registration's
precise HM against a mode-level cache and exports features only when the current
run becomes the best-so-far registration.

The cache lives one directory above the regis run, e.g.

  exp/scannetpp/Direction_Ablation_k1/isotropic/best_feature_pairs/

and is overwritten whenever a later registration obtains a higher precise HM.
At any time there is therefore at most one full feature snapshot on disk.
"""

import json
import os
import shutil

import torch

import pointcept.utils.comm as comm

from .builder import HOOKS
from .stage2_phase3_feature_export import Stage2Phase3BestFeaturePairExporter


@HOOKS.register_module()
class Stage2Phase3CachedBestFeaturePairExporter(
    Stage2Phase3BestFeaturePairExporter
):
    """Export train/val feature pairs only for the best-so-far precise-HM run."""

    def __init__(
        self,
        output_dir="best_feature_pairs",
        export_train=True,
        export_val=True,
        export_seed=3407,
        num_workers=0,
        metric_key="hm_mIoU",
    ):
        super().__init__(
            output_dir=output_dir,
            export_train=export_train,
            export_val=export_val,
            export_seed=export_seed,
            num_workers=num_workers,
        )
        self.metric_key = str(metric_key)

    @staticmethod
    def _read_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_json(path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, path)

    def after_train(self):
        # PreciseEvaluator is ordered before this hook and writes the final
        # precise metrics. Only rank 0 owns the shared feature cache.
        comm.synchronize()
        if not comm.is_main_process():
            comm.synchronize()
            return

        cfg = self.trainer.cfg
        regis = str(getattr(cfg, "phase3_regis_name", "unknown"))
        mode = str(
            getattr(
                self.trainer.model.module
                if comm.get_world_size() > 1
                else self.trainer.model,
                "augmentation_mode",
                "unknown",
            )
        )

        metric_path = os.path.join(cfg.save_path, "phase3_metrics_final.json")
        if not os.path.isfile(metric_path):
            raise FileNotFoundError(
                "Cached feature export requires precise Phase-3 metrics first: "
                f"{metric_path}"
            )
        metric_record = self._read_json(metric_path)
        if self.metric_key not in metric_record:
            raise KeyError(
                f"Metric {self.metric_key!r} missing from {metric_path}; "
                f"available={sorted(metric_record.keys())}"
            )
        current_metric = float(metric_record[self.metric_key])

        # cfg.save_path is .../<mode>/<regis>. Cache at .../<mode>/best_feature_pairs.
        mode_root = os.path.dirname(os.path.normpath(cfg.save_path))
        cache_root = os.path.join(mode_root, self.output_dir)
        cache_state_path = os.path.join(cache_root, "cache_state.json")
        status_path = os.path.join(cfg.save_path, "feature_export_cache_status.json")

        old_state = None
        if os.path.isfile(cache_state_path):
            old_state = self._read_json(cache_state_path)
        old_metric = (
            float(old_state.get("best_metric", float("-inf")))
            if old_state is not None
            else float("-inf")
        )

        # Ties keep the earlier cache to avoid redundant multi-GB I/O.
        if current_metric <= old_metric:
            status = {
                "regis": regis,
                "augmentation_mode": mode,
                "metric_key": self.metric_key,
                "current_metric": current_metric,
                "cached_best_metric": old_metric,
                "cached_best_regis": old_state.get("best_regis") if old_state else None,
                "exported": False,
                "reason": "not_better_than_cached_best",
                "cache_root": cache_root,
            }
            self._write_json(status_path, status)
            self.trainer.logger.info(
                "Skip feature export: current precise {}={:.6f} <= cached best "
                "{:.6f} ({})".format(
                    self.metric_key,
                    current_metric,
                    old_metric,
                    status["cached_best_regis"],
                )
            )
            comm.synchronize()
            return

        best_path = os.path.join(cfg.save_path, "model", "model_best.pth")
        if not os.path.isfile(best_path):
            raise FileNotFoundError(
                f"Cannot refresh best feature cache; checkpoint missing: {best_path}"
            )

        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Refresh Best Phase-3 Feature Cache >>>>>>>>>>>>>>>>"
        )
        self.trainer.logger.info(
            "New best-so-far registration: {} precise {} {:.6f} > {:.6f}".format(
                regis, self.metric_key, current_metric, old_metric
            )
        )

        checkpoint = torch.load(
            best_path,
            map_location=lambda storage, loc: storage.cuda(),
        )
        self.trainer.model.load_state_dict(checkpoint["state_dict"], strict=True)
        raw_model = (
            self.trainer.model.module
            if comm.get_world_size() > 1
            else self.trainer.model
        )
        raw_model.eval()

        # Export into a temporary sibling first. The previous valid cache remains
        # intact until the new snapshot has been written successfully.
        temp_root = os.path.join(mode_root, f".{self.output_dir}.tmp_{regis}")
        if os.path.isdir(temp_root):
            shutil.rmtree(temp_root)
        os.makedirs(temp_root, exist_ok=True)

        train_bs = int(getattr(cfg, "batch_size_per_gpu", 1))
        val_bs = int(getattr(cfg, "batch_size_val_per_gpu", 1))
        summaries = {}
        try:
            if self.export_train:
                summaries["train"] = self._export_split(
                    raw_model=raw_model,
                    dataset=self.trainer.train_loader.dataset,
                    split_name="train",
                    batch_size=train_bs,
                    seed=self.export_seed,
                    output_root=temp_root,
                    is_train=True,
                    best_path=best_path,
                )
            if self.export_val:
                summaries["val"] = self._export_split(
                    raw_model=raw_model,
                    dataset=self.trainer.val_loader.dataset,
                    split_name="val",
                    batch_size=val_bs,
                    seed=self.export_seed + 100000,
                    output_root=temp_root,
                    is_train=False,
                    best_path=best_path,
                )

            cache_state = {
                "dataset": "scannetpp",
                "k_shot": 1,
                "augmentation_mode": mode,
                "best_regis": regis,
                "metric_key": self.metric_key,
                "best_metric": current_metric,
                "selected_checkpoint": "best_validation_HM",
                "checkpoint": best_path,
                "splits": summaries,
            }
            self._write_json(
                os.path.join(temp_root, "cache_state.json"), cache_state
            )
            self._write_json(
                os.path.join(temp_root, "best_regis.json"),
                {
                    "best_regis": regis,
                    "selection_metric": f"precise {self.metric_key}",
                    "best_metric": current_metric,
                    "augmentation_mode": mode,
                },
            )

            # Replace only after successful completion of the new snapshot.
            old_root = cache_root + ".old"
            if os.path.isdir(old_root):
                shutil.rmtree(old_root)
            if os.path.isdir(cache_root):
                os.replace(cache_root, old_root)
            os.replace(temp_root, cache_root)
            if os.path.isdir(old_root):
                shutil.rmtree(old_root)
        except Exception:
            if os.path.isdir(temp_root):
                shutil.rmtree(temp_root)
            raise

        status = {
            "regis": regis,
            "augmentation_mode": mode,
            "metric_key": self.metric_key,
            "current_metric": current_metric,
            "previous_best_metric": old_metric,
            "exported": True,
            "reason": "new_best_so_far",
            "cache_root": cache_root,
        }
        self._write_json(status_path, status)

        torch.cuda.empty_cache()
        self.trainer.logger.info(
            f"Best feature cache refreshed: {cache_root}"
        )
        self.trainer.logger.info(
            "<<<<<<<<<<<<<<<< Feature Cache Refresh Complete <<<<<<<<<<<<<<<<"
        )
        comm.synchronize()
