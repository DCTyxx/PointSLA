"""Final-best, validation-only feature export for Stage-2 Phase-3.

This hook is intentionally optimized for the ScanNet++ K=1 direction ablation:

1. regis1..regis5 are trained/evaluated normally with NO feature export;
2. only the final trigger run (default regis5) compares all precise-HM results;
3. the single best registration checkpoint is loaded;
4. validation data are scanned ONCE;
5. only trusted Novel anchors are saved (Base and label=-1 points are skipped);
6. data are written batch-by-batch as uncompressed NPZ shards.

Therefore each direction mode produces exactly one full feature snapshot and
never performs the former count-pass + write-pass memmap export.

Output layout:

  exp/scannetpp/Direction_Ablation_k1/<mode>/best_feature_pairs/
    cache_state.json
    best_regis.json
    val/
      manifest.json
      metadata.json
      scene_names.json
      shards/
        shard_00000.npz
        ...

Each non-empty shard contains:

  features          float32 [A, D]      trusted Novel real anchors only
  labels            int64   [A]
  scene_ids         int32   [A]
  point_ids         int32   [A]
  virtual_features  float32 [M*A, D]    empty for real_only
  directions        float32 [M*A, D]    actual unit directions, empty for real_only

Because virtual rows are stored anchor-major with a fixed M, the real/virtual
mapping is implicit inside each shard:

  virtual_anchor_index = arange(A).repeat_interleave(M)

so virtual_labels and virtual_anchor_indices are deliberately not written.
"""

import json
import os
import random
import shutil
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

import pointcept.utils.comm as comm
from pointcept.datasets import collate_fn
from pointcept.models.utils.structure import Point

from .builder import HOOKS
from .default import HookBase


class _NamedDataset(torch.utils.data.Dataset):
    """Expose scene names without modifying the original dataset pipeline."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = dict(self.dataset[idx])
        item["name"] = self.dataset.get_data_name(idx)
        return item

    def get_data_name(self, idx):
        return self.dataset.get_data_name(idx)


@HOOKS.register_module()
class Stage2Phase3CachedBestFeaturePairExporter(HookBase):
    """Export only the final precise-HM-best registration, validation only."""

    EXPORT_VERSION = "final_best_val_novel_sharded_v2"

    def __init__(
        self,
        output_dir="best_feature_pairs",
        export_train=False,
        export_val=True,
        export_seed=3407,
        num_workers=0,
        metric_key="hm_mIoU",
        expected_regis=5,
        final_trigger_regis="regis5",
    ):
        self.output_dir = str(output_dir)
        self.export_train = bool(export_train)
        self.export_val = bool(export_val)
        self.export_seed = int(export_seed)
        self.num_workers = int(num_workers)
        self.metric_key = str(metric_key)
        self.expected_regis = int(expected_regis)
        self.final_trigger_regis = str(final_trigger_regis)

        if self.export_train:
            raise ValueError(
                "Optimized feature export is validation-only; set export_train=False."
            )
        if not self.export_val:
            raise ValueError("At least validation export must remain enabled.")
        if self.num_workers != 0:
            raise ValueError(
                "Feature export uses num_workers=0 for deterministic one-pass snapshots."
            )
        if self.expected_regis <= 0:
            raise ValueError("expected_regis must be positive")

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

    @staticmethod
    def _set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _move_model_input(batch):
        result = {}
        for key, value in batch.items():
            if key == "name":
                continue
            if isinstance(value, torch.Tensor):
                result[key] = value.cuda(non_blocking=True)
            else:
                result[key] = value
        return result

    @staticmethod
    def _unique_scene_names(dataset):
        names = []
        seen = set()
        for idx in range(len(dataset)):
            name = str(dataset.get_data_name(idx))
            if name not in seen:
                seen.add(name)
                names.append(name)
        return names

    @staticmethod
    def _batch_identity(batch, scene_to_id):
        if "name" not in batch or "offset" not in batch:
            raise KeyError("Exporter requires name and offset in every validation batch")

        names = [str(x) for x in batch["name"]]
        offsets = batch["offset"].detach().cpu().long().tolist()
        if len(names) != len(offsets):
            raise RuntimeError(
                f"Scene-name/offset mismatch: names={len(names)}, offsets={len(offsets)}"
            )

        scene_parts = []
        point_parts = []
        start = 0
        for name, end in zip(names, offsets):
            end = int(end)
            count = end - start
            if count < 0:
                raise RuntimeError("Offsets must be monotonically increasing")
            if name not in scene_to_id:
                raise KeyError(f"Unknown scene name: {name}")
            scene_parts.append(
                torch.full((count,), int(scene_to_id[name]), dtype=torch.int32)
            )
            # Track the exact post-transform point consumed by PTv3.
            point_parts.append(torch.arange(count, dtype=torch.int32))
            start = end

        if not scene_parts:
            return torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.int32)
        return torch.cat(scene_parts), torch.cat(point_parts)

    def _build_val_loader(self, dataset, batch_size):
        return torch.utils.data.DataLoader(
            _NamedDataset(dataset),
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
            persistent_workers=False,
        )

    def _resolve_best_radius_and_seed(self, mode_root, best_regis, raw_model):
        """Restore non-state_dict augmentation settings of the selected run."""
        suffix = best_regis.replace("regis", "")
        try:
            regis_idx = int(suffix)
        except ValueError as exc:
            raise ValueError(f"Cannot parse registration index from {best_regis}") from exc

        # Launcher convention: regis1..5 -> seeds 10,20,...,50.
        raw_model.random_subspace_seed = regis_idx * 10
        raw_model._random_subspace_cache = None

        radius_file = os.path.join(mode_root, best_regis, "base_only_radius.txt")
        if os.path.isfile(radius_file):
            text = open(radius_file, "r", encoding="utf-8").read().strip()
            if text:
                raw_model.virtual_radius = float(text)
        elif best_regis != self.final_trigger_regis:
            raise FileNotFoundError(
                f"Best registration radius file missing: {radius_file}"
            )
        # If best_regis is the current final trigger, cfg/model already carries
        # the exact radius passed by the launcher for this run.

    def _select_final_best(self, mode_root):
        records = []
        for idx in range(1, self.expected_regis + 1):
            regis = f"regis{idx}"
            path = os.path.join(mode_root, regis, "phase3_metrics_final.json")
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    "Final feature export requires all precise metrics first; "
                    f"missing {path}"
                )
            rec = dict(self._read_json(path))
            rec["regis"] = regis
            if self.metric_key not in rec:
                raise KeyError(
                    f"Metric {self.metric_key!r} missing from {path}"
                )
            records.append(rec)
        return max(records, key=lambda r: float(r[self.metric_key])), records

    def _export_val_sharded(
        self,
        raw_model,
        dataset,
        output_root,
        batch_size,
        best_path,
        best_regis,
    ):
        """One-pass, Novel-only validation export with uncompressed NPZ shards."""
        split_root = os.path.join(output_root, "val")
        shard_root = os.path.join(split_root, "shards")
        os.makedirs(shard_root, exist_ok=True)

        scene_names = self._unique_scene_names(dataset)
        scene_to_id = {name: idx for idx, name in enumerate(scene_names)}
        with open(
            os.path.join(split_root, "scene_names.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(scene_names, f, indent=2, ensure_ascii=False)

        self._set_seed(self.export_seed)
        loader = self._build_val_loader(dataset, batch_size)
        num_base = int(raw_model.num_base_classes)
        feature_dim = int(raw_model.feature_dim)
        virtual_samples = int(raw_model.virtual_samples)
        mode = str(raw_model.augmentation_mode)
        use_virtual = mode != "real_only"

        manifest = []
        total_anchor = 0
        total_virtual = 0
        shard_idx = 0
        raw_model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                scene_ids_full, point_ids_full = self._batch_identity(
                    batch, scene_to_id
                )
                labels_cpu = batch["segment"].long()
                model_input = self._move_model_input(batch)

                with torch.cuda.amp.autocast(enabled=False):
                    point = raw_model.backbone(Point(model_input))
                    feature = point.feat if isinstance(point, Point) else point
                    feature = feature.detach().float()

                labels_gpu = model_input["segment"].long().to(feature.device)
                # Save only trusted Novel anchors. This simultaneously removes
                # all Base points and all ignore/background labels (-1).
                novel_mask_gpu = labels_gpu >= num_base
                novel_mask_cpu = labels_cpu >= num_base
                if not bool(novel_mask_gpu.any().item()):
                    continue

                real_feature = feature[novel_mask_gpu]
                real_labels = labels_gpu[novel_mask_gpu]
                scene_ids = scene_ids_full[novel_mask_cpu]
                point_ids = point_ids_full[novel_mask_cpu]
                anchor_count = int(real_feature.shape[0])

                if use_virtual:
                    virtual_feature, step = raw_model._sample_virtual_novel_features(
                        real_feature, real_labels
                    )
                    directions = F.normalize(
                        step, p=2, dim=-1, eps=1e-8
                    ).reshape(-1, feature_dim)
                else:
                    virtual_feature = real_feature.new_empty((0, feature_dim))
                    directions = real_feature.new_empty((0, feature_dim))

                virtual_count = int(virtual_feature.shape[0])
                shard_name = f"shard_{shard_idx:05d}.npz"
                shard_path = os.path.join(shard_root, shard_name)
                # np.savez is ZIP_STORED (uncompressed): portable but avoids the
                # large CPU cost of savez_compressed.
                np.savez(
                    shard_path,
                    features=real_feature.cpu().numpy().astype(np.float32, copy=False),
                    labels=real_labels.cpu().numpy().astype(np.int64, copy=False),
                    scene_ids=scene_ids.numpy().astype(np.int32, copy=False),
                    point_ids=point_ids.numpy().astype(np.int32, copy=False),
                    virtual_features=virtual_feature.cpu().numpy().astype(
                        np.float32, copy=False
                    ),
                    directions=directions.cpu().numpy().astype(np.float32, copy=False),
                )

                manifest.append(
                    {
                        "file": os.path.join("shards", shard_name),
                        "batch_index": int(batch_idx),
                        "num_anchors": anchor_count,
                        "num_virtual": virtual_count,
                    }
                )
                total_anchor += anchor_count
                total_virtual += virtual_count
                shard_idx += 1

                if shard_idx % 20 == 0:
                    self.trainer.logger.info(
                        "Feature export [val] shards={} anchors={} virtual={}".format(
                            shard_idx, total_anchor, total_virtual
                        )
                    )

        with open(os.path.join(split_root, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

        metadata = {
            "export_version": self.EXPORT_VERSION,
            "split": "val",
            "augmentation_mode": mode,
            "best_regis": best_regis,
            "best_checkpoint": best_path,
            "feature_dim": feature_dim,
            "num_novel_anchors": total_anchor,
            "num_virtual_features": total_virtual,
            "num_shards": shard_idx,
            "virtual_samples_per_anchor": virtual_samples if use_virtual else 0,
            "virtual_radius": float(raw_model.virtual_radius),
            "num_base_classes": num_base,
            "features_scope": "trusted Novel validation anchors only; Base and label=-1 omitted",
            "scene_ids_dtype": "int32",
            "point_ids_dtype": "int32",
            "labels_dtype": "int64",
            "features_dtype": "float32",
            "directions_dtype": "float32",
            "scene_id_definition": "integer index into scene_names.json",
            "point_id_definition": "zero-based point index within the post-transform validation scene snapshot",
            "virtual_mapping": (
                "anchor-major within each shard: local_anchor_index = "
                "arange(num_anchors).repeat_interleave(M)"
            ),
            "directions_definition": "unit feature-space direction actually used for each virtual feature",
            "single_pass": True,
            "compression": "none (numpy savez / ZIP_STORED)",
            "export_seed": self.export_seed,
            "batch_size": int(batch_size),
        }
        self._write_json(os.path.join(split_root, "metadata.json"), metadata)
        return metadata

    def after_train(self):
        comm.synchronize()
        if not comm.is_main_process():
            comm.synchronize()
            return

        cfg = self.trainer.cfg
        current_regis = str(getattr(cfg, "phase3_regis_name", "unknown"))
        mode_root = os.path.dirname(os.path.normpath(cfg.save_path))
        status_path = os.path.join(cfg.save_path, "feature_export_cache_status.json")

        # regis1..regis4 only leave a tiny status record; absolutely no feature I/O.
        if current_regis != self.final_trigger_regis:
            self._write_json(
                status_path,
                {
                    "export_version": self.EXPORT_VERSION,
                    "regis": current_regis,
                    "exported": False,
                    "reason": "deferred_until_all_registrations_finish",
                },
            )
            self.trainer.logger.info(
                f"Defer feature export for {current_regis}; final export runs on "
                f"{self.final_trigger_regis}."
            )
            comm.synchronize()
            return

        best, records = self._select_final_best(mode_root)
        best_regis = str(best["regis"])
        best_metric = float(best[self.metric_key])
        best_path = os.path.join(mode_root, best_regis, "model", "model_best.pth")
        if not os.path.isfile(best_path):
            raise FileNotFoundError(f"Best Phase-3 checkpoint missing: {best_path}")

        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Final-Best Validation Feature Export >>>>>>>>>>>>>>>>"
        )
        self.trainer.logger.info(
            f"Selected {best_regis} by precise {self.metric_key}={best_metric:.6f}; "
            "exporting validation Novel anchors once."
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
        self._resolve_best_radius_and_seed(mode_root, best_regis, raw_model)
        raw_model.eval()

        cache_root = os.path.join(mode_root, self.output_dir)
        temp_root = os.path.join(mode_root, f".{self.output_dir}.tmp_final")
        if os.path.isdir(temp_root):
            shutil.rmtree(temp_root)
        os.makedirs(temp_root, exist_ok=True)

        try:
            val_bs = int(getattr(cfg, "batch_size_val_per_gpu", 1))
            val_meta = self._export_val_sharded(
                raw_model=raw_model,
                dataset=self.trainer.val_loader.dataset,
                output_root=temp_root,
                batch_size=val_bs,
                best_path=best_path,
                best_regis=best_regis,
            )

            state = {
                "export_version": self.EXPORT_VERSION,
                "dataset": "scannetpp",
                "k_shot": 1,
                "augmentation_mode": str(raw_model.augmentation_mode),
                "best_regis": best_regis,
                "metric_key": self.metric_key,
                "best_metric": best_metric,
                "selected_checkpoint": "best_validation_HM",
                "checkpoint": best_path,
                "val": val_meta,
                "all_precise_metrics": [
                    {
                        "regis": r["regis"],
                        self.metric_key: float(r[self.metric_key]),
                    }
                    for r in records
                ],
            }
            self._write_json(os.path.join(temp_root, "cache_state.json"), state)
            self._write_json(
                os.path.join(temp_root, "best_regis.json"),
                {
                    "best_regis": best_regis,
                    "selection_metric": f"precise {self.metric_key}",
                    "best_metric": best_metric,
                    "augmentation_mode": str(raw_model.augmentation_mode),
                },
            )

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

        self._write_json(
            status_path,
            {
                "export_version": self.EXPORT_VERSION,
                "regis": current_regis,
                "exported": True,
                "reason": "all_registrations_complete_final_best_export",
                "best_regis": best_regis,
                "best_metric": best_metric,
                "cache_root": cache_root,
            },
        )
        torch.cuda.empty_cache()
        self.trainer.logger.info(f"Final feature snapshot saved to: {cache_root}")
        self.trainer.logger.info(
            "<<<<<<<<<<<<<<<< Final-Best Feature Export Complete <<<<<<<<<<<<<<<<"
        )
        comm.synchronize()
