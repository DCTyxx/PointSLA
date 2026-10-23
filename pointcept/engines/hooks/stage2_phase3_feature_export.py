"""Best-checkpoint real/virtual feature export for Stage-2 Phase-3.

The exporter is intentionally post-training: it reloads the validation-HM best
checkpoint and runs one deterministic snapshot pass over the Phase-3 train and
validation datasets.  Therefore saved feature pairs always correspond to the
best selected model rather than an arbitrary training epoch.

Each split is written as memory-mappable .npy arrays:

    features.npy                float32 [N, D]  all trusted real Base+Novel features
    labels.npy                  int64   [N]
    scene_ids.npy               int64   [N]
    point_ids.npy               int64   [N]     post-transform point id within scene
    virtual_features.npy        float32 [V, D]  V=M*A for virtual modes, 0 for real_only
    directions.npy              float32 [V, D]  actual unit directions
    virtual_anchor_indices.npy  int64   [V]     row in features.npy for each virtual sample
    virtual_labels.npy          int64   [V]     convenience copy of anchor labels
    scene_names.json                    scene_id -> host-scene name
    metadata.json

Virtual anchors are trusted Novel points only, matching Phase-3 augmentation.
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
from pointcept.datasets import collate_fn, point_collate_fn
from pointcept.models.utils.structure import Point

from .builder import HOOKS
from .default import HookBase


class _NamedDataset(torch.utils.data.Dataset):
    """Expose scene name after the original transform/Collect pipeline."""

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
class Stage2Phase3BestFeaturePairExporter(HookBase):
    """Save train/val real-virtual feature pairs from model_best only."""

    def __init__(
        self,
        output_dir="best_feature_pairs",
        export_train=True,
        export_val=True,
        export_seed=3407,
        num_workers=0,
    ):
        self.output_dir = str(output_dir)
        self.export_train = bool(export_train)
        self.export_val = bool(export_val)
        self.export_seed = int(export_seed)
        self.num_workers = int(num_workers)
        if self.num_workers != 0:
            raise ValueError(
                "Feature-pair export currently requires num_workers=0 so the "
                "two deterministic passes see exactly the same transformed points."
            )

    @staticmethod
    def _set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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
    def _make_memmap(path, dtype, shape):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if int(np.prod(shape)) == 0:
            np.save(path, np.empty(shape, dtype=dtype))
            return None
        return np.lib.format.open_memmap(
            path, mode="w+", dtype=dtype, shape=shape
        )

    def _build_loader(self, dataset, batch_size, is_train):
        wrapped = _NamedDataset(dataset)
        if is_train:
            fn = partial(point_collate_fn, mix_prob=0.0)
        else:
            fn = collate_fn
        return torch.utils.data.DataLoader(
            wrapped,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=fn,
            drop_last=False,
            persistent_workers=False,
        )

    @staticmethod
    def _batch_identity(batch, scene_to_id):
        if "name" not in batch:
            raise KeyError("Feature exporter requires per-sample scene names")
        if "offset" not in batch:
            raise KeyError("Feature exporter requires point offsets")

        names = [str(v) for v in batch["name"]]
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
                raise KeyError(f"Unknown scene name during export: {name}")
            scene_parts.append(
                torch.full((count,), int(scene_to_id[name]), dtype=torch.long)
            )
            # This id tracks the exact post-transform point consumed by PTv3.
            point_parts.append(torch.arange(count, dtype=torch.long))
            start = end

        if not scene_parts:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        return torch.cat(scene_parts), torch.cat(point_parts)

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

    def _count_split(self, dataset, batch_size, seed, num_base, virtual_samples, use_virtual):
        self._set_seed(seed)
        loader = self._build_loader(dataset, batch_size, is_train=False)
        n_real = 0
        n_anchor = 0
        for batch in loader:
            labels = batch["segment"].long()
            valid = labels >= 0
            trusted = labels[valid]
            n_real += int(valid.sum().item())
            n_anchor += int((trusted >= num_base).sum().item())
        n_virtual = n_anchor * virtual_samples if use_virtual else 0
        return n_real, n_anchor, n_virtual

    def _export_split(
        self,
        raw_model,
        dataset,
        split_name,
        batch_size,
        seed,
        output_root,
        is_train,
        best_path,
    ):
        mode = str(raw_model.augmentation_mode)
        use_virtual = mode != "real_only"
        num_base = int(raw_model.num_base_classes)
        virtual_samples = int(raw_model.virtual_samples)
        feature_dim = int(raw_model.feature_dim)

        # First pass only counts the deterministic transformed snapshot so the
        # second pass can write directly to fixed-size memory maps without ever
        # materializing the whole dataset in host RAM.
        self._set_seed(seed)
        count_loader = self._build_loader(dataset, batch_size, is_train=is_train)
        n_real = 0
        n_anchor = 0
        for batch in count_loader:
            labels = batch["segment"].long()
            valid = labels >= 0
            trusted = labels[valid]
            n_real += int(valid.sum().item())
            n_anchor += int((trusted >= num_base).sum().item())
        n_virtual = n_anchor * virtual_samples if use_virtual else 0

        split_root = os.path.join(output_root, split_name)
        if os.path.isdir(split_root):
            shutil.rmtree(split_root)
        os.makedirs(split_root, exist_ok=True)

        features_mm = self._make_memmap(
            os.path.join(split_root, "features.npy"), np.float32, (n_real, feature_dim)
        )
        labels_mm = self._make_memmap(
            os.path.join(split_root, "labels.npy"), np.int64, (n_real,)
        )
        scene_ids_mm = self._make_memmap(
            os.path.join(split_root, "scene_ids.npy"), np.int64, (n_real,)
        )
        point_ids_mm = self._make_memmap(
            os.path.join(split_root, "point_ids.npy"), np.int64, (n_real,)
        )
        virtual_mm = self._make_memmap(
            os.path.join(split_root, "virtual_features.npy"),
            np.float32,
            (n_virtual, feature_dim),
        )
        directions_mm = self._make_memmap(
            os.path.join(split_root, "directions.npy"),
            np.float32,
            (n_virtual, feature_dim),
        )
        anchor_mm = self._make_memmap(
            os.path.join(split_root, "virtual_anchor_indices.npy"),
            np.int64,
            (n_virtual,),
        )
        virtual_labels_mm = self._make_memmap(
            os.path.join(split_root, "virtual_labels.npy"),
            np.int64,
            (n_virtual,),
        )

        scene_names = self._unique_scene_names(dataset)
        scene_to_id = {name: idx for idx, name in enumerate(scene_names)}
        with open(os.path.join(split_root, "scene_names.json"), "w", encoding="utf-8") as f:
            json.dump(scene_names, f, indent=2, ensure_ascii=False)

        # Rewind every relevant RNG. With num_workers=0 this reproduces exactly
        # the same transformed snapshot counted above.
        self._set_seed(seed)
        loader = self._build_loader(dataset, batch_size, is_train=is_train)
        real_cursor = 0
        virtual_cursor = 0
        raw_model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                scene_ids_full, point_ids_full = self._batch_identity(
                    batch, scene_to_id
                )
                labels_cpu = batch["segment"].long()
                valid_cpu = labels_cpu >= 0
                model_input = self._move_model_input(batch)

                with torch.cuda.amp.autocast(enabled=False):
                    point = raw_model.backbone(Point(model_input))
                    feature = point.feat if isinstance(point, Point) else point
                    feature = feature.detach().float()

                labels_gpu = model_input["segment"].long().to(feature.device)
                valid_gpu = labels_gpu >= 0
                real_feature = feature[valid_gpu]
                real_labels = labels_gpu[valid_gpu]
                count = int(real_feature.shape[0])
                end_real = real_cursor + count

                if count:
                    features_mm[real_cursor:end_real] = (
                        real_feature.detach().cpu().numpy().astype(np.float32, copy=False)
                    )
                    labels_mm[real_cursor:end_real] = (
                        real_labels.detach().cpu().numpy().astype(np.int64, copy=False)
                    )
                    scene_ids_mm[real_cursor:end_real] = (
                        scene_ids_full[valid_cpu].numpy().astype(np.int64, copy=False)
                    )
                    point_ids_mm[real_cursor:end_real] = (
                        point_ids_full[valid_cpu].numpy().astype(np.int64, copy=False)
                    )

                novel_mask = real_labels >= num_base
                if use_virtual and bool(novel_mask.any().item()):
                    novel_feature = real_feature[novel_mask]
                    novel_labels = real_labels[novel_mask]
                    virtual_feature, step = raw_model._sample_virtual_novel_features(
                        novel_feature, novel_labels
                    )
                    directions = F.normalize(step, p=2, dim=-1, eps=1e-8).reshape(
                        -1, feature_dim
                    )
                    virtual_count = int(virtual_feature.shape[0])
                    end_virtual = virtual_cursor + virtual_count
                    local_anchor = torch.nonzero(
                        novel_mask, as_tuple=False
                    ).flatten()
                    global_anchor = (
                        local_anchor + real_cursor
                    ).repeat_interleave(virtual_samples)
                    repeated_labels = novel_labels.repeat_interleave(virtual_samples)

                    virtual_mm[virtual_cursor:end_virtual] = (
                        virtual_feature.detach().cpu().numpy().astype(np.float32, copy=False)
                    )
                    directions_mm[virtual_cursor:end_virtual] = (
                        directions.detach().cpu().numpy().astype(np.float32, copy=False)
                    )
                    anchor_mm[virtual_cursor:end_virtual] = (
                        global_anchor.detach().cpu().numpy().astype(np.int64, copy=False)
                    )
                    virtual_labels_mm[virtual_cursor:end_virtual] = (
                        repeated_labels.detach().cpu().numpy().astype(np.int64, copy=False)
                    )
                    virtual_cursor = end_virtual

                real_cursor = end_real
                if (batch_idx + 1) % 20 == 0:
                    self.trainer.logger.info(
                        f"Feature export [{split_name}] {batch_idx + 1}/{len(loader)} "
                        f"real={real_cursor}/{n_real}, virtual={virtual_cursor}/{n_virtual}"
                    )

        if real_cursor != n_real or virtual_cursor != n_virtual:
            raise RuntimeError(
                f"Deterministic feature-export count mismatch for {split_name}: "
                f"real {real_cursor}/{n_real}, virtual {virtual_cursor}/{n_virtual}."
            )

        for array in (
            features_mm,
            labels_mm,
            scene_ids_mm,
            point_ids_mm,
            virtual_mm,
            directions_mm,
            anchor_mm,
            virtual_labels_mm,
        ):
            if array is not None:
                array.flush()

        metadata = {
            "split": split_name,
            "augmentation_mode": mode,
            "best_checkpoint": best_path,
            "feature_dim": feature_dim,
            "num_real_features": n_real,
            "num_novel_anchors": n_anchor,
            "num_virtual_features": n_virtual,
            "virtual_samples_per_anchor": virtual_samples if use_virtual else 0,
            "virtual_radius": float(raw_model.virtual_radius),
            "num_base_classes": num_base,
            "features_scope": "all trusted Base+Novel points after Phase3 preprocessing",
            "virtual_scope": "trusted Novel anchors only",
            "scene_id_definition": "integer index into scene_names.json; host scene consumed by PTv3",
            "point_id_definition": "zero-based point index within the post-transform host scene snapshot",
            "virtual_anchor_indices_definition": "row index into features.npy",
            "directions_definition": "unit feature-space direction actually used to generate each virtual feature",
            "export_seed": seed,
            "batch_size": int(batch_size),
        }
        with open(os.path.join(split_root, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        self.trainer.logger.info(
            f"Feature export complete [{split_name}] -> {split_root}; "
            f"real={n_real}, anchors={n_anchor}, virtual={n_virtual}"
        )
        return metadata

    def after_train(self):
        # Other ranks wait while rank 0 performs a single full-dataset export.
        comm.synchronize()
        if not comm.is_main_process():
            comm.synchronize()
            return

        cfg = self.trainer.cfg
        best_path = os.path.join(cfg.save_path, "model", "model_best.pth")
        if not os.path.isfile(best_path):
            raise FileNotFoundError(
                f"Cannot export feature pairs; best Phase3 checkpoint missing: {best_path}"
            )

        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Export Best Phase-3 Feature Pairs >>>>>>>>>>>>>>>>"
        )
        checkpoint = torch.load(
            best_path,
            map_location=lambda storage, loc: storage.cuda(),
        )
        # Checkpoint keys were saved from the same wrapped model used here.
        self.trainer.model.load_state_dict(checkpoint["state_dict"], strict=True)
        raw_model = (
            self.trainer.model.module
            if comm.get_world_size() > 1
            else self.trainer.model
        )
        raw_model.eval()

        output_root = os.path.join(cfg.save_path, self.output_dir)
        if os.path.isdir(output_root):
            shutil.rmtree(output_root)
        os.makedirs(output_root, exist_ok=True)

        train_bs = int(getattr(cfg, "batch_size_per_gpu", 1))
        val_bs = int(getattr(cfg, "batch_size_val_per_gpu", 1))
        summaries = {}
        if self.export_train:
            summaries["train"] = self._export_split(
                raw_model=raw_model,
                dataset=self.trainer.train_loader.dataset,
                split_name="train",
                batch_size=train_bs,
                seed=self.export_seed,
                output_root=output_root,
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
                output_root=output_root,
                is_train=False,
                best_path=best_path,
            )

        run_metadata = {
            "regis": str(getattr(cfg, "phase3_regis_name", "unknown")),
            "augmentation_mode": str(raw_model.augmentation_mode),
            "selected_checkpoint": "best_validation_HM",
            "best_validation_HM": float(checkpoint.get("best_metric_value", float("nan"))),
            "splits": summaries,
        }
        with open(os.path.join(output_root, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(run_metadata, f, indent=2, sort_keys=True)

        torch.cuda.empty_cache()
        self.trainer.logger.info(
            f"Best Phase-3 feature pairs saved to: {output_root}"
        )
        self.trainer.logger.info(
            "<<<<<<<<<<<<<<<< Feature Pair Export Complete <<<<<<<<<<<<<<<<"
        )
        comm.synchronize()
