#!/usr/bin/env python3
"""Standalone Stage-2 Phase-3 validation feature-pair exporter.

Loads one explicit Phase-3 checkpoint and exports the same validation Novel
feature-pair layout used by Stage2Phase3CachedBestFeaturePairExporter:

  <output_dir>/
    cache_state.json
    best_regis.json
    val/
      manifest.json
      metadata.json
      scene_names.json
      shards/shard_*.npz

No training is performed.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from collections import OrderedDict
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pointcept.datasets import build_dataset
from pointcept.engines.hooks.stage2_phase3_feature_cache import (
    Stage2Phase3CachedBestFeaturePairExporter,
)
from pointcept.models import build_model
from pointcept.utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/scannetpp/semseg-pt-v3m1-0-stage2-phase3-k1-cdc-aug.py",
    )
    parser.add_argument(
        "--checkpoint",
        default="exp/scannetpp/split_phase3_k1/regis3/model/model_best.pth",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "exp/scannetpp/Direction_Ablation_k1/"
            "split_phase3_k1_regis3/best_feature_pairs"
        ),
    )
    parser.add_argument("--regis", default="regis3")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--export-seed", type=int, default=3407)
    parser.add_argument(
        "--augmentation-mode",
        default=None,
        choices=["cdc", "real_only", "isotropic", "random_subspace", "local_pca"],
    )
    parser.add_argument(
        "--virtual-radius",
        type=float,
        default=None,
        help=(
            "Optional override. If omitted, automatically read base_only_radius.txt "
            "beside the checkpoint run when available; otherwise use config value."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_checkpoint_strict(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint has no state_dict: {checkpoint_path}")

    state_dict = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("module."):
            key = key[7:]
        state_dict[key] = value
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def resolve_virtual_radius(args, checkpoint_path, model, logger):
    if args.virtual_radius is not None:
        radius = float(args.virtual_radius)
        source = "command_line"
    else:
        # checkpoint = <regis>/model/model_best.pth
        regis_root = os.path.dirname(os.path.dirname(checkpoint_path))
        radius_file = os.path.join(regis_root, "base_only_radius.txt")
        if os.path.isfile(radius_file):
            text = open(radius_file, "r", encoding="utf-8").read().strip()
            if not text:
                raise RuntimeError(f"Empty radius file: {radius_file}")
            radius = float(text)
            source = radius_file
        else:
            radius = float(model.virtual_radius)
            source = "config_default"

    if radius <= 0:
        raise ValueError(f"virtual_radius must be positive, got {radius}")
    model.virtual_radius = radius
    logger.info("Virtual radius: %.8f (%s)", radius, source)
    return radius, source


def main():
    args = parse_args()

    config_path = os.path.abspath(args.config)
    checkpoint_path = os.path.abspath(args.checkpoint)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    if os.path.exists(output_dir):
        if not args.force:
            raise FileExistsError(
                f"Output already exists: {output_dir}. Use --force to replace it."
            )
        shutil.rmtree(output_dir)

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    temp_dir = output_dir + ".tmp"
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("phase3-feature-export")

    cfg = Config.fromfile(config_path)

    logger.info("Building Phase-3 model")
    model = build_model(cfg.model).cuda()
    checkpoint = load_checkpoint_strict(model, checkpoint_path)

    if args.augmentation_mode is not None:
        model.augmentation_mode = args.augmentation_mode

    virtual_radius, radius_source = resolve_virtual_radius(
        args, checkpoint_path, model, logger
    )

    # Reproduce the registration-specific random-subspace seed convention used
    # by the direction-ablation launcher: regis1..5 -> 10,20,...,50.
    suffix = args.regis.replace("regis", "")
    try:
        regis_idx = int(suffix)
    except ValueError as exc:
        raise ValueError(f"Cannot parse registration index from {args.regis}") from exc
    model.random_subspace_seed = regis_idx * 10
    model._random_subspace_cache = None
    model.eval()

    logger.info("Building validation dataset")
    val_dataset = build_dataset(cfg.data.val)

    exporter = Stage2Phase3CachedBestFeaturePairExporter(
        output_dir="best_feature_pairs",
        export_train=False,
        export_val=True,
        export_seed=args.export_seed,
        num_workers=0,
        expected_regis=1,
        final_trigger_regis=args.regis,
    )
    # _export_val_sharded logs through self.trainer.logger. A tiny standalone
    # namespace is sufficient because no trainer state is otherwise required.
    exporter.trainer = SimpleNamespace(logger=logger)

    try:
        logger.info("Exporting validation Novel feature pairs")
        val_meta = exporter._export_val_sharded(
            raw_model=model,
            dataset=val_dataset,
            output_root=temp_dir,
            batch_size=args.batch_size,
            best_path=checkpoint_path,
            best_regis=args.regis,
        )

        epoch = checkpoint.get("epoch", None)
        state = {
            "export_version": exporter.EXPORT_VERSION,
            "dataset": "scannetpp",
            "k_shot": 1,
            "augmentation_mode": str(model.augmentation_mode),
            "best_regis": args.regis,
            "selection_metric": "explicit_checkpoint",
            "selected_checkpoint": checkpoint_path,
            "checkpoint_epoch": epoch,
            "config": config_path,
            "virtual_radius": virtual_radius,
            "virtual_radius_source": radius_source,
            "val": val_meta,
        }
        write_json(os.path.join(temp_dir, "cache_state.json"), state)
        write_json(
            os.path.join(temp_dir, "best_regis.json"),
            {
                "best_regis": args.regis,
                "selection_metric": "explicit_checkpoint",
                "augmentation_mode": str(model.augmentation_mode),
                "checkpoint": checkpoint_path,
                "virtual_radius": virtual_radius,
                "virtual_radius_source": radius_source,
            },
        )

        os.replace(temp_dir, output_dir)
    except Exception:
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
        raise

    logger.info("Feature-pair export complete: %s", output_dir)
    logger.info(
        "Novel anchors=%s, virtual features=%s, shards=%s",
        val_meta["num_novel_anchors"],
        val_meta["num_virtual_features"],
        val_meta["num_shards"],
    )


if __name__ == "__main__":
    main()
