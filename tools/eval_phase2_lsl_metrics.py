"""Post-hoc LSL diagnostics for an existing Stage-2 Phase-2 checkpoint.

This script does not update any parameter. It loads the final Phase-2 model and
runs the trusted Base + K-shot Novel training set once to report five LSL
metrics for Overall/Base/Novel anchors:

- TAS (higher is better)
- Intra-EVR (higher is better)
- Inter-EVR (lower is better)
- GDG = Intra-EVR - Inter-EVR (higher is better)
- Mean Principal Angle in degrees (lower is better)

The aggregation is anchor-weighted over all evaluated batches. The default
single-GPU batch size is 4, matching the per-GPU batch size of the standard
batch_size=12, 3-GPU Phase-2 runs; it can be overridden explicitly.
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from functools import partial

# Allow direct execution via `python tools/eval_phase2_lsl_metrics.py` from any
# working directory by making the repository root importable before importing
# the local `pointcept` package.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from pointcept.datasets import build_dataset, point_collate_fn
from pointcept.models import build_model
from pointcept.utils.config import Config
from pointcept.utils.env import set_seed


LSL_FIELDS = (
    "tas",
    "intra_evr",
    "inter_evr",
    "gdg",
    "principal_angle_deg",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Phase-2 config .py")
    parser.add_argument("--checkpoint", required=True, help="Phase-2 model_last.pth")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--k-shot", type=int, required=True, choices=(1, 5))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def metric_key(group, field):
    prefix = "train" if group == "overall" else group
    return f"{prefix}_{field}"


def load_checkpoint_strict(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint has no state_dict: {path}")
    weight = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("module."):
            key = key[7:]
        weight[key] = value
    info = model.load_state_dict(weight, strict=True)
    if info.missing_keys or info.unexpected_keys:
        raise RuntimeError(
            f"Strict Phase-2 load mismatch: missing={info.missing_keys}, "
            f"unexpected={info.unexpected_keys}"
        )
    return checkpoint


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = Config.fromfile(args.config)
    cfg.data.train.k_shot = args.k_shot
    cfg.data.train.seed = args.seed
    if hasattr(cfg, "epoch") and hasattr(cfg, "eval_epoch"):
        cfg.data.train.loop = cfg.epoch // cfg.eval_epoch

    device = torch.device(args.device)
    model = build_model(cfg.model).to(device)
    checkpoint = load_checkpoint_strict(model, args.checkpoint)
    model.eval()

    dataset = build_dataset(cfg.data.train)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(point_collate_fn, mix_prob=float(getattr(cfg, "mix_prob", 0.0))),
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    totals = {
        group: {"count": 0.0, **{field: 0.0 for field in LSL_FIELDS}}
        for group in ("overall", "base", "novel")
    }

    num_batches = 0
    with torch.no_grad():
        for input_dict in loader:
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.to(device, non_blocking=True)

            output = model(input_dict)
            base_count = float(output["num_base_anchors"].item())
            novel_count = float(output["num_novel_anchors"].item())
            counts = {
                "overall": base_count + novel_count,
                "base": base_count,
                "novel": novel_count,
            }

            for group in ("overall", "base", "novel"):
                count = counts[group]
                totals[group]["count"] += count
                if count <= 0.0:
                    continue
                for field in LSL_FIELDS:
                    value = float(output[metric_key(group, field)].item())
                    totals[group][field] += value * count
            num_batches += 1

    groups = {}
    for group in ("overall", "base", "novel"):
        count = totals[group]["count"]
        groups[group] = {
            "TAS": totals[group]["tas"] / count if count > 0 else 0.0,
            "Intra-EVR": totals[group]["intra_evr"] / count if count > 0 else 0.0,
            "Inter-EVR": totals[group]["inter_evr"] / count if count > 0 else 0.0,
            "GDG": totals[group]["gdg"] / count if count > 0 else 0.0,
            "Mean-Principal-Angle-Deg": (
                totals[group]["principal_angle_deg"] / count if count > 0 else 0.0
            ),
            "num_anchors": int(round(count)),
        }

    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch", None),
        "k_shot": args.k_shot,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_batches": num_batches,
        "aggregation": "anchor_weighted_single_gpu_posthoc",
        "groups": groups,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print("=" * 88)
    print(f"Checkpoint: {args.checkpoint}")
    for group in ("overall", "base", "novel"):
        g = groups[group]
        print(
            f"{group.upper():7s} | "
            f"TAS {g['TAS']:.6f} | "
            f"Intra-EVR {g['Intra-EVR']:.6f} | "
            f"Inter-EVR {g['Inter-EVR']:.6f} | "
            f"GDG {g['GDG']:.6f} | "
            f"MeanPrincipalAngle {g['Mean-Principal-Angle-Deg']:.6f} deg | "
            f"anchors {g['num_anchors']}"
        )
    print(f"Saved: {args.output}")
    print("=" * 88)


if __name__ == "__main__":
    main()
