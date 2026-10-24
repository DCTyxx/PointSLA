#!/usr/bin/env python3
"""Build scene-level Baseline-vs-Our comparisons and paper-ready point files.

The script consumes the scene outputs produced by GFSSemSegTester with cfg.vis:

  <vis_root>/<scene>/query_label.npy
  <vis_root>/<scene>/query_pred.npy

Raw XYZ/RGB are reloaded from the dataset directory instead of query.npy.
This is deliberate: after inverse mapping, GT/pred are full-resolution while the
legacy query.npy coordinate array can still be GridSample-resolution.

For every selected scene the showcase directory contains exactly four text files:

  input.txt     x y z label r g b   (label=-1, RGB=input color)
  GT.txt        x y z label r g b
  baseline.txt  x y z label r g b
  Our.txt       x y z label r g b

ScanNet and ScanNet200 semantic RGB values come from the benchmark color maps
already shipped in this repository. ScanNet++ does not ship an official semantic
palette here, so it uses a deterministic class-name-stable fallback palette and
records that fact in evaluation_summary.json.
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _dataset_meta(dataset: str):
    if dataset == "scannet":
        from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
            CLASS20_LABELS_BASE,
            CLASS20_LABELS_BASE_NOVEL,
            CLASS_LABELS_20,
            SCANNET_COLOR_MAP_20,
            VALID_CLASS_IDS_20,
        )

        names = list(CLASS20_LABELS_BASE_NOVEL)
        official = []
        for name in names:
            idx = list(CLASS_LABELS_20).index(name)
            official_id = int(VALID_CLASS_IDS_20[idx])
            official.append(tuple(int(v) for v in SCANNET_COLOR_MAP_20[official_id]))
        return {
            "names": names,
            "num_base": len(CLASS20_LABELS_BASE),
            "palette": official,
            "palette_source": "ScanNet benchmark SCANNET_COLOR_MAP_20",
        }

    if dataset == "scannet200":
        from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
            CLASS_LABELS_200,
            CLASS_LABELS_BASE,
            CLASS_LABELS_BASE_NOVEL,
            SCANNET_COLOR_MAP_200,
            VALID_CLASS_IDS_200,
        )

        names = list(CLASS_LABELS_BASE_NOVEL)
        official = []
        class_labels_200 = list(CLASS_LABELS_200)
        for name in names:
            idx = class_labels_200.index(name)
            official_id = int(VALID_CLASS_IDS_200[idx])
            official.append(tuple(int(v) for v in SCANNET_COLOR_MAP_200[official_id]))
        return {
            "names": names,
            "num_base": len(CLASS_LABELS_BASE),
            "palette": official,
            "palette_source": "ScanNet200 benchmark SCANNET_COLOR_MAP_200",
        }

    if dataset == "scannetpp":
        from pointcept.datasets.preprocessing.scannetpp.metadata.semantic_benchmark.top100 import (
            CLASS_LABELS_100,
            CLASS_LABELS_BASE,
            CLASS_LABELS_BASE_NOVEL,
        )

        names = list(CLASS_LABELS_BASE_NOVEL)
        full_names = list(CLASS_LABELS_100)
        palette = []
        # Repository-local ScanNet++ metadata contains class definitions but no
        # official semantic color table. Use a deterministic hue generated from
        # the official top-100 class index so colors are stable across runs.
        for name in names:
            class_idx = full_names.index(name)
            hue = (class_idx * 0.6180339887498949) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.68, 0.95)
            palette.append(tuple(int(round(v * 255.0)) for v in rgb))
        return {
            "names": names,
            "num_base": len(CLASS_LABELS_BASE),
            "palette": palette,
            "palette_source": (
                "deterministic fallback based on ScanNet++ top-100 class index; "
                "repository has no official semantic RGB palette"
            ),
        }

    raise ValueError(f"Unsupported dataset: {dataset}")


def _safe_float(x):
    if x is None or not math.isfinite(float(x)):
        return None
    return float(x)


def _inter_union(pred: np.ndarray, gt: np.ndarray, num_classes: int):
    pred = np.asarray(pred).reshape(-1).astype(np.int64, copy=False)
    gt = np.asarray(gt).reshape(-1).astype(np.int64, copy=False)
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction/GT size mismatch: {pred.shape} vs {gt.shape}")
    valid = gt >= 0
    pred = pred[valid]
    gt = gt[valid]
    if pred.size == 0:
        z = np.zeros(num_classes, dtype=np.int64)
        return z.copy(), z.copy(), z.copy()
    if pred.min(initial=0) < 0 or pred.max(initial=-1) >= num_classes:
        raise ValueError(
            f"Prediction outside [0,{num_classes - 1}]: min={pred.min()}, max={pred.max()}"
        )
    intersection = pred[pred == gt]
    area_inter = np.bincount(intersection, minlength=num_classes)[:num_classes]
    area_pred = np.bincount(pred, minlength=num_classes)[:num_classes]
    area_gt = np.bincount(gt, minlength=num_classes)[:num_classes]
    union = area_pred + area_gt - area_inter
    return area_inter.astype(np.int64), union.astype(np.int64), area_gt.astype(np.int64)


def _metrics_from_counts(inter, union, target, num_base: int, active_only: bool):
    inter = inter.astype(np.float64)
    union = union.astype(np.float64)
    target = target.astype(np.float64)
    iou = inter / (union + 1e-10)
    acc = inter / (target + 1e-10)

    if active_only:
        all_mask = union > 0
        base_mask = union[:num_base] > 0
        novel_mask = union[num_base:] > 0
        miou = float(iou[all_mask].mean()) if all_mask.any() else float("nan")
        base = float(iou[:num_base][base_mask].mean()) if base_mask.any() else float("nan")
        novel = (
            float(iou[num_base:][novel_mask].mean()) if novel_mask.any() else float("nan")
        )
    else:
        miou = float(iou.mean())
        base = float(iou[:num_base].mean())
        novel = float(iou[num_base:].mean())

    hm = (
        2.0 * base * novel / (base + novel + 1e-10)
        if math.isfinite(base) and math.isfinite(novel)
        else float("nan")
    )
    all_acc = float(inter.sum() / (target.sum() + 1e-10))
    macc = float(acc.mean())
    return {
        "mIoU": miou,
        "base_mIoU": base,
        "novel_mIoU": novel,
        "hm_mIoU": hm,
        "mAcc": macc,
        "allAcc": all_acc,
        "iou_class": iou,
    }


def _scene_dirs(vis_root: Path):
    if not vis_root.is_dir():
        raise FileNotFoundError(f"Visualization root not found: {vis_root}")
    out = {}
    for p in vis_root.iterdir():
        if not p.is_dir():
            continue
        if (p / "query_label.npy").is_file() and (p / "query_pred.npy").is_file():
            out[p.name] = p
    return out


def _find_raw_scene(data_root: Path, scene: str):
    candidates = [
        data_root / "val" / scene,
        data_root / "validation" / scene,
        data_root / scene,
    ]
    for path in candidates:
        if (path / "coord.npy").is_file():
            return path
    matches = list(data_root.glob(f"**/{scene}"))
    matches = [p for p in matches if p.is_dir() and (p / "coord.npy").is_file()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Cannot find raw scene {scene!r} under {data_root}")
    raise RuntimeError(f"Ambiguous raw scene {scene!r}: {matches[:5]}")


def _input_rgb(color: np.ndarray | None, n: int):
    if color is None:
        return np.full((n, 3), 180, dtype=np.uint8)
    color = np.asarray(color)
    if color.ndim != 2 or color.shape[0] != n or color.shape[1] < 3:
        raise ValueError(f"Unexpected raw color shape: {color.shape}, expected ({n},3+)")
    c = color[:, :3].astype(np.float32, copy=False)
    # Support both raw [0,255] and normalized [-1,1] files.
    if c.size and c.min() >= -1.1 and c.max() <= 1.1:
        c = (c + 1.0) * 127.5
    return np.clip(np.rint(c), 0, 255).astype(np.uint8)


def _semantic_rgb(labels: np.ndarray, palette):
    labels = np.asarray(labels).reshape(-1).astype(np.int64, copy=False)
    rgb = np.zeros((labels.size, 3), dtype=np.uint8)
    for idx, color in enumerate(palette):
        rgb[labels == idx] = np.asarray(color, dtype=np.uint8)
    # label=-1 / unknown remains black.
    return rgb


def _write_xyz_label_rgb(path: Path, xyz, labels, rgb):
    xyz = np.asarray(xyz, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"Bad xyz shape for {path}: {xyz.shape}")
    xyz = xyz[:, :3]
    if xyz.shape[0] != labels.size or rgb.shape != (labels.size, 3):
        raise ValueError(
            f"Point table mismatch for {path}: xyz={xyz.shape}, label={labels.shape}, rgb={rgb.shape}"
        )
    table = np.column_stack([xyz, labels, rgb])
    np.savetxt(
        path,
        table,
        fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d", "%d"],
        header="x y z label r g b",
        comments="# ",
    )


def _json_dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["scannet", "scannet200", "scannetpp"])
    parser.add_argument("--baseline-vis", required=True)
    parser.add_argument("--ours-vis", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--comparison-dir", required=True)
    parser.add_argument("--showcase-dir", required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--regis", default="unknown")
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--baseline-ckpt", default="")
    parser.add_argument("--ours-ckpt", default="")
    args = parser.parse_args()

    meta = _dataset_meta(args.dataset)
    names = meta["names"]
    num_classes = len(names)
    num_base = int(meta["num_base"])
    palette = meta["palette"]

    baseline_vis = Path(args.baseline_vis)
    ours_vis = Path(args.ours_vis)
    data_root = Path(args.data_root)
    comparison_dir = Path(args.comparison_dir)
    showcase_dir = Path(args.showcase_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    baseline_scenes = _scene_dirs(baseline_vis)
    ours_scenes = _scene_dirs(ours_vis)
    common = sorted(set(baseline_scenes) & set(ours_scenes))
    if len(common) < args.topk:
        raise RuntimeError(
            f"Need at least topk={args.topk} common evaluated scenes, found {len(common)}"
        )

    total_b_inter = np.zeros(num_classes, dtype=np.int64)
    total_b_union = np.zeros(num_classes, dtype=np.int64)
    total_b_target = np.zeros(num_classes, dtype=np.int64)
    total_o_inter = np.zeros(num_classes, dtype=np.int64)
    total_o_union = np.zeros(num_classes, dtype=np.int64)
    total_o_target = np.zeros(num_classes, dtype=np.int64)

    scene_rows = []
    scene_cache = {}
    for scene in common:
        b_dir = baseline_scenes[scene]
        o_dir = ours_scenes[scene]
        gt_b = np.load(b_dir / "query_label.npy").reshape(-1).astype(np.int64)
        gt_o = np.load(o_dir / "query_label.npy").reshape(-1).astype(np.int64)
        pred_b = np.load(b_dir / "query_pred.npy").reshape(-1).astype(np.int64)
        pred_o = np.load(o_dir / "query_pred.npy").reshape(-1).astype(np.int64)
        if gt_b.shape != gt_o.shape or not np.array_equal(gt_b, gt_o):
            raise RuntimeError(f"Baseline/Our GT mismatch in scene {scene}")
        if pred_b.shape != gt_b.shape or pred_o.shape != gt_b.shape:
            raise RuntimeError(
                f"Prediction size mismatch in {scene}: GT={gt_b.shape}, baseline={pred_b.shape}, ours={pred_o.shape}"
            )

        bi, bu, bt = _inter_union(pred_b, gt_b, num_classes)
        oi, ou, ot = _inter_union(pred_o, gt_b, num_classes)
        total_b_inter += bi
        total_b_union += bu
        total_b_target += bt
        total_o_inter += oi
        total_o_union += ou
        total_o_target += ot

        bm = _metrics_from_counts(bi, bu, bt, num_base, active_only=True)
        om = _metrics_from_counts(oi, ou, ot, num_base, active_only=True)
        delta = om["mIoU"] - bm["mIoU"]
        delta_novel = (
            om["novel_mIoU"] - bm["novel_mIoU"]
            if math.isfinite(om["novel_mIoU"]) and math.isfinite(bm["novel_mIoU"])
            else float("nan")
        )
        row = {
            "scene": scene,
            "num_points": int(gt_b.size),
            "baseline_mIoU": bm["mIoU"],
            "ours_mIoU": om["mIoU"],
            "delta_mIoU": delta,
            "baseline_novel_mIoU": bm["novel_mIoU"],
            "ours_novel_mIoU": om["novel_mIoU"],
            "delta_novel_mIoU": delta_novel,
            "baseline_allAcc": bm["allAcc"],
            "ours_allAcc": om["allAcc"],
            "delta_allAcc": om["allAcc"] - bm["allAcc"],
        }
        scene_rows.append(row)
        scene_cache[scene] = (gt_b, pred_b, pred_o)

    # Primary selection criterion: Our scene mIoU - Baseline scene mIoU.
    # Novel delta is only a deterministic tie-breaker when available.
    scene_rows.sort(
        key=lambda r: (
            float(r["delta_mIoU"]),
            float(r["delta_novel_mIoU"])
            if math.isfinite(float(r["delta_novel_mIoU"]))
            else -1e9,
        ),
        reverse=True,
    )
    selected = scene_rows[: args.topk]

    baseline_global = _metrics_from_counts(
        total_b_inter, total_b_union, total_b_target, num_base, active_only=False
    )
    ours_global = _metrics_from_counts(
        total_o_inter, total_o_union, total_o_target, num_base, active_only=False
    )

    summary = {
        "dataset": args.dataset,
        "k_shot": args.k_shot,
        "regis": args.regis,
        "num_classes": num_classes,
        "num_base_classes": num_base,
        "num_novel_classes": num_classes - num_base,
        "num_common_scenes": len(common),
        "selection": {
            "criterion": "descending (ours scene mIoU - baseline scene mIoU)",
            "topk": args.topk,
            "num_selected_with_positive_gain": int(sum(r["delta_mIoU"] > 0 for r in selected)),
        },
        "palette_source": meta["palette_source"],
        "baseline_checkpoint": args.baseline_ckpt,
        "ours_checkpoint": args.ours_ckpt,
        "baseline": {
            key: _safe_float(value)
            for key, value in baseline_global.items()
            if key != "iou_class"
        },
        "ours": {
            key: _safe_float(value)
            for key, value in ours_global.items()
            if key != "iou_class"
        },
        "delta": {
            "mIoU": ours_global["mIoU"] - baseline_global["mIoU"],
            "base_mIoU": ours_global["base_mIoU"] - baseline_global["base_mIoU"],
            "novel_mIoU": ours_global["novel_mIoU"] - baseline_global["novel_mIoU"],
            "hm_mIoU": ours_global["hm_mIoU"] - baseline_global["hm_mIoU"],
        },
        "class_names": names,
        "baseline_class_iou": baseline_global["iou_class"].astype(float).tolist(),
        "ours_class_iou": ours_global["iou_class"].astype(float).tolist(),
    }
    _json_dump(comparison_dir / "evaluation_summary.json", summary)

    fields = [
        "rank",
        "selected",
        "scene",
        "num_points",
        "baseline_mIoU",
        "ours_mIoU",
        "delta_mIoU",
        "baseline_novel_mIoU",
        "ours_novel_mIoU",
        "delta_novel_mIoU",
        "baseline_allAcc",
        "ours_allAcc",
        "delta_allAcc",
    ]
    selected_names = {r["scene"] for r in selected}
    rank_map = {r["scene"]: i + 1 for i, r in enumerate(selected)}
    with (comparison_dir / "scene_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in scene_rows:
            out = dict(row)
            out["rank"] = rank_map.get(row["scene"], "")
            out["selected"] = int(row["scene"] in selected_names)
            writer.writerow(out)
    _json_dump(comparison_dir / "scene_metrics.json", scene_rows)

    # Rebuild showcase atomically at dataset level so stale scenes cannot survive.
    if showcase_dir.exists():
        shutil.rmtree(showcase_dir)
    showcase_dir.mkdir(parents=True, exist_ok=True)

    selection_rows = []
    for rank, row in enumerate(selected, start=1):
        scene = row["scene"]
        gt, pred_b, pred_o = scene_cache[scene]
        raw_scene = _find_raw_scene(data_root, scene)
        xyz = np.load(raw_scene / "coord.npy")
        color_path = raw_scene / "color.npy"
        raw_color = np.load(color_path) if color_path.is_file() else None
        if xyz.shape[0] != gt.size:
            raise RuntimeError(
                f"Raw coord/GT point count mismatch for {scene}: coord={xyz.shape[0]}, GT={gt.size}. "
                f"raw_scene={raw_scene}"
            )

        scene_out = showcase_dir / f"{rank:02d}_{scene}"
        scene_out.mkdir(parents=True, exist_ok=False)
        input_label = np.full(gt.shape, -1, dtype=np.int64)
        _write_xyz_label_rgb(
            scene_out / "input.txt", xyz, input_label, _input_rgb(raw_color, gt.size)
        )
        _write_xyz_label_rgb(
            scene_out / "GT.txt", xyz, gt, _semantic_rgb(gt, palette)
        )
        _write_xyz_label_rgb(
            scene_out / "baseline.txt", xyz, pred_b, _semantic_rgb(pred_b, palette)
        )
        _write_xyz_label_rgb(
            scene_out / "Our.txt", xyz, pred_o, _semantic_rgb(pred_o, palette)
        )
        selection_rows.append({"rank": rank, **row})

    with (comparison_dir / "showcase_selection.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank"] + [k for k in scene_rows[0].keys()])
        writer.writeheader()
        writer.writerows(selection_rows)
    _json_dump(comparison_dir / "showcase_selection.json", selection_rows)

    print(f"[DONE] {args.dataset}: compared {len(common)} scenes")
    print(
        "[METRIC] Baseline mIoU/Base/Novel/HM = "
        f"{baseline_global['mIoU']:.4f}/{baseline_global['base_mIoU']:.4f}/"
        f"{baseline_global['novel_mIoU']:.4f}/{baseline_global['hm_mIoU']:.4f}"
    )
    print(
        "[METRIC] Our      mIoU/Base/Novel/HM = "
        f"{ours_global['mIoU']:.4f}/{ours_global['base_mIoU']:.4f}/"
        f"{ours_global['novel_mIoU']:.4f}/{ours_global['hm_mIoU']:.4f}"
    )
    print(f"[SHOWCASE] {showcase_dir} ({args.topk} scenes)")


if __name__ == "__main__":
    main()
