#!/usr/bin/env python3
"""Visualize and quantify PointSLA Point-Encoder / CLIP-text feature separation.

The primary comparison is intentionally between:
  1) raw Point Encoder features from the frozen PTv3 backbone (64-D), and
  2) frozen CLIP text prototypes (one 512-D vector per Base class).

The semantic projector is NOT used in the primary analysis. This keeps the
visualization focused on the intrinsic gap between the point-encoder feature
space and the language feature space.

Because the two sides have different sample semantics (many point samples per
class versus one text prototype per class), the script produces:
  1) <method>_point_vs_text.{pdf,png}
     - Point Encoder: 1024 balanced point samples per eligible class by default.
     - CLIP Text Encoder: one frozen text prototype per eligible class.
  2) <method>_centroid_vs_text.{pdf,png}
     - Point Encoder class centroids versus CLIP text prototypes, giving a fair
       one-vector-per-class comparison.

Outputs:
  sampled_features.npz
  metrics.json
  metrics.csv
  class_map.csv
  <method>_point_vs_text.{pdf,png}
  <method>_centroid_vs_text.{pdf,png}
  <method>_coordinates.npz

Example:
  CUDA_VISIBLE_DEVICES=4 python tools/visualize_point_text_feature_space.py \
    --config configs/scannetpp/semseg-pt-v3m1-0-conservative-boundary-semantic-tangent-cdcbase-frozen-r24.py \
    --checkpoint exp/scannetpp/cdc_stage1_r24/model/model_best.pth \
    --output-dir exp/scannetpp/feature_space_separability \
    --method auto
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import json
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.models.utils.structure import Point
from pointcept.utils.config import Config


DEFAULT_CONFIG = (
    "configs/scannetpp/"
    "semseg-pt-v3m1-0-conservative-boundary-semantic-tangent-cdcbase-frozen-r24.py"
)
DEFAULT_CHECKPOINT = "exp/scannetpp/cdc_stage1_r24/model/model_best.pth"
DEFAULT_OUTPUT = "exp/scannetpp/feature_space_separability"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "UMAP/t-SNE comparison of raw Point Encoder features and "
            "frozen CLIP text prototypes."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--method",
        choices=("auto", "umap", "tsne", "both"),
        default="auto",
        help="auto uses UMAP when available and otherwise falls back to t-SNE.",
    )
    parser.add_argument("--max-points-per-class", type=int, default=1024)
    parser.add_argument("--min-points-per-class", type=int, default=1024)
    parser.add_argument("--max-per-scene-per-class", type=int, default=256)
    parser.add_argument("--silhouette-sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--point-alpha", type=float, default=0.58)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint_strict(model, checkpoint_path: str):
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
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


def move_to_cuda(batch):
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.cuda(non_blocking=True)
        else:
            moved[key] = value
    return moved


def collect_balanced_raw_features(
    model,
    dataset,
    num_classes: int,
    max_points_per_class: int,
    max_per_scene_per_class: int,
    seed: int,
):
    """Collect raw Point Encoder validation features with class/scene balancing."""
    raw_parts = [[] for _ in range(num_classes)]
    counts = np.zeros(num_classes, dtype=np.int64)
    scene_counts = np.zeros(num_classes, dtype=np.int64)

    rng = np.random.default_rng(seed)
    model.eval()

    with torch.no_grad():
        for scene_idx in range(len(dataset)):
            sample = dataset[scene_idx]
            batch = move_to_cuda(collate_fn([sample]))
            labels = batch["segment"].long()

            with torch.cuda.amp.autocast(enabled=False):
                point = model.backbone(Point(batch))
                raw = point.feat if isinstance(point, Point) else point
                raw = raw.detach().float()

            if int(raw.shape[0]) != int(labels.shape[0]):
                raise RuntimeError(
                    f"Feature/label mismatch at scene {scene_idx}: "
                    f"feature={tuple(raw.shape)}, labels={tuple(labels.shape)}"
                )

            valid = (labels >= 0) & (labels < num_classes)
            if not bool(valid.any().item()):
                continue

            present = torch.unique(labels[valid]).detach().cpu().tolist()
            for class_id in present:
                class_id = int(class_id)
                remaining = max_points_per_class - int(counts[class_id])
                if remaining <= 0:
                    continue

                idx = torch.nonzero(labels == class_id, as_tuple=False).flatten()
                take = min(
                    int(idx.numel()),
                    int(remaining),
                    int(max_per_scene_per_class),
                )
                if take <= 0:
                    continue

                selected_local = rng.choice(
                    int(idx.numel()), size=take, replace=False
                )
                selected_local = torch.tensor(
                    selected_local.tolist(),
                    device=idx.device,
                    dtype=torch.long,
                )
                chosen = idx[selected_local]
                raw_parts[class_id].append(raw[chosen].cpu())
                counts[class_id] += take
                scene_counts[class_id] += 1

            if np.all(counts >= max_points_per_class):
                break

    raw_by_class = []
    for class_id in range(num_classes):
        if raw_parts[class_id]:
            raw_by_class.append(torch.cat(raw_parts[class_id], dim=0))
        else:
            raw_by_class.append(torch.empty((0, int(model.feature_dim))))

    return raw_by_class, counts, scene_counts


def build_equal_sample_arrays(
    raw_by_class,
    class_names,
    min_points_per_class: int,
    max_points_per_class: int,
    seed: int,
):
    eligible = [
        class_id
        for class_id, feature in enumerate(raw_by_class)
        if int(feature.shape[0]) >= min_points_per_class
    ]
    if len(eligible) < 2:
        collected = [int(feature.shape[0]) for feature in raw_by_class]
        raise RuntimeError(
            "Fewer than two classes satisfy --min-points-per-class. "
            f"Required={min_points_per_class}, collected={collected}. "
            "Reduce the threshold only if you intentionally accept fewer "
            "than 1024 point samples per class."
        )

    per_class = min(
        int(max_points_per_class),
        min(int(raw_by_class[class_id].shape[0]) for class_id in eligible),
    )
    if per_class < min_points_per_class:
        raise RuntimeError(
            f"Balanced sample count {per_class} is below the requested "
            f"minimum {min_points_per_class}."
        )

    rng = np.random.default_rng(seed)
    raw_all, labels_all = [], []
    selected_names, original_ids = [], []

    for new_id, class_id in enumerate(eligible):
        count = int(raw_by_class[class_id].shape[0])
        selected = rng.choice(count, size=per_class, replace=False)
        selected_t = torch.tensor(selected.tolist(), dtype=torch.long)
        raw_all.append(raw_by_class[class_id][selected_t].numpy())
        labels_all.append(np.full(per_class, new_id, dtype=np.int64))
        selected_names.append(str(class_names[class_id]))
        original_ids.append(int(class_id))

    return (
        np.concatenate(raw_all, axis=0).astype(np.float32, copy=False),
        np.concatenate(labels_all, axis=0),
        selected_names,
        original_ids,
        per_class,
    )


def l2_normalize(x: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def class_centroids(x: np.ndarray, labels: np.ndarray):
    x = l2_normalize(x)
    result = []
    for class_id in np.unique(labels):
        centroid = x[labels == class_id].mean(axis=0, keepdims=True)
        result.append(l2_normalize(centroid)[0])
    return np.stack(result, axis=0).astype(np.float32)


def prototype_distance_metrics(x: np.ndarray):
    x = l2_normalize(x)
    cosine = np.clip(x @ x.T, -1.0, 1.0)
    distance = 1.0 - cosine
    values = distance[np.triu_indices(x.shape[0], k=1)]
    return {
        "mean_pairwise_cosine_distance": float(values.mean()),
        "median_pairwise_cosine_distance": float(np.median(values)),
        "min_pairwise_cosine_distance": float(values.min()),
        "max_pairwise_cosine_distance": float(values.max()),
    }


def sample_separability_metrics(
    x: np.ndarray,
    labels: np.ndarray,
    seed: int,
    silhouette_sample_size: int,
):
    from sklearn.metrics import silhouette_score

    x = l2_normalize(x)
    classes = np.unique(labels)
    centroids = []
    intra = []

    for class_id in classes:
        current = x[labels == class_id]
        centroid = l2_normalize(current.mean(axis=0, keepdims=True))[0]
        centroids.append(centroid)
        intra.extend(1.0 - np.clip(current @ centroid, -1.0, 1.0))

    centroids = np.stack(centroids, axis=0)
    centroid_cosine = np.clip(centroids @ centroids.T, -1.0, 1.0)
    centroid_distance = 1.0 - centroid_cosine
    inter = centroid_distance[np.triu_indices(len(classes), k=1)]
    intra = np.asarray(intra, dtype=np.float64)

    sample_size = min(int(silhouette_sample_size), int(x.shape[0]))
    silhouette = float(
        silhouette_score(
            x,
            labels,
            metric="cosine",
            sample_size=sample_size if sample_size < x.shape[0] else None,
            random_state=seed,
        )
    )

    rng = np.random.default_rng(seed)
    train_centroids = []
    test_x, test_y = [], []
    for new_id, class_id in enumerate(classes):
        current = x[labels == class_id]
        order = rng.permutation(len(current))
        split = max(1, len(current) // 2)
        train_idx = order[:split]
        test_idx = order[split:]
        if len(test_idx) == 0:
            test_idx = train_idx
        train_centroids.append(
            l2_normalize(current[train_idx].mean(axis=0, keepdims=True))[0]
        )
        test_x.append(current[test_idx])
        test_y.append(np.full(len(test_idx), new_id, dtype=np.int64))

    train_centroids = np.stack(train_centroids, axis=0)
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)
    pred = np.argmax(test_x @ train_centroids.T, axis=1)

    return {
        "num_samples": int(x.shape[0]),
        "num_classes": int(len(classes)),
        "silhouette_cosine": silhouette,
        "mean_intra_class_cosine_distance": float(intra.mean()),
        "median_intra_class_cosine_distance": float(np.median(intra)),
        "mean_inter_centroid_cosine_distance": float(inter.mean()),
        "median_inter_centroid_cosine_distance": float(np.median(inter)),
        "min_inter_centroid_cosine_distance": float(inter.min()),
        "inter_over_intra_ratio": float(inter.mean() / max(intra.mean(), 1e-12)),
        "heldout_nearest_centroid_accuracy": float(np.mean(pred == test_y)),
    }


def resolve_methods(requested: str):
    try:
        import umap  # noqa: F401

        has_umap = True
    except ImportError:
        has_umap = False

    if requested == "auto":
        if has_umap:
            return ["umap"]
        print("[WARN] umap-learn is not installed; falling back to t-SNE.")
        return ["tsne"]
    if requested == "umap":
        if not has_umap:
            raise ImportError(
                "UMAP requested but umap-learn is missing. Install `umap-learn` "
                "or run with --method tsne."
            )
        return ["umap"]
    if requested == "both":
        if has_umap:
            return ["umap", "tsne"]
        print("[WARN] umap-learn is not installed; running t-SNE only.")
        return ["tsne"]
    return ["tsne"]


def reduce_2d(x: np.ndarray, method: str, seed: int, prototype_mode=False):
    x = l2_normalize(x).astype(np.float32)
    n = int(x.shape[0])
    if n < 3:
        raise ValueError(f"Need at least three samples for {method}, got {n}")

    if method == "umap":
        import umap

        n_neighbors = min(15, max(2, n - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.10,
            metric="cosine",
            random_state=seed,
            transform_seed=seed,
        )
        return reducer.fit_transform(x).astype(np.float32)

    from sklearn.manifold import TSNE

    if prototype_mode:
        perplexity = min(10.0, max(2.0, (n - 1) / 3.0))
    else:
        perplexity = min(30.0, max(5.0, (n - 1) / 10.0))
    perplexity = min(perplexity, float(n - 1))

    kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        metric="cosine",
        init="random",
        learning_rate="auto",
        random_state=seed,
    )
    try:
        reducer = TSNE(max_iter=1500, **kwargs)
    except TypeError:
        reducer = TSNE(n_iter=1500, **kwargs)
    return reducer.fit_transform(x).astype(np.float32)


def colors_for_classes(num_classes: int):
    """Generate a soft pastel palette while keeping classes distinguishable."""
    if num_classes <= 0:
        return []
    colors = []
    for i in range(num_classes):
        hue = (i / max(num_classes, 1) + 0.035 * (i % 5)) % 1.0
        saturation = 0.28 + 0.08 * (i % 3) / 2.0
        value = 0.96 - 0.05 * (i % 4) / 3.0
        colors.append(colorsys.hsv_to_rgb(hue, saturation, value))
    return colors


def style_axis(ax):
    """Paper-style UMAP/t-SNE panel with an explicit rectangular frame."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    try:
        ax.set_box_aspect(1.0)
    except AttributeError:
        pass
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.15)
        spine.set_edgecolor("black")


def scatter_samples(ax, coords, labels, colors, point_size, alpha):
    for class_id in np.unique(labels):
        mask = labels == class_id
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=alpha,
            c=[colors[int(class_id)]],
            linewidths=0,
            rasterized=True,
        )
    style_axis(ax)


def scatter_prototypes(
    ax,
    coords,
    names,
    colors,
    marker="o",
    annotate=True,
    marker_size=58,
):
    for class_id, _name in enumerate(names):
        ax.scatter(
            coords[class_id, 0],
            coords[class_id, 1],
            s=marker_size,
            c=[colors[class_id]],
            marker=marker,
            edgecolors="black",
            linewidths=0.45,
            zorder=3,
        )
        if annotate:
            ax.text(
                coords[class_id, 0],
                coords[class_id, 1],
                str(class_id),
                fontsize=6.5,
                ha="left",
                va="bottom",
                zorder=4,
            )
    style_axis(ax)


def save_figure(fig, base_path: Path, dpi: int):
    fig.savefig(str(base_path) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base_path) + ".png", dpi=dpi, bbox_inches="tight")


def plot_method(
    method,
    output_dir,
    raw,
    labels,
    raw_centroids,
    text_prototypes,
    class_names,
    seed,
    point_size,
    point_alpha,
    dpi,
):
    import matplotlib.pyplot as plt

    colors = colors_for_classes(len(class_names))

    # Primary figure: the raw point-encoder sample distribution versus the
    # frozen CLIP text-prototype distribution. These embeddings are performed
    # separately because the original spaces are 64-D and 512-D, respectively.
    print(f"[{method}] Reducing Point Encoder samples: {raw.shape}")
    raw_xy = reduce_2d(raw, method, seed, prototype_mode=False)
    print(f"[{method}] Reducing CLIP text prototypes: {text_prototypes.shape}")
    text_xy = reduce_2d(
        text_prototypes, method, seed + 1, prototype_mode=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6))
    scatter_samples(
        axes[0], raw_xy, labels, colors, point_size, point_alpha
    )
    scatter_prototypes(
        axes[1],
        text_xy,
        class_names,
        colors,
        marker="o",
        annotate=True,
        marker_size=64,
    )
    axes[0].set_title("Point Encoder", fontsize=13, fontweight="bold", pad=8)
    axes[1].set_title("CLIP Text Encoder", fontsize=13, fontweight="bold", pad=8)
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, output_dir / f"{method}_point_vs_text", dpi)
    plt.close(fig)

    # Fair class-level figure: one point centroid versus one text prototype per
    # class. Each panel is still reduced separately because the feature
    # dimensionalities differ.
    print(f"[{method}] Reducing Point Encoder class centroids")
    raw_centroid_xy = reduce_2d(
        raw_centroids, method, seed + 2, prototype_mode=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6))
    scatter_prototypes(
        axes[0],
        raw_centroid_xy,
        class_names,
        colors,
        marker="o",
        annotate=True,
        marker_size=60,
    )
    scatter_prototypes(
        axes[1],
        text_xy,
        class_names,
        colors,
        marker="o",
        annotate=True,
        marker_size=64,
    )
    axes[0].set_title(
        "Point Encoder (Class Centroids)",
        fontsize=13,
        fontweight="bold",
        pad=8,
    )
    axes[1].set_title("CLIP Text Encoder", fontsize=13, fontweight="bold", pad=8)
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, output_dir / f"{method}_centroid_vs_text", dpi)
    plt.close(fig)

    np.savez(
        output_dir / f"{method}_coordinates.npz",
        point_encoder_xy=raw_xy,
        labels=labels,
        point_centroid_xy=raw_centroid_xy,
        clip_text_xy=text_xy,
    )


def flatten_metrics(metrics):
    rows = []
    for group, values in metrics.items():
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            rows.append((group, name, value))
    return rows


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to extract Point Encoder features")
    if args.max_points_per_class <= 1:
        raise ValueError("--max-points-per-class must be > 1")
    if args.min_points_per_class <= 1:
        raise ValueError("--min-points-per-class must be > 1")
    if args.max_per_scene_per_class <= 0:
        raise ValueError("--max-per-scene-per-class must be positive")
    if args.min_points_per_class > args.max_points_per_class:
        raise ValueError(
            "--min-points-per-class cannot exceed --max-points-per-class"
        )

    set_seed(args.seed)
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(str(config_path))
    model = build_model(cfg.model).cuda()
    checkpoint = load_checkpoint_strict(model, str(checkpoint_path))
    model.eval()

    text_all = model.text_embeddings.detach().float().cpu()
    if text_all.ndim != 2 or int(text_all.shape[0]) != int(model.num_classes):
        raise RuntimeError(
            "Unexpected text embedding shape: "
            f"{tuple(text_all.shape)}, num_classes={model.num_classes}"
        )
    text_norm = text_all.norm(dim=-1)
    if bool((text_norm < 1e-8).any().item()):
        bad = torch.nonzero(text_norm < 1e-8, as_tuple=False).flatten().tolist()
        raise RuntimeError(
            "Checkpoint contains zero/uninitialized text embeddings for classes: "
            f"{bad}. Use the trained Stage-1 SSC checkpoint."
        )

    class_names = list(getattr(cfg.data, "names", []))
    if len(class_names) != int(model.num_classes):
        class_names = [f"class_{i}" for i in range(int(model.num_classes))]

    print("============================================================")
    print("PointSLA Point Encoder / CLIP Text Encoder feature comparison")
    print(f"Config       : {config_path}")
    print(f"Checkpoint   : {checkpoint_path}")
    print(f"Output       : {output_dir}")
    print(f"Base classes : {model.num_classes}")
    print(f"Point dim    : {model.feature_dim}")
    print(f"Text dim     : {model.text_dim}")
    print(f"Target/class : {args.max_points_per_class}")
    print("============================================================")

    val_dataset = build_dataset(cfg.data.val)
    raw_by_class, collected, scene_counts = collect_balanced_raw_features(
        model=model,
        dataset=val_dataset,
        num_classes=int(model.num_classes),
        max_points_per_class=args.max_points_per_class,
        max_per_scene_per_class=args.max_per_scene_per_class,
        seed=args.seed,
    )

    (
        raw,
        labels,
        selected_names,
        original_ids,
        balanced_per_class,
    ) = build_equal_sample_arrays(
        raw_by_class=raw_by_class,
        class_names=class_names,
        min_points_per_class=args.min_points_per_class,
        max_points_per_class=args.max_points_per_class,
        seed=args.seed,
    )

    raw = l2_normalize(raw).astype(np.float32)

    selected_index = torch.tensor(original_ids, dtype=torch.long)
    text_selected = F.normalize(
        text_all[selected_index],
        dim=-1,
        eps=1e-6,
    ).numpy().astype(np.float32)

    raw_centroids = class_centroids(raw, labels)

    raw_metrics = sample_separability_metrics(
        raw, labels, args.seed, args.silhouette_sample_size
    )
    raw_centroid_metrics = prototype_distance_metrics(raw_centroids)
    text_metrics = prototype_distance_metrics(text_selected)

    metrics = {
        "point_encoder_features_64d": raw_metrics,
        "point_encoder_class_centroids_64d": raw_centroid_metrics,
        "clip_text_prototypes_512d": text_metrics,
        "comparison": {
            "text_minus_point_centroid_mean_pairwise_cosine_distance": float(
                text_metrics["mean_pairwise_cosine_distance"]
                - raw_centroid_metrics["mean_pairwise_cosine_distance"]
            ),
            "text_minus_point_centroid_median_pairwise_cosine_distance": float(
                text_metrics["median_pairwise_cosine_distance"]
                - raw_centroid_metrics["median_pairwise_cosine_distance"]
            ),
            "text_minus_point_centroid_min_pairwise_cosine_distance": float(
                text_metrics["min_pairwise_cosine_distance"]
                - raw_centroid_metrics["min_pairwise_cosine_distance"]
            ),
        },
        "sampling": {
            "balanced_points_per_class": int(balanced_per_class),
            "total_point_samples": int(len(labels)),
            "num_selected_classes": int(len(selected_names)),
            "max_points_per_class": int(args.max_points_per_class),
            "min_points_per_class": int(args.min_points_per_class),
            "max_per_scene_per_class": int(args.max_per_scene_per_class),
            "seed": int(args.seed),
        },
        "provenance": {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch", None),
            "primary_comparison": (
                "raw Point Encoder features versus frozen CLIP text prototypes; "
                "semantic projector excluded"
            ),
        },
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, sort_keys=True)

    with open(output_dir / "metrics.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "metric", "value"])
        for group, name, value in flatten_metrics(metrics):
            writer.writerow([group, name, value])

    with open(output_dir / "class_map.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "plot_class_id",
                "original_base_label",
                "class_name",
                "collected_points",
                "contributing_scenes",
            ]
        )
        for plot_id, original_id in enumerate(original_ids):
            writer.writerow(
                [
                    plot_id,
                    original_id,
                    selected_names[plot_id],
                    int(collected[original_id]),
                    int(scene_counts[original_id]),
                ]
            )

    np.savez(
        output_dir / "sampled_features.npz",
        point_encoder_features=raw,
        labels=labels,
        point_encoder_centroids=raw_centroids,
        clip_text_prototypes=text_selected,
        original_base_labels=np.asarray(original_ids, dtype=np.int64),
        class_names=np.asarray(selected_names, dtype=object),
    )

    methods = resolve_methods(args.method)
    for method in methods:
        plot_method(
            method=method,
            output_dir=output_dir,
            raw=raw,
            labels=labels,
            raw_centroids=raw_centroids,
            text_prototypes=text_selected,
            class_names=selected_names,
            seed=args.seed,
            point_size=args.point_size,
            point_alpha=args.point_alpha,
            dpi=args.dpi,
        )

    print("\nQuantitative comparison")
    print("-----------------------")
    print(
        "Point Encoder samples/class : "
        f"{balanced_per_class}"
    )
    print(
        "Total Point Encoder samples : "
        f"{len(labels)}"
    )
    print(
        "Selected Base classes       : "
        f"{len(selected_names)}"
    )
    print(
        "Point silhouette            : "
        f"{raw_metrics['silhouette_cosine']:.6f}"
    )
    print(
        "Point inter/intra ratio      : "
        f"{raw_metrics['inter_over_intra_ratio']:.6f}"
    )
    print(
        "Point centroid accuracy      : "
        f"{raw_metrics['heldout_nearest_centroid_accuracy']:.6f}"
    )
    print(
        "Point centroid mean distance : "
        f"{raw_centroid_metrics['mean_pairwise_cosine_distance']:.6f}"
    )
    print(
        "CLIP text mean distance      : "
        f"{text_metrics['mean_pairwise_cosine_distance']:.6f}"
    )
    print(
        "CLIP-text minus point-centroid mean distance: "
        f"{metrics['comparison']['text_minus_point_centroid_mean_pairwise_cosine_distance']:.6f}"
    )
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
