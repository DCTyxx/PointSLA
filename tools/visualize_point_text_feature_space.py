#!/usr/bin/env python3
"""Visualize and quantify PointSLA point/text feature separability.

The script loads a trained Stage-1 SSC checkpoint and compares:
  1) raw PTv3 point features (64-D),
  2) the SAME point samples after the learned semantic projector (512-D),
  3) frozen text prototypes (one 512-D vector per Base class).

For a fair sample-level comparison, (1) and (2) use exactly the same points and
labels. Text prototypes have no within-class variance, so they are compared with
point-class centroids at class level instead of using a sample-level silhouette.

Outputs:
  sampled_features.npz
  metrics.json
  metrics.csv
  class_map.csv
  <method>_point_vs_semantic.{pdf,png}
  <method>_centroid_vs_text.{pdf,png}
  <method>_semantic_text_alignment.{pdf,png}
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
import csv
import json
import os
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
        description="UMAP/t-SNE comparison of PTv3 and text-semantic feature spaces."
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
    parser.add_argument("--max-points-per-class", type=int, default=128)
    parser.add_argument("--min-points-per-class", type=int, default=32)
    parser.add_argument("--max-per-scene-per-class", type=int, default=32)
    parser.add_argument("--silhouette-sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--point-size", type=float, default=7.0)
    parser.add_argument("--point-alpha", type=float, default=0.72)
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


def collect_balanced_features(
    model,
    dataset,
    num_classes: int,
    max_points_per_class: int,
    max_per_scene_per_class: int,
    seed: int,
):
    """Collect matching raw/semantic validation features with balanced sampling."""
    raw_parts = [[] for _ in range(num_classes)]
    semantic_parts = [[] for _ in range(num_classes)]
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
                semantic = model.semantic_projector(
                    raw, detach_parameters=True
                ).detach().float()

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

                chosen_cpu = rng.choice(int(idx.numel()), size=take, replace=False)
                chosen = idx[
                    torch.as_tensor(chosen_cpu, device=idx.device, dtype=torch.long)
                ]
                raw_parts[class_id].append(raw[chosen].cpu())
                semantic_parts[class_id].append(semantic[chosen].cpu())
                counts[class_id] += take
                scene_counts[class_id] += 1

            if np.all(counts >= max_points_per_class):
                break

    raw_by_class = []
    semantic_by_class = []
    for class_id in range(num_classes):
        if raw_parts[class_id]:
            raw_by_class.append(torch.cat(raw_parts[class_id], dim=0))
            semantic_by_class.append(torch.cat(semantic_parts[class_id], dim=0))
        else:
            raw_by_class.append(torch.empty((0, int(model.feature_dim))))
            semantic_by_class.append(torch.empty((0, int(model.text_dim))))

    return raw_by_class, semantic_by_class, counts, scene_counts


def build_equal_sample_arrays(
    raw_by_class,
    semantic_by_class,
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
        raise RuntimeError(
            "Fewer than two classes satisfy --min-points-per-class. "
            "Reduce the threshold or inspect the validation labels."
        )

    per_class = min(
        int(max_points_per_class),
        min(int(raw_by_class[class_id].shape[0]) for class_id in eligible),
    )
    if per_class < 2:
        raise RuntimeError("Balanced sample count per class is smaller than 2")

    rng = np.random.default_rng(seed)
    raw_all, semantic_all, labels_all = [], [], []
    selected_names, original_ids = [], []

    for new_id, class_id in enumerate(eligible):
        count = int(raw_by_class[class_id].shape[0])
        selected = rng.choice(count, size=per_class, replace=False)
        raw_all.append(raw_by_class[class_id][selected].numpy())
        semantic_all.append(semantic_by_class[class_id][selected].numpy())
        labels_all.append(np.full(per_class, new_id, dtype=np.int64))
        selected_names.append(str(class_names[class_id]))
        original_ids.append(int(class_id))

    return (
        np.concatenate(raw_all, axis=0).astype(np.float32, copy=False),
        np.concatenate(semantic_all, axis=0).astype(np.float32, copy=False),
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

    # Held-out nearest-centroid classification. This is more interpretable than
    # only judging 2-D UMAP/t-SNE geometry.
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


def semantic_text_alignment_metrics(semantic_centroids, text_prototypes):
    semantic = l2_normalize(semantic_centroids)
    text = l2_normalize(text_prototypes)
    similarity = semantic @ text.T
    diagonal = np.diag(similarity)
    pred = np.argmax(similarity, axis=1)
    target = np.arange(similarity.shape[0])

    wrong = similarity.copy()
    wrong[target, target] = -np.inf
    margin = diagonal - np.max(wrong, axis=1)
    return {
        "mean_matched_cosine_similarity": float(diagonal.mean()),
        "median_matched_cosine_similarity": float(np.median(diagonal)),
        "mean_text_retrieval_margin": float(margin.mean()),
        "min_text_retrieval_margin": float(margin.min()),
        "text_prototype_top1_accuracy": float(np.mean(pred == target)),
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
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("turbo")
    if num_classes == 1:
        return [cmap(0.5)]
    return [cmap(i / (num_classes - 1)) for i in range(num_classes)]


def style_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


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


def scatter_prototypes(ax, coords, names, colors, marker="o", annotate=True):
    for class_id, name in enumerate(names):
        ax.scatter(
            coords[class_id, 0],
            coords[class_id, 1],
            s=42,
            c=[colors[class_id]],
            marker=marker,
            edgecolors="black",
            linewidths=0.35,
        )
        if annotate:
            ax.text(
                coords[class_id, 0],
                coords[class_id, 1],
                str(class_id),
                fontsize=6,
                ha="left",
                va="bottom",
            )
    style_axis(ax)


def save_figure(fig, base_path: Path, dpi: int):
    fig.savefig(str(base_path) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base_path) + ".png", dpi=dpi, bbox_inches="tight")


def plot_method(
    method,
    output_dir,
    raw,
    semantic,
    labels,
    raw_centroids,
    semantic_centroids,
    text_prototypes,
    class_names,
    seed,
    point_size,
    point_alpha,
    dpi,
):
    import matplotlib.pyplot as plt

    colors = colors_for_classes(len(class_names))

    print(f"[{method}] Reducing raw point samples: {raw.shape}")
    raw_xy = reduce_2d(raw, method, seed, prototype_mode=False)
    print(f"[{method}] Reducing semantic point samples: {semantic.shape}")
    semantic_xy = reduce_2d(semantic, method, seed, prototype_mode=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    scatter_samples(
        axes[0], raw_xy, labels, colors, point_size, point_alpha
    )
    scatter_samples(
        axes[1], semantic_xy, labels, colors, point_size, point_alpha
    )
    axes[0].set_title("PTv3 Point Features (64-D)", fontsize=12, fontweight="bold")
    axes[1].set_title(
        "Projected Semantic Features (512-D)", fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    save_figure(fig, output_dir / f"{method}_point_vs_semantic", dpi)
    plt.close(fig)

    print(f"[{method}] Reducing class centroids and text prototypes")
    raw_centroid_xy = reduce_2d(
        raw_centroids, method, seed + 1, prototype_mode=True
    )
    text_xy = reduce_2d(
        text_prototypes, method, seed + 2, prototype_mode=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    scatter_prototypes(
        axes[0], raw_centroid_xy, class_names, colors, marker="o", annotate=True
    )
    scatter_prototypes(
        axes[1], text_xy, class_names, colors, marker="*", annotate=True
    )
    axes[0].set_title("Point-Class Centroids (64-D)", fontsize=12, fontweight="bold")
    axes[1].set_title("Frozen Text Prototypes (512-D)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir / f"{method}_centroid_vs_text", dpi)
    plt.close(fig)

    # Semantic centroids and text prototypes share the same 512-D CLIP space,
    # so they can be embedded jointly to visualize semantic alignment directly.
    joint = np.concatenate([semantic_centroids, text_prototypes], axis=0)
    joint_xy = reduce_2d(joint, method, seed + 3, prototype_mode=True)
    semantic_centroid_xy = joint_xy[: len(class_names)]
    text_joint_xy = joint_xy[len(class_names) :]

    fig, ax = plt.subplots(1, 1, figsize=(5.4, 5.0))
    for class_id, name in enumerate(class_names):
        c = colors[class_id]
        p = semantic_centroid_xy[class_id]
        t = text_joint_xy[class_id]
        ax.plot(
            [p[0], t[0]],
            [p[1], t[1]],
            linewidth=0.65,
            alpha=0.45,
            color=c,
        )
        ax.scatter(
            p[0], p[1], s=34, c=[c], marker="o", edgecolors="black", linewidths=0.3
        )
        ax.scatter(
            t[0], t[1], s=58, c=[c], marker="*", edgecolors="black", linewidths=0.3
        )
        ax.text(t[0], t[1], str(class_id), fontsize=6, ha="left", va="bottom")
    style_axis(ax)
    ax.set_title(
        "Semantic Point Centroids vs. Text Prototypes",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.01,
        "circle: projected point centroid   star: text prototype",
        transform=ax.transAxes,
        fontsize=7,
        ha="left",
        va="bottom",
    )
    fig.tight_layout()
    save_figure(fig, output_dir / f"{method}_semantic_text_alignment", dpi)
    plt.close(fig)

    np.savez(
        output_dir / f"{method}_coordinates.npz",
        raw_point_xy=raw_xy,
        semantic_point_xy=semantic_xy,
        labels=labels,
        raw_centroid_xy=raw_centroid_xy,
        text_prototype_xy=text_xy,
        joint_semantic_centroid_xy=semantic_centroid_xy,
        joint_text_prototype_xy=text_joint_xy,
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
        raise RuntimeError("A CUDA GPU is required to extract PTv3 features")
    if args.max_points_per_class <= 1:
        raise ValueError("--max-points-per-class must be > 1")
    if args.min_points_per_class <= 1:
        raise ValueError("--min-points-per-class must be > 1")
    if args.max_per_scene_per_class <= 0:
        raise ValueError("--max-per-scene-per-class must be positive")

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
    print("PointSLA feature-space separability")
    print(f"Config      : {config_path}")
    print(f"Checkpoint  : {checkpoint_path}")
    print(f"Output      : {output_dir}")
    print(f"Base classes: {model.num_classes}")
    print(f"Point dim   : {model.feature_dim}")
    print(f"Text dim    : {model.text_dim}")
    print("============================================================")

    val_dataset = build_dataset(cfg.data.val)
    raw_by_class, semantic_by_class, collected, scene_counts = collect_balanced_features(
        model=model,
        dataset=val_dataset,
        num_classes=int(model.num_classes),
        max_points_per_class=args.max_points_per_class,
        max_per_scene_per_class=args.max_per_scene_per_class,
        seed=args.seed,
    )

    (
        raw,
        semantic,
        labels,
        selected_names,
        original_ids,
        balanced_per_class,
    ) = build_equal_sample_arrays(
        raw_by_class=raw_by_class,
        semantic_by_class=semantic_by_class,
        class_names=class_names,
        min_points_per_class=args.min_points_per_class,
        max_points_per_class=args.max_points_per_class,
        seed=args.seed,
    )

    raw = l2_normalize(raw).astype(np.float32)
    semantic = l2_normalize(semantic).astype(np.float32)
    text_selected = F.normalize(
        text_all[torch.as_tensor(original_ids, dtype=torch.long)],
        dim=-1,
        eps=1e-6,
    ).numpy().astype(np.float32)

    raw_centroids = class_centroids(raw, labels)
    semantic_centroids = class_centroids(semantic, labels)

    raw_metrics = sample_separability_metrics(
        raw, labels, args.seed, args.silhouette_sample_size
    )
    semantic_metrics = sample_separability_metrics(
        semantic, labels, args.seed, args.silhouette_sample_size
    )
    raw_centroid_metrics = prototype_distance_metrics(raw_centroids)
    semantic_centroid_metrics = prototype_distance_metrics(semantic_centroids)
    text_metrics = prototype_distance_metrics(text_selected)
    alignment_metrics = semantic_text_alignment_metrics(
        semantic_centroids, text_selected
    )

    metrics = {
        "raw_point_features_64d": raw_metrics,
        "projected_semantic_features_512d": semantic_metrics,
        "raw_point_class_centroids_64d": raw_centroid_metrics,
        "semantic_point_class_centroids_512d": semantic_centroid_metrics,
        "frozen_text_prototypes_512d": text_metrics,
        "semantic_text_alignment": alignment_metrics,
        "comparison": {
            "silhouette_gain_semantic_minus_raw": float(
                semantic_metrics["silhouette_cosine"]
                - raw_metrics["silhouette_cosine"]
            ),
            "inter_over_intra_ratio_gain": float(
                semantic_metrics["inter_over_intra_ratio"]
                - raw_metrics["inter_over_intra_ratio"]
            ),
            "nearest_centroid_accuracy_gain": float(
                semantic_metrics["heldout_nearest_centroid_accuracy"]
                - raw_metrics["heldout_nearest_centroid_accuracy"]
            ),
            "text_minus_raw_centroid_mean_pairwise_cosine_distance": float(
                text_metrics["mean_pairwise_cosine_distance"]
                - raw_centroid_metrics["mean_pairwise_cosine_distance"]
            ),
            "text_minus_raw_centroid_min_pairwise_cosine_distance": float(
                text_metrics["min_pairwise_cosine_distance"]
                - raw_centroid_metrics["min_pairwise_cosine_distance"]
            ),
        },
        "sampling": {
            "balanced_points_per_class": int(balanced_per_class),
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
        raw_point_features=raw,
        semantic_point_features=semantic,
        labels=labels,
        raw_point_centroids=raw_centroids,
        semantic_point_centroids=semantic_centroids,
        text_prototypes=text_selected,
        original_base_labels=np.asarray(original_ids, dtype=np.int64),
        class_names=np.asarray(selected_names, dtype=object),
    )

    methods = resolve_methods(args.method)
    for method in methods:
        plot_method(
            method=method,
            output_dir=output_dir,
            raw=raw,
            semantic=semantic,
            labels=labels,
            raw_centroids=raw_centroids,
            semantic_centroids=semantic_centroids,
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
        "Raw 64-D silhouette        : "
        f"{raw_metrics['silhouette_cosine']:.6f}"
    )
    print(
        "Semantic 512-D silhouette  : "
        f"{semantic_metrics['silhouette_cosine']:.6f}"
    )
    print(
        "Raw inter/intra ratio       : "
        f"{raw_metrics['inter_over_intra_ratio']:.6f}"
    )
    print(
        "Semantic inter/intra ratio  : "
        f"{semantic_metrics['inter_over_intra_ratio']:.6f}"
    )
    print(
        "Raw centroid accuracy       : "
        f"{raw_metrics['heldout_nearest_centroid_accuracy']:.6f}"
    )
    print(
        "Semantic centroid accuracy  : "
        f"{semantic_metrics['heldout_nearest_centroid_accuracy']:.6f}"
    )
    print(
        "Raw centroid mean distance  : "
        f"{raw_centroid_metrics['mean_pairwise_cosine_distance']:.6f}"
    )
    print(
        "Text prototype mean distance: "
        f"{text_metrics['mean_pairwise_cosine_distance']:.6f}"
    )
    print(
        "Semantic->text top-1        : "
        f"{alignment_metrics['text_prototype_top1_accuracy']:.6f}"
    )
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
