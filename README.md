# PointSLA: Subspace Learning and Augmentation with Vision-Language Models for Few-Shot Point Cloud Segmentation

<p align="center">
  <b>Generalized Few-Shot 3D Point Cloud Segmentation</b><br>
  Local Subspace Learning + Semantic Scale Calibration + Virtual Feature Augmentation
</p>

> **Repository:** `DCTyxx/PointSLA`  
> **Default branch:** `feature/cdc-base-geometry-step1`
>
> PointSLA is developed on top of the original GFS-VL codebase. Historical `CDC` names are retained in several implementation files for backward compatibility, while the paper-level modules are referred to as **LSL** and **SSC**.

---

## Quick Start

Clone the current PointSLA repository:

```bash
git clone https://github.com/DCTyxx/PointSLA.git
cd PointSLA
git checkout feature/cdc-base-geometry-step1
```

If an existing local checkout still points to the former repository name, update the remote with:

```bash
git remote set-url origin https://github.com/DCTyxx/PointSLA.git
```

---

## Overview

Generalized few-shot point cloud segmentation aims to extend a base-trained point cloud segmentation model to novel classes using only a small annotated support set, while retaining its ability to segment base classes in the same scene.

PointSLA addresses a key limitation of few-shot adaptation: sparse novel-class support samples provide insufficient coverage of local intra-class feature variations. Our framework learns transferable local feature subspaces from base classes, calibrates admissible expansion scales with frozen vision-language semantics, and generates virtual point features to improve novel-class adaptation without sacrificing base-class performance.

The framework contains two core components:

- **Local Subspace Learning (LSL)** learns local intra-class variation directions from geometric neighborhoods while explicitly suppressing inter-class intrusion.
- **Semantic Scale Calibration (SSC)** projects point features into a frozen CLIP text space and estimates a conservative feature expansion radius that preserves target-class similarity and inter-class semantic margins.

The learned directions and calibrated scales are then used to synthesize virtual features for few-shot adaptation.

---

## Method

### 1. Local Subspace Learning

Given a point feature \(\mathbf{z}_i \in \mathbb{R}^{D}\), LSL predicts a low-rank local basis

\[
\mathbf{Q}_i \in \mathbb{R}^{D \times r},
\qquad
\mathbf{Q}_i^{\top}\mathbf{Q}_i = \mathbf{I}.
\]

An empirical local tangent basis is constructed from same-class neighbors using SVD. The learned basis is optimized to align with this empirical subspace while maintaining discrimination against different-class neighborhoods.

Main settings used in the current implementation:

- feature dimension: `D = 64`
- local subspace rank: `r = 24`
- same-class neighbors: `K+ = 32`
- different-class neighbors: `K- = 32`
- discriminative margin: `0.05`
- discriminative loss weight: `0.1`

The codebase still uses the historical `CDC` naming in several files, but this module corresponds to the paper-level **LSL** component.

### 2. Semantic Scale Calibration

SSC maps point features into the frozen CLIP text space through a learned semantic projector:

\[
\mathbf{e}_i = g(\mathbf{z}_i) \in \mathbb{R}^{512}.
\]

For a Base-class label \(y_i\), semantic consistency is evaluated using target-class similarity and the margin against competing Base classes. A candidate expansion radius is accepted only when all probe points preserve both quantities within a small tolerance.

Current semantic constraints use:

- text feature dimension: `512`
- target similarity tolerance: `0.005`
- inter-class margin tolerance: `0.005`
- path probes: `{0.25, 0.50, 0.75, 1.00}`

The resulting safe radius controls how far a feature may be expanded along its learned local subspace.

### 3. Virtual Feature Augmentation

For a support feature \(\mathbf{z}\), PointSLA generates a virtual feature as

\[
\tilde{\mathbf{z}}
=
\mathbf{z}
+
\rho\,\|\mathbf{z}\|\,\mathbf{Q}(\mathbf{z})\bar{\mathbf{a}},
\]

where \(\rho\) is the calibrated expansion scale and \(\bar{\mathbf{a}}\) is a normalized coefficient vector sampled in the learned local subspace.

These virtual features improve coverage of novel-class feature variation while keeping augmentation semantically conservative.

---

## Training Pipeline

The current implementation follows a staged training pipeline.

### Stage 1: Base Geometry and Semantic Calibration

- freeze PTv3 backbone
- freeze Base segmentation head
- learn LSL/CDC geometry
- learn semantic projector
- learn conservative safe-radius head
- use Base-class supervision only

For ScanNet++:

```bash
GPU_LIST=0,3,5 \
bash scripts/run_scannetpp_cdc_full_pipeline.sh
```

The full pipeline script automatically launches all later phases after Stage 1.

### Stage 2 Phase 1: Few-Shot Task Adaptation

For each fixed registration/support split:

- adapt PTv3 task model and Base/Novel classification heads
- keep the Stage-1 geometry stack frozen
- run each registration independently to avoid state leakage

Both 1-shot and 5-shot settings use five registration splits.

### Stage 2 Phase 2: Local Subspace Adaptation

- freeze the task model
- adapt the LSL/CDC component using trusted Base GT and K-shot Novel GT
- ignore unknown/unlabeled points
- do not use VLM pseudo labels for this phase

### Stage 2 Phase 3: Virtual Feature Refinement

- freeze PTv3
- freeze LSL/CDC
- freeze Base classifier
- refine the Novel classifier with real and virtual Novel features

The expansion radius can be automatically read from Stage-1 conservative semantic calibration.

---

## Current Results

### ScanNet200

| Setting | Base mIoU | Novel mIoU | All mIoU | HM |
|---|---:|---:|---:|---:|
| 1-shot | 69.16 | 30.30 | 38.48 | 42.13 |
| 5-shot | 68.89 | 32.85 | 40.43 | 44.49 |

### ScanNet++

| Setting | Base mIoU | Novel mIoU | All mIoU | HM |
|---|---:|---:|---:|---:|
| 1-shot | 62.94 | 23.17 | 39.08 | 33.84 |
| 5-shot | 62.05 | 27.43 | 41.28 | 38.03 |

---

## Ablation Utilities

### Direction Ablation

The ScanNet++ 1-shot direction ablation compares:

1. real-only training
2. isotropic random directions
3. random low-rank subspace
4. local PCA directions
5. learned LSL/CDC directions

Run with:

```bash
GPU_LIST=0,3,5 \
bash scripts/run_scannetpp_k1_direction_ablation.sh
```

### LSL Post-hoc Evaluation

```bash
bash scripts/run_split_phase2_lsl_posthoc_eval.sh
```

Typical geometry diagnostics include:

- Tangent Alignment Score (TAS)
- geometric discrimination gap
- explained variation ratio
- mean principal angle

### Virtual Sample Ablation

```bash
bash scripts/run_phase3_virtual_samples_ablation.sh
```

---

## Point/Text Feature-Space Analysis

The repository includes UMAP/t-SNE utilities for comparing the raw Point Encoder feature space with frozen CLIP text semantics.

```bash
GPU=4 bash scripts/run_scannetpp_point_text_umap.sh
```

Run both UMAP and t-SNE:

```bash
METHOD=both GPU=4 \
bash scripts/run_scannetpp_point_text_umap.sh
```

Outputs are written to:

```text
exp/scannetpp/feature_space_separability/
```

The analysis includes Point Encoder features, class centroids, CLIP text prototypes, quantitative cosine-distance statistics, and dimensionality-reduction visualizations.

---

## Feature Export and Visualization

### Export Phase-3 Feature Pairs

```bash
GPU=4 \
bash scripts/export_scannetpp_split_phase3_k1_regis3_features.sh
```

The exported cache contains real support features, labels, virtual features, directions, scene IDs, and point IDs.

### Four-Panel Segmentation Visualization

The visualization utilities are under:

```text
visilization/
```

They support synchronized views of:

```text
Input | GFS-VL Baseline | PointSLA (Ours) | Ground Truth
```

Here, **GFS-VL Baseline** explicitly denotes the original baseline method and is intentionally retained as a method name rather than a repository/project name.

---

## Data Preparation

This project follows the Pointcept data format.

### ScanNet / ScanNet200

Download ScanNet v2 and preprocess it with the existing Pointcept preprocessing pipeline.

### ScanNet++

Download ScanNet++ and preprocess it using:

```bash
python pointcept/datasets/preprocessing/scannetpp/preprocess_scannetpp.py \
  --dataset_root ${RAW_SCANNETPP_DIR} \
  --output_root ${PROCESSED_SCANNETPP_DIR} \
  --num_workers ${NUM_WORKERS}
```

For training splits, create sampled/chunked scenes with:

```bash
python pointcept/datasets/preprocessing/sampling_chunking_data.py \
  --dataset_root ${PROCESSED_SCANNETPP_DIR} \
  --grid_size 0.01 \
  --chunk_range 6 6 \
  --chunk_stride 3 3 \
  --split train \
  --num_workers ${NUM_WORKERS}
```

By default the ScanNet++ experiments expect:

```text
data/ScanNetpp
```

---

## Environment

PointSLA is based on Pointcept/PTv3 and extends the original GFS-VL implementation. A typical environment requires:

- Python 3.10+
- PyTorch 2.x
- CUDA 12.x or a compatible CUDA/PyTorch stack
- Pointcept dependencies
- OpenAI CLIP / frozen text embeddings
- pointops / pointops2

For dimensionality-reduction visualization:

```bash
pip install umap-learn scikit-learn matplotlib
```

---

## Repository Structure

```text
PointSLA/
├── configs/
│   ├── scannet/
│   ├── scannet200/
│   └── scannetpp/
├── pointcept/models/
│   ├── cdc_base_geometry.py
│   ├── tangent_cdc_base_geometry.py
│   ├── semantic_tangent_cdc_base_geometry.py
│   ├── boundary_semantic_tangent_cdc_base_geometry.py
│   ├── stage2_phase2_cdc_adapt.py
│   └── stage2_phase3_cdc_aug.py
├── scripts/
│   ├── run_scannet_cdc_full_pipeline.sh
│   ├── run_scannet200_cdc_full_pipeline.sh
│   ├── run_scannetpp_cdc_full_pipeline.sh
│   ├── run_scannetpp_k1_direction_ablation.sh
│   ├── run_split_phase2_lsl_posthoc_eval.sh
│   ├── run_phase3_virtual_samples_ablation.sh
│   └── run_scannetpp_point_text_umap.sh
├── tools/
│   ├── eval_phase2_lsl_metrics.py
│   ├── export_phase3_feature_pairs.py
│   └── visualize_point_text_feature_space.py
└── visilization/
```

---

## Notes on Naming

The current research code evolved from an earlier CDC-based implementation. For backward compatibility, several modules and scripts still contain `cdc` in their file names.

In the current PointSLA formulation:

- `CDC / tangent predictor` corresponds to **Local Subspace Learning (LSL)**
- semantic projector + conservative radius estimation corresponds to **Semantic Scale Calibration (SSC)**
- Phase-3 feature synthesis corresponds to **subspace-guided virtual feature augmentation**

Legacy configuration names such as `gfspretrain`, `gfsregistrain`, and some variables related to the original GFS-VL baseline are intentionally retained because they are part of the executable training/evaluation interface. They should not be interpreted as the current repository name.

---

## Acknowledgement

PointSLA is built upon the original **GFS-VL: Generalized Few-shot 3D Point Cloud Segmentation with Vision-Language Model** implementation and the excellent open-source projects Pointcept, PTv3, RegionPLC/PLA, OpenScene, and COSeg.

We thank the authors of these projects for making their code and benchmarks publicly available.

Original GFS-VL reference:

```bibtex
@inproceedings{an2025generalized,
  title={Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Model},
  author={An, Zhaochong and Sun, Guolei and Liu, Yun and Li, Runjia and Han, Junlin and Konukoglu, Ender and Belongie, Serge},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

---

## Citation

The PointSLA citation will be added after the paper metadata is finalized.
