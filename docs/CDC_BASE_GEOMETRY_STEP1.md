# Step 1: Text-Constrained CDC Base Geometry Learning

This branch adds Step 1 of the proposed manifold-expansion pipeline on top of
the **original GFS-VL baseline**. It intentionally contains no novel/few-shot
feature-generation logic.

## Goal

Use abundant base-class data to learn a transferable CDC/Riemannian local
geometry prior in the final PTv3 point-feature space. A **frozen CLIP text
space** is used as a stable semantic prior so that learned CDC directions do
not easily destroy class semantics.

The recommended implementation uses a freeze-then-unfreeze schedule:

- **Epochs 1-5:** PTv3 is frozen. Train CDC and the PTv3-to-CLIP semantic bridge.
- **Epoch 6 onward:** PTv3 is unfrozen and jointly optimized with a 10x smaller
  learning rate than CDC/projector.
- **Original base segmentation head:** frozen, but its CE + Lovasz objective
  anchors PTv3 after unfreezing.
- **CLIP text encoder/prototypes:** always frozen.

## Strict base-only protocol

Step 1 may use only:

- base-class training/validation point clouds;
- base-class GT labels;
- base-class names encoded by the frozen CLIP text encoder.

Step 1 explicitly forbids registration/few-shot sets, novel GT labels, novel
pseudo labels/prototypes, and novel class names/text embeddings. The trainer
contains runtime guards for these conditions. Hyper-parameter selection should
also use base-only held-out diagnostics rather than novel few-shot mIoU.

## Objectives

### 1. CDC conditional metric matching

For a clean base feature `z`, sample a bandwidth and Gaussian probe:

```text
z_h = z + h * eps
```

The CDC head predicts a rank-`r` factor `U`, with local metric/covariance factor
`Gamma = U U^T`. Training uses the low-rank Frobenius objective equivalent to
the conditional metric-matching target, avoiding explicit `D x D` target
materialization.

### 2. Frozen-text semantic bridge

A trainable semantic projector maps the 64-D PTv3 point feature to the 512-D
CLIP text space. Only base-class text prototypes are encoded and they remain
fixed.

### 3. Text-constrained CDC directions

After the 5-epoch warm-up, directions sampled from the CDC span are probed in
both signs:

```text
delta = normalize(U a)
z+ = z + radius * delta
z- = z - radius * delta
```

A probe is penalized when either its GT-text similarity or its GT-vs-other-base
text margin drops beyond the configured tolerance. For this directional loss,
projector parameters are detached so the semantic bridge cannot simply move to
make an unsafe CDC direction appear valid.

### 4. Base segmentation anchor after PTv3 unfreezing

The original base segmentation head stays frozen, while CE + Lovasz loss
backpropagates into PTv3 after epoch 5. This prevents CDC/text optimization from
arbitrarily distorting the original task representation.

## Three Step-1 CDC diagnostics

The recommended joint config evaluates only three primary CDC metrics on the
held-out **base validation split**, every 5 epochs and at the final epoch.

### TAS - Tangent Alignment Score (higher is better)

For each held-out base anchor, build an empirical local tangent basis from the
same-class nearest-neighbor residuals using SVD. Compare it with the CDC
subspace:

```text
TAS = (1/r) * || Q_cdc^T Q_emp ||_F^2
```

TAS directly measures whether CDC principal directions agree with real local
base-class feature variation.

### GDG - Geometry Discrimination Gap (higher is better)

Project held-out same-class and different-class residuals into the CDC
subspace:

```text
GDG = IntraEVR - InterEVR
```

A high GDG means the CDC subspace explains class-internal variation better than
cross-class variation.

### SVR - Semantic Valid Rate (higher is better)

Use normalized CDC singular directions with a fixed relative probe radius, in
both positive and negative directions. A probe is valid only when both its
GT-text similarity and GT-vs-other-base margin remain within tolerance. Using
normalized fixed-radius directions prevents a collapsed CDC factor from
trivially achieving high SVR.

Training logs report exactly:

```text
CDC geometry eval (BASE-ONLY) | TAS: ... | GDG: ... | SVR: ...
```

TensorBoard keys are:

```text
cdc_geometry/TAS
cdc_geometry/GDG
cdc_geometry/SVR
```

## Recommended configuration

```text
configs/scannet/semseg-pt-v3m1-0-cdcbase-joint.py
```

Important defaults:

```text
CDC/projector LR : 1e-3
PTv3 LR          : 1e-4
PTv3 freeze      : first 5 epochs
CDC rank         : 8
geometry eval    : every 5 epochs
TAS/GDG kNN      : 16
SVR directions   : top 4 CDC singular directions, both signs
```

The config expects the original GFS-VL ScanNet base checkpoint at:

```text
./exp/scannet/ptv3_bk/model/model_best.pth
```

## Start Step 1

```bash
git fetch origin
git checkout feature/cdc-base-geometry-step1
git pull origin feature/cdc-base-geometry-step1

ls ./exp/scannet/ptv3_bk/model/model_best.pth

sh scripts/train.sh -p python -d scannet \
  -c semseg-pt-v3m1-0-cdcbase-joint \
  -n cdc_base_geometry_step1_joint -g 4
```

Expected behavior:

- epochs 1-5: `backbone_frozen: 1`;
- epoch 6 onward: PTv3 is unfrozen and `backbone_frozen: 0`;
- text-direction loss becomes active after the warm-up;
- epochs 5, 10, 15, ... and the final epoch print TAS/GDG/SVR;
- checkpoints are written under the experiment's `model/` directory.

## Output for later few-shot use

The Step-1 checkpoint contains PTv3, the original segmentation head, the CDC
head and semantic projector. The later few-shot stage can query the learned CDC
geometry through:

```python
factor = model.get_cdc_factor(feature, relative_h=0.02)
```

Step 2 is intentionally not implemented here yet.
