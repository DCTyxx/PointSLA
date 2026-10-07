# Step 1: CDC Base Geometry Pretraining

This branch adds the first stage of the proposed manifold-expansion pipeline on
top of the **original GFS-VL baseline**. It does not contain novel/few-shot
feature generation logic.

## Goal

Use abundant base-class data to learn a transferable CDC/Riemannian local
geometry prior in the final PTv3 point-feature space, while a **frozen CLIP
text space** acts as a stable semantic prior that discourages CDC directions
from destroying class semantics.

Role split:

- **PTv3 (frozen):** original GFS-VL point representation.
- **CDC head (trainable):** learns low-rank local CDC factor `U(z,h)`, with
  `Gamma = U U^T`.
- **Semantic projector (trainable):** maps 64-D PTv3 features to 512-D CLIP
  text space.
- **CLIP text encoder/prototypes (frozen):** constrain CDC directional probes.
- **Original segmentation head (frozen):** kept only for checkpoint
  compatibility and not used to supervise CDC geometry.

## Objectives

### 1. CDC conditional metric matching

For a clean base feature `z`, sample a bandwidth and Gaussian probe:

```text
z_h = z + h * eps
```

The CDC head predicts rank-`r` factor `U`. Training uses the exact low-rank
Frobenius objective equivalent to the conditional metric-matching target, so a
full `D x D` target matrix is never materialized.

### 2. Frozen-text semantic bridge

The semantic projector is trained on clean base features using the GT base
class and its fixed CLIP text prototype. This only builds a bridge into the
fixed semantic coordinate system; PTv3 is not updated.

### 3. Text-constrained CDC directions

After a short warm-up, sample a random direction from the CDC span:

```text
delta = normalize(U a)
z+ = z + radius * delta
z- = z - radius * delta
```

For this loss, the projector parameters are detached. Therefore the projector
cannot move merely to make an unsafe CDC direction look valid; gradients act
on CDC. A virtual probe is penalized when either its GT-text similarity or its
GT-vs-other-text margin drops beyond a tolerance.

## Run

The default config expects the original GFS-VL ScanNet base checkpoint at:

```text
./exp/scannet/ptv3_bk/model/model_best.pth
```

Run:

```bash
sh scripts/train.sh -p python -d scannet \
  -c semseg-pt-v3m1-0-cdcbase \
  -n cdc_base_geometry_step1 -g 4
```

## Logged diagnostics

- `cdc_loss`: conditional metric-matching loss.
- `text_align_loss`: PTv3-to-frozen-CLIP bridge loss.
- `text_acc`: base GT text-retrieval accuracy of that bridge.
- `text_direction_loss`: semantic penalty on CDC-sampled directions.
- `text_virtual_margin`: mean GT-vs-best-other text margin after probing.
- `text_virtual_valid_rate`: fraction of probes satisfying both semantic
  constraints.
- `cdc_factor_energy`: factor-energy collapse/explosion diagnostic.
- `relative_h`: sampled relative Gaussian bandwidth.

## Output

The checkpoint contains the unchanged baseline modules plus `cdc_head` and
`semantic_projector`. The later few-shot stage can reuse the learned geometry
through:

```python
factor = model.get_cdc_factor(feature, relative_h=0.02)
```

Stage 2 is intentionally not implemented here.
