"""Step 1: frozen-PTv3 CDC base-geometry learning with rank 24.

Protocol:
- base-class train/val only;
- PTv3 is frozen for the entire Step-1 training;
- original base segmentation head is frozen;
- CDC geometry head and semantic projector are trainable;
- CLIP text prototypes are frozen and restricted to base classes;
- CDC rank r=24 in the 64-D PTv3 feature space.
"""

_base_ = ["./semseg-pt-v3m1-0-cdcbase.py"]

# PTv3/seg-head are already requires_grad=False before DDP construction in the
# frozen model, so no dynamic-unfreeze handling is needed.
find_unused_parameters = False

model = dict(
    type="CDCBaseGeometrySegmentor",
    freeze_backbone=True,
    freeze_seg_head=True,
    cdc=dict(
        # D=64 PTv3 feature space, local CDC subspace rank r=24.
        rank=24,
    ),
    text_constraint=dict(
        # Keep the non-saturated semantic probe setting established in v4.
        direction_loss_weight=0.10,
        positive_tolerance=0.005,
        margin_tolerance=0.005,
        probe_radius=0.10,
    ),
)

# Only CDC + semantic projector are optimized.
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.0001)
param_dicts = None
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.001,
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# For r=24, empirical tangent estimation needs at least 24 same-class residuals.
# Use k=32 so TAS is well-defined and still local. GDG uses the full r=24 CDC
# subspace. SVR probes the top 8 CDC singular directions for a practical balance
# between coverage and evaluation cost.
cdc_geometry_eval = dict(
    enabled=True,
    interval=5,
    max_batches=32,
    max_points_per_class=256,
    num_anchors=256,
    knn=32,
    inter_knn=32,
    svr_directions=8,
    num_workers=0,
)
