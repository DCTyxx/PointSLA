"""Joint orthonormal Tangent-CDC + semantic safe-radius Step 1.

Protocol:
- Base train/val only; no novel labels/names/support/query/registration data;
- PTv3 and original segmentation head remain frozen;
- CDC learns a rank-24 local tangent in the fixed 64-D PTv3 feature space;
- CDC factor is converted to Q by QR before tangent/radius use;
- Base CLIP alignment trains the semantic projector;
- a language-conditioned radius head learns how far a sampled tangent direction
  can be traversed while preserving the Base class semantics;
- semantic safety gradients update the radius head only (not CDC/projector).
"""

_base_ = ["./semseg-pt-v3m1-0-tangent-cdcbase-frozen-r24.py"]

train = dict(type="SemanticTangentCDCTrainer")
find_unused_parameters = False
enable_amp = False

model = dict(
    type="SemanticTangentCDCBaseGeometrySegmentor",
    freeze_backbone=True,
    freeze_seg_head=True,
    safe_radius=dict(
        hidden_dim=128,
        max_radius=0.15,
        initial_radius=0.05,
        num_directions=4,
        path_fractions=(0.5, 1.0),
        safe_loss_weight=1.0,
        expansion_loss_weight=0.01,
        warmup_epochs=1,
        qr_eps=1.0e-6,
    ),
    text_constraint=dict(
        align_loss_weight=1.0,
        direction_loss_weight=0.0,
        warmup_epochs=1000000,
        positive_tolerance=0.005,
        margin_tolerance=0.005,
        probe_radius=0.10,
    ),
)

epoch = 30
eval_epoch = 30
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

cdc_geometry_eval = dict(
    enabled=True,
    interval=5,
    max_batches=32,
    max_points_per_class=256,
    num_anchors=256,
    knn=32,
    inter_knn=32,
    svr_directions=8,
    safe_radius_directions=8,
    num_workers=0,
)
