"""Diagnostic Tangent-CDC Step 1 on frozen PTv3 Base features.

Purpose:
- keep the PTv3 representation fixed;
- replace Gaussian metric-matching with direct Base-train local tangent
  supervision;
- encourage intra-class residuals to be explained more strongly than nearby
  inter-class residuals;
- keep the PTv3->CLIP projector only for Base-only semantic evaluation;
- deliberately disable text-direction regularization in this diagnostic run.

No novel labels, novel names, registration data, or few-shot support/query data
are used by this configuration.
"""

_base_ = ["./semseg-pt-v3m1-0-cdcbase-frozen-r24.py"]

# The inherited frozen-r24 config already keeps PTv3/seg-head frozen and AMP off.
find_unused_parameters = False
enable_amp = False

model = dict(
    type="TangentCDCBaseGeometrySegmentor",
    freeze_backbone=True,
    freeze_seg_head=True,
    # Inherited CDC rank is r=24 in the D=64 PTv3 feature space.
    tangent=dict(
        # Same-class feature KNN used to estimate Q_emp. K must be >= r.
        knn=32,
        # Nearest different-class residuals used by the discrimination hinge.
        inter_knn=32,
        # Class-balanced stochastic Base-train feature bank per iteration.
        max_bank_points=4096,
        max_anchors=256,
        # L_geo = L_tan + 0.1 * L_disc.
        loss_weight=1.0,
        disc_loss_weight=0.1,
        disc_margin=0.05,
        projection_eps=1.0e-4,
    ),
    text_constraint=dict(
        # Projector alignment is retained so held-out SVR is meaningful, but
        # language does not constrain CDC directions in this Tangent-CDC test.
        align_loss_weight=1.0,
        direction_loss_weight=0.0,
        warmup_epochs=1000000,
        positive_tolerance=0.005,
        margin_tolerance=0.005,
        probe_radius=0.10,
    ),
)

# Keep the optimization identical to the frozen-r24 CDC experiment so the main
# intervention is the geometry target rather than optimizer tuning.
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

# A 30-epoch diagnostic budget is sufficient to observe whether held-out TAS
# moves clearly above the r/D = 24/64 = 0.375 random-subspace reference.
epoch = 30
eval_epoch = 30

# Held-out Base validation geometry remains unchanged from frozen-r24, ensuring
# training uses Base-train KNN while TAS/GDG/SVR are measured independently on
# Base-val features.
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
