"""Conservative boundary-supervised QR Tangent-CDC for Step-1.

Goal: learn geometry as before, but estimate a deliberately conservative
semantic-safe radius distribution on Base classes so Stage 2 has a reliable
reference range for novel few-shot manifold expansion.
"""

_base_ = ["./semseg-pt-v3m1-0-tangent-cdcbase-frozen-r24.py"]

train = dict(type="ConservativeBoundarySemanticTangentCDCTrainer")
find_unused_parameters = True
enable_amp = False

model = dict(
    type="ConservativeBoundarySemanticTangentCDCBaseGeometrySegmentor",
    freeze_backbone=True,
    freeze_seg_head=True,
    safe_radius=dict(
        hidden_dim=128,
        max_radius=0.15,
        initial_radius=0.05,
        num_directions=4,
        # Conservative path check: every candidate radius must remain semantically
        # valid at 25%, 50%, 75%, and 100% of the path.
        path_fractions=(0.25, 0.50, 0.75, 1.00),
        warmup_epochs=1,
        qr_eps=1.0e-6,
        # Dense search: 0.005 increments from 0 to 0.15, then three binary
        # refinements in the first unsafe interval. The oracle always returns the
        # verified-safe LOWER bound and subtracts another 0.0025 safety guard.
        boundary_step=0.005,
        boundary_refine_steps=3,
        boundary_safety_margin=0.0025,
        boundary_loss_weight=1.0,
        overshoot_loss_weight=0.5,
        smooth_l1_beta=0.1,
        # Stage-2 reference is the 5th percentile among positive conservative
        # Base-val oracle radii. Strict min/P01/P05/P10 are all logged as well.
        step2_reference_quantile=0.05,
        # Legacy indirect objective is disabled.
        safe_loss_weight=0.0,
        expansion_loss_weight=0.0,
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
    # Validation probes more directions than training for a stable Step-2 radius
    # reference distribution.
    safe_radius_directions=8,
    num_workers=0,
)
