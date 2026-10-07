"""Boundary-supervised QR Tangent-CDC + semantic safe-radius Step 1.

This is the replacement for the saturated ``safe hinge + expansion reward``
variant. Geometry remains QR Tangent-CDC; the radius head is now trained against
an explicit Base-only semantic boundary target rho* searched along each sampled
tangent direction.
"""

_base_ = ["./semseg-pt-v3m1-0-tangent-cdcbase-frozen-r24.py"]

train = dict(type="BoundarySemanticTangentCDCTrainer")
find_unused_parameters = True
enable_amp = False

model = dict(
    type="BoundarySemanticTangentCDCBaseGeometrySegmentor",
    freeze_backbone=True,
    freeze_seg_head=True,
    safe_radius=dict(
        hidden_dim=128,
        max_radius=0.15,
        initial_radius=0.05,
        num_directions=4,
        # A candidate radius is considered safe only if the path probes remain
        # semantically valid at both half-distance and the endpoint.
        path_fractions=(0.5, 1.0),
        warmup_epochs=1,
        qr_eps=1.0e-6,
        # Explicit semantic-boundary search. The largest prefix-safe coarse
        # radius is refined twice by bisection inside the first unsafe interval.
        boundary_grid=(0.025, 0.050, 0.075, 0.100, 0.125, 0.150),
        boundary_refine_steps=2,
        boundary_loss_weight=1.0,
        # Overshooting rho* directly creates unsafe virtual features, so use a
        # modest asymmetric penalty on top of normalized SmoothL1 regression.
        overshoot_loss_weight=0.5,
        smooth_l1_beta=0.1,
        # Legacy indirect-objective weights are unused by the boundary model.
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
    safe_radius_directions=8,
    num_workers=0,
)
