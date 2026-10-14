"""Stage-2 Phase-1: GFS-VL 5-shot registration with frozen CDC.

The complete Stage-1 best checkpoint initializes PTv3, the Base segmentation
head and the geometry stack. During this phase PTv3/base/novel segmentation
heads follow the original GFS-VL 5-shot registration training, while CDC, the
semantic projector and the safe-radius head remain frozen and do not
participate in the loss.

Important: run one registration split per process. ``GFS_VL_Trainer`` updates
one model sequentially when multiple entries are placed in ``regis_train_list``;
that would make regis2 inherit the adapted weights of regis1, etc. Keeping a
single registration here preserves the required independent provenance from
the same Stage-1 checkpoint for every 5-shot split. A launcher can override
``regis_train_list`` to ["regis2"], ..., ["regis5"] in separate runs.
"""

_base_ = ["./semseg-pt-v3m1-0-gfsregistrain_k5.py"]

# 5-shot support protocol. The inherited registration datasets use this value
# to read/generate sc200_regis_5_{10,20,30,40,50}.txt.
k_shot = 5

# Default to regis1 only. Run regis1..regis5 independently from the same
# Stage-1 checkpoint rather than chaining task adaptation across registrations.
regis_train_list = ["regis1"]

# Strict Phase-1 loader restores every Stage-1 key except the newly introduced
# novel_seg_head, and verifies PTv3/base-head trainability and CDC freezing.
hooks = [
    dict(type="Stage2Phase1CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver_MultiRegis", save_freq=None),
    dict(type="GFS_MultiregTrain_PreciseEvaluator", test_last=True),
]

# Shared Stage-1 initialization for all K=5 registration splits.
backbone_weight = (
    "./exp/scannet/conservative_boundary_semantic_tangent_r24_v1/"
    "model/model_best.pth"
)

model = dict(
    type="Stage2Phase1FrozenCDCRegisSegmentor",
    # Keep the exact Stage-1 CDC architecture so checkpoint loading is
    # shape/name compatible and the learned Base geometry prior is preserved.
    cdc=dict(
        rank=24,
        hidden_dim=128,
        bandwidth_dim=32,
        h_min=0.01,
        h_max=0.05,
        sampling_method="loguniform",
        probe_h=0.02,
        max_points=4096,
        tikhonov_epsilon=1.0e-4,
        loss_weight=1.0,
    ),
    tangent=dict(
        knn=32,
        inter_knn=32,
        max_bank_points=4096,
        max_anchors=256,
        loss_weight=1.0,
        disc_loss_weight=0.1,
        disc_margin=0.05,
        projection_eps=1.0e-4,
    ),
    text_constraint=dict(
        text_dim=512,
        projector_hidden_dim=256,
        logit_scale=20.0,
        align_loss_weight=1.0,
        direction_loss_weight=0.0,
        warmup_epochs=1000000,
        positive_tolerance=0.005,
        margin_tolerance=0.005,
        probe_radius=0.10,
        num_direction_probes=1,
        max_points=1024,
    ),
    safe_radius=dict(
        hidden_dim=128,
        max_radius=0.15,
        initial_radius=0.05,
        num_directions=4,
        path_fractions=(0.25, 0.50, 0.75, 1.00),
        warmup_epochs=1,
        qr_eps=1.0e-6,
        boundary_step=0.005,
        boundary_refine_steps=3,
        boundary_safety_margin=0.0025,
        boundary_loss_weight=1.0,
        overshoot_loss_weight=0.5,
        smooth_l1_beta=0.1,
        step2_reference_quantile=0.05,
        safe_loss_weight=0.0,
        expansion_loss_weight=0.0,
    ),
)
