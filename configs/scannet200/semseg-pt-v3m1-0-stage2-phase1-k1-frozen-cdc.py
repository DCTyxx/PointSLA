"""Stage-2 Phase-1 for ScanNet200 K=1 with the Stage-1 CDC stack frozen."""

_base_ = ["./semseg-pt-v3m1-0-gfsregistrain_k1.py"]

k_shot = 1
regis_train_list = ["regis1", "regis2", "regis3", "regis4", "regis5"]

hooks = [
    dict(type="Stage2Phase1CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver_MultiRegis", save_freq=None),
    dict(type="GFS_MultiregTrain_PreciseEvaluator", test_last=True),
]

backbone_weight = "./exp/scannet200/cdc_stage1_r24/model/model_best.pth"

model = dict(
    type="Stage2Phase1FrozenCDCRegisSegmentor",
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
