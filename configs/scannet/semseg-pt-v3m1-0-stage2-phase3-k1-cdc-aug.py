"""Stage-2 Phase-3: CDC-guided Novel feature augmentation for K=1 GFS-VL.

Each registration split starts from its matching Phase-2 checkpoint.  PTv3,
CDC and the Base head are frozen; only the Novel head is refined using trusted
Base+Novel GT plus online CDC-generated Novel tangent features.
"""

_base_ = ["./semseg-pt-v3m1-0-stage2-phase2-k1-cdc-adapt.py"]

train = dict(type="DefaultTrainer")

# Supplied independently by the all-regis launcher.
phase2_weight = ""
phase3_regis_name = "regis1"

hooks = [
    dict(type="Stage2Phase3CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    # Lightweight validation every epoch.  CheckpointSaver immediately consumes
    # its HM metric and therefore stores model_best by generalized few-shot HM.
    dict(type="Stage2Phase3GFSEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    # At the end, run the repository's precise fragment-based GFSSemSegTester on
    # the best-HM checkpoint and persist Base/Novel/HM metrics to JSON.
    dict(type="Stage2Phase3PreciseEvaluator", test_last=False),
]

evaluate = True
find_unused_parameters = False

# Keep frozen PTv3/CDC geometry in fp32, matching Phase 2.
enable_amp = False

model = dict(
    type="Stage2Phase3CDCAugSegmentor",
    virtual_samples=4,
    virtual_radius=0.025,
    virtual_loss_weight=0.5,
)

# Default is regis1.  The launcher changes only the split seed and checkpoint.
data = dict(
    train=dict(
        k_shot=1,
        seed=10,
        nb_mix_blks=3,
    )
)

# Only the small Novel linear head is optimized in Phase 3.
epoch = 10
eval_epoch = 10
optimizer = dict(type="AdamW", lr=1.0e-3, weight_decay=1.0e-4)
param_dicts = None

# Replace the inherited Phase-2 scheduler completely.
scheduler = dict(
    _delete_=True,
    type="OneCycleLR",
    max_lr=1.0e-3,
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
