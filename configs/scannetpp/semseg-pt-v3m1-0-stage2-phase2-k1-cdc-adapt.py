"""Stage-2 Phase-2: adapt CDC on each frozen ScanNet++ K=1 task space."""

_base_ = ["./semseg-pt-v3m1-0-stage2-phase1-k1-frozen-cdc.py"]

train = dict(type="DefaultTrainer")

phase1_weight = ""
phase2_regis_name = "regis1"

hooks = [
    dict(type="Stage2Phase2CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="Stage2Phase2MetricSummary"),
    dict(type="CheckpointSaver", save_freq=None),
]

evaluate = False
find_unused_parameters = False
enable_amp = False

model = dict(type="Stage2Phase2CDCAdaptSegmentor")

data = dict(
    train=dict(
        k_shot=1,
        seed=10,
        nb_mix_blks=3,
    )
)

epoch = 10
eval_epoch = 10
optimizer = dict(type="AdamW", lr=1.0e-4, weight_decay=1.0e-4)
param_dicts = None
scheduler = dict(
    _delete_=True,
    type="OneCycleLR",
    max_lr=1.0e-4,
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
