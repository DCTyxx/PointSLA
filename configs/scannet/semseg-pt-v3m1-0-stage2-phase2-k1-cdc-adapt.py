"""Stage-2 Phase-2: adapt CDC on each frozen K=1 task-specific PTv3 space.

This config is intentionally regis-agnostic. The launcher supplies the matching
Phase-1 checkpoint and registration seed for regis1..regis5, so one command can
run all five splits without sharing state between them.
"""

_base_ = ["./semseg-pt-v3m1-0-stage2-phase1-k1-frozen-cdc.py"]

# Phase 2 no longer needs the GFS-VL 3D-VLM pseudo-label trainer. It uses only
# Base GT + inserted K-shot Novel GT already present in ScanNetDataset_REGISTrain.
train = dict(type="DefaultTrainer")

# The launcher overrides these two fields independently for each registration
# split and the strict loader checks the complete Phase-1 state dict.
phase1_weight = ""
phase2_regis_name = "regis1"

hooks = [
    dict(type="Stage2Phase2CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    # Persist epoch-average geometry metrics in JSON. The all-regis launcher
    # aggregates the five final JSON files into mean/std CSV + JSON summaries.
    dict(type="Stage2Phase2MetricSummary"),
    dict(type="CheckpointSaver", save_freq=None),
]

evaluate = False
find_unused_parameters = False

# Frozen PTv3 runs in eval mode. Keep Phase 2 in fp32, matching the Stage-1
# geometry setup and avoiding the known eval-mode spconv AMP autotuning issue.
enable_amp = False

model = dict(
    type="Stage2Phase2CDCAdaptSegmentor",
)

# Default values correspond to regis1. The all-regis launcher changes only
# ``seed`` to 10/20/30/40/50 while keeping K=1 and the same insertion protocol.
data = dict(
    train=dict(
        k_shot=1,
        seed=10,
        nb_mix_blks=3,
    )
)

# CDC adaptation is short and conservative: only cdc_head has requires_grad=True.
epoch = 10
eval_epoch = 10
optimizer = dict(type="AdamW", lr=1.0e-4, weight_decay=1.0e-4)
param_dicts = None

# IMPORTANT: Phase-1 inherits a MultiStepLR with ``milestones`` and ``gamma``.
# The config system recursively merges dictionaries unless ``_delete_=True`` is
# set. Without deleting the inherited scheduler dict those MultiStepLR-only
# fields are passed into OneCycleLR and cause:
#   OneCycleLR.__init__() got an unexpected keyword argument 'milestones'
# Replace the scheduler dictionary completely here.
scheduler = dict(
    _delete_=True,
    type="OneCycleLR",
    max_lr=1.0e-4,
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
