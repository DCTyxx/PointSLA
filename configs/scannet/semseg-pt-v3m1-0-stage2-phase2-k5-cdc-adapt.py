"""Stage-2 Phase-2: adapt CDC on each frozen K=5 task-specific PTv3 space.

Phase 1 has already adapted PTv3 and the Base/Novel segmentation heads to one
5-shot registration split while keeping the Stage-1 geometry stack frozen.
Phase 2 loads the matching Phase-1 checkpoint, freezes the task representation
and classifiers, and updates only ``cdc_head`` with Base GT + inserted 5-shot
Novel GT from the same registration split.

Run regis1..regis5 independently.  No Phase-2 state is shared across splits.
"""

_base_ = ["./semseg-pt-v3m1-0-stage2-phase1-k5-frozen-cdc.py"]

# Phase 2 no longer uses the GFS-VL 3D-VLM pseudo-label trainer.  Trusted labels
# are the Base GT plus Novel GT inserted from the fixed 5-shot registration set.
train = dict(type="DefaultTrainer")

# Override per registration at launch time.  The strict loader restores the
# complete matching Phase-1 state before CDC adaptation starts.
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

# Match Stage-1/K=1 Phase-2 geometry optimization: frozen PTv3 in fp32 and only
# CDC trainable.  This also avoids the eval-mode spconv AMP autotuning issue.
enable_amp = False

model = dict(
    type="Stage2Phase2CDCAdaptSegmentor",
)

# Default = regis1.  For regis2..5, keep k_shot=5 and override only seed and
# ``phase2_regis_name`` together with the corresponding Phase-1 checkpoint.
data = dict(
    train=dict(
        k_shot=5,
        seed=10,
        nb_mix_blks=3,
    )
)

# Conservative task-specific CDC adaptation.  Keep the same optimization as the
# K=1 experiment so 1-shot vs 5-shot remains a clean support-size comparison.
epoch = 10
eval_epoch = 10
optimizer = dict(type="AdamW", lr=1.0e-4, weight_decay=1.0e-4)
param_dicts = None

# Replace the inherited GFS-VL MultiStepLR dictionary completely.
scheduler = dict(
    _delete_=True,
    type="OneCycleLR",
    max_lr=1.0e-4,
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
