"""Split protocol, Step A: Base-only Stage-1 on a task-adapted ScanNet200 model."""

from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
    CLASS_LABELS_BASE,
    CLASS_LABELS_BASE_NOVEL,
)

_base_ = [
    "./semseg-pt-v3m1-0-conservative-boundary-semantic-tangent-cdcbase-frozen-r24.py"
]

task_weight = ""
task_stage1_regis_name = "regis1"

hooks = [
    dict(type="TaskSpaceStage1CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=10),
]

model = dict(
    type="Stage2TaskSpaceStage1Segmentor",
    num_base_classes=len(CLASS_LABELS_BASE),
    num_novel_classes=len(CLASS_LABELS_BASE_NOVEL) - len(CLASS_LABELS_BASE),
)

# All Stage-1 geometry / semantic-boundary hyperparameters are inherited
# unchanged: r=24, KNN=32, lambda_disc=0.1, 30 epochs, AdamW 1e-3, P05 radius.
