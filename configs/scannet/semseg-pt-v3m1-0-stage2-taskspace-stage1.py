"""Split protocol, Step A: original Base-only Stage-1 on a task-adapted ScanNet model.

The official GFS-VL registration checkpoint supplies the already fine-tuned
PTv3/Base/Novel heads.  They are frozen.  The original conservative Stage-1
recipe is then run unchanged on BaseTrain only, learning CDC + semantic
projector + safe-radius head before the ordinary Stage-2 Phase-2 fine-tuning.
"""

from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
    CLASS20_LABELS_BASE,
    CLASS20_LABELS_BASE_NOVEL,
)

_base_ = [
    "./semseg-pt-v3m1-0-conservative-boundary-semantic-tangent-cdcbase-frozen-r24.py"
]

# Supplied per registration by the launcher from pres_exp/sc_k1 or sc_k5.
task_weight = ""
task_stage1_regis_name = "regis1"

# Replace the original Base-backbone loader with an official task-checkpoint
# loader.  All subsequent Stage-1 training/evaluation logic is unchanged.
hooks = [
    dict(type="TaskSpaceStage1CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=10),
]

model = dict(
    type="Stage2TaskSpaceStage1Segmentor",
    num_base_classes=len(CLASS20_LABELS_BASE),
    num_novel_classes=len(CLASS20_LABELS_BASE_NOVEL) - len(CLASS20_LABELS_BASE),
)

# IMPORTANT: all optimization, tangent, text and safe-radius hyperparameters are
# inherited verbatim from the validated previous Stage-1 recipe:
# r=24, KNN=32, lambda_disc=0.1, 30 epochs, AdamW 1e-3, Base-only P05 radius.
