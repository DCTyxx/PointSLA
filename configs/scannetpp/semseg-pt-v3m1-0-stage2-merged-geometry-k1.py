"""ScanNet++ K=1 merged Stage-1 + Stage-2 Phase-2 geometry learning."""

from pointcept.datasets.preprocessing.scannetpp.metadata.semantic_benchmark.top100 import (
    CLASS_LABELS_BASE,
)

_base_ = ["./semseg-pt-v3m1-0-stage2-phase2-k1-cdc-adapt.py"]

train = dict(type="MergedStage1Phase2GeometryTrainer")
task_weight = ""
merged_regis_name = "regis1"

hooks = [
    dict(type="MergedStage1Phase2CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="MergedStage1Phase2MetricSummary"),
    dict(type="CheckpointSaver", save_freq=None),
]

evaluate = False
find_unused_parameters = True
enable_amp = False
model = dict(type="Stage2MergedGeometrySegmentor")

data_root = "data/ScanNetpp"
data = dict(
    num_classes=len(CLASS_LABELS_BASE),
    names=CLASS_LABELS_BASE,
    train=dict(k_shot=1, seed=10, nb_mix_blks=3),
    val=dict(
        type="ScanNetPPDataset_BASETrain",
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("color", "normal")),
        ],
        test_mode=False,
    ),
)

epoch = 30
eval_epoch = 30
optimizer = dict(type="AdamW", lr=1.0e-3, weight_decay=1.0e-4)
param_dicts = None
scheduler = dict(
    _delete_=True,
    type="OneCycleLR",
    max_lr=1.0e-3,
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
