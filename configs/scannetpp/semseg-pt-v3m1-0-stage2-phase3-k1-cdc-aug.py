"""Stage-2 Phase-3: Novel feature augmentation for ScanNet++ K=1."""

_base_ = ["./semseg-pt-v3m1-0-stage2-phase2-k1-cdc-adapt.py"]

train = dict(type="DefaultTrainer")

phase2_weight = ""
phase3_regis_name = "regis1"

hooks = [
    dict(type="Stage2Phase3CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="Stage2Phase3GFSEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="Stage2Phase3PreciseEvaluator", test_last=False),
    # Reload model_best and export one deterministic train/val feature snapshot.
    # The direction-ablation launcher later keeps only the best-HM regis copy.
    dict(
        type="Stage2Phase3BestFeaturePairExporter",
        output_dir="best_feature_pairs",
        export_train=True,
        export_val=True,
        export_seed=3407,
        num_workers=0,
    ),
]

evaluate = True
find_unused_parameters = False
enable_amp = False

model = dict(
    type="Stage2Phase3CDCAugSegmentor",
    # Main method remains CDC by default. The dedicated direction-ablation
    # launcher overrides only augmentation_mode while keeping all other Phase-3
    # settings fixed.
    augmentation_mode="cdc",
    random_subspace_seed=0,
    local_pca_knn=32,
    local_pca_chunk_size=512,
    virtual_samples=4,
    virtual_radius=0.025,
    virtual_loss_weight=0.5,
)

data = dict(
    train=dict(
        k_shot=1,
        seed=10,
        nb_mix_blks=3,
    )
)

epoch = 10
eval_epoch = 10
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
