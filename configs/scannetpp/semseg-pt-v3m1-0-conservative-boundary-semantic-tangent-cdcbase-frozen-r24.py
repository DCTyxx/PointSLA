"""Stage-1 conservative Base-only CDC geometry learning for ScanNet++.

PTv3 and the Base segmentation head are frozen. CDC, the semantic projector,
and safe-radius head are learned using only the ScanNet++ Base split. No Novel
registration data is visible during Stage 1.
"""

from pointcept.datasets.preprocessing.scannetpp.metadata.semantic_benchmark.top100 import (
    CLASS_LABELS_BASE,
)

_base_ = [
    "../scannet/semseg-pt-v3m1-0-conservative-boundary-semantic-tangent-cdcbase-frozen-r24.py"
]

# ScanNet++ Base-pretrained PTv3 checkpoint.
weight = "./exp/scannetpp/ptv3_bk/model/model_best.pth"

model = dict(
    num_classes=len(CLASS_LABELS_BASE),
)

data_root = "data/ScanNetpp"
data = dict(
    num_classes=len(CLASS_LABELS_BASE),
    ignore_index=-1,
    names=CLASS_LABELS_BASE,
    train=dict(
        type="ScanNetPPDataset_BASETrain",
        split="train_grid1mm_chunk6x6_stride3x3",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="SphereCrop", point_max=204800, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("color", "normal")),
        ],
        test_mode=False,
    ),
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
