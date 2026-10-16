"""Stage-1 conservative Base-only CDC geometry learning for ScanNet200.

This reuses the validated ScanNet CDC geometry recipe while switching the
benchmark-specific pieces to the ScanNet200 Base split. PTv3 and the Base
segmentation head are frozen; CDC, the semantic projector, and safe-radius head
are learned from Base data only. No Novel registration data is visible here.
"""

from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
    CLASS_LABELS_BASE,
)

_base_ = [
    "../scannet/semseg-pt-v3m1-0-conservative-boundary-semantic-tangent-cdcbase-frozen-r24.py"
]

# User-provided ScanNet200 Base-pretrained checkpoint.
weight = "./exp/scannet200/ptv3_bk/model_best.pth"

# The inherited text encoder is already the ScanNet200 CLIP configuration.
model = dict(
    num_classes=len(CLASS_LABELS_BASE),
)

# Strict Base-only ScanNet200 protocol. The inherited transforms and geometry
# hyperparameters are kept unchanged so ScanNet vs ScanNet200 differ only in
# benchmark split/class semantics and the pretrained Base checkpoint.
data_root = "data/ScanNet200"
data = dict(
    num_classes=len(CLASS_LABELS_BASE),
    ignore_index=-1,
    names=CLASS_LABELS_BASE,
    train=dict(
        type="ScanNet200Dataset_BASETrain",
        split="train",
        data_root=data_root,
    ),
    val=dict(
        type="ScanNet200Dataset_BASETrain",
        split="val",
        data_root=data_root,
    ),
)
