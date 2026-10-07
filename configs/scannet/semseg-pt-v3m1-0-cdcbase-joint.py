"""Step 1 (recommended): freeze-then-unfreeze CDC base geometry training.

Strict protocol:
- base-class training split only;
- base-class labels only;
- base-class CLIP text names only;
- no registration/few-shot data and no novel class names;
- PTv3 frozen for the first 5 epochs, then jointly fine-tuned with a 10x
  smaller learning rate than CDC/projector;
- the original base segmentation loss anchors PTv3 after unfreezing.
"""

_base_ = ["./semseg-pt-v3m1-0-cdcbase.py"]

model = dict(
    type="CDCBaseGeometryJointSegmentor",
    # False means not permanently frozen; the warm-up schedule below controls it.
    freeze_backbone=False,
    freeze_seg_head=True,
    backbone_freeze_epochs=5,
    seg_loss_weight=1.0,
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=1.0,
            ignore_index=-1,
        ),
    ],
    # Keep the existing CDC/text settings from the base config. The text
    # direction constraint is also enabled after its 5-epoch warm-up.
)

# CDC head + semantic projector: 1e-3.
# PTv3: 1e-4 after epoch 5. During epochs 1-5 requires_grad=False, so the
# optimizer carries the params but does not update them.
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.0001)
param_dicts = [dict(keyword="backbone", lr=0.0001)]
scheduler = dict(
    type="OneCycleLR",
    # build_optimizer creates [default, backbone] groups in this order.
    max_lr=[0.001, 0.0001],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
