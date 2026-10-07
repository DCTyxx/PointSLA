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

# PTv3 parameters are registered by DDP from the beginning, but the warm-up
# forward uses no_grad(). Therefore DDP must tolerate unused PTv3 parameters in
# epochs 1-5; after epoch 5 the same parameters participate in backpropagation.
find_unused_parameters = True

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
# PTv3: 1e-4 after epoch 5. During epochs 1-5 the PTv3 forward is wrapped in
# no_grad(), so the optimizer carries the params but receives no PTv3 grads.
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

# Base-only held-out CDC geometry diagnostics. This is deliberately independent
# of the standard segmentation evaluator so no novel/few-shot information is
# involved in Step-1 model diagnosis or hyper-parameter selection.
cdc_geometry_eval = dict(
    enabled=True,
    interval=5,
    max_batches=32,
    max_points_per_class=256,
    num_anchors=256,
    knn=16,
    inter_knn=16,
    svr_directions=4,
    num_workers=4,
)
