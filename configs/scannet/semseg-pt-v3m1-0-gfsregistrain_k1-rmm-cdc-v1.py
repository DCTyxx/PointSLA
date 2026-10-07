_base_ = ["./semseg-pt-v3m1-0-gfsregistrain_k1.py"]

# Preserve the original GFS-VL protocol and PTv3 backbone.  The only model-side
# change is a conditional-CDC Riemannian geometry branch plus a residual
# geometry adapter before the original base/novel segmentation heads.
train = dict(type="GFS_VL_RMM_CDC_Trainer")

model = dict(
    type="RegisTrainRMMCDCSegmentor",
    rmm_rank=4,
    rmm_hidden_dim=128,
    rmm_adapter_hidden_dim=128,
    rmm_epsilon_min=1e-4,
    rmm_epsilon_max=1e-2,
    rmm_ref_epsilon=1e-4,
    rmm_max_points=4096,
    rmm_refine_alpha=0.1,
    rmm_normalize_feature=True,
    # The original RMM treats X as observed data.  V1 therefore stops the CDC
    # target from moving with PTv3; segmentation still trains PTv3 through the
    # geometry-refined forward path.
    rmm_detach_cdc_target=True,
)

# L = L_seg + lambda_CDC * L_CDC
rmm_cdc_loss_weight = 0.01

# Explicitly keep baseline semantic matching unchanged.
semantic_metric = "cosine"
