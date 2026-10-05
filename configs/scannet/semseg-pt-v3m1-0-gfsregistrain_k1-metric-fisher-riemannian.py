_base_ = ["./semseg-pt-v3m1-0-gfsregistrain_k1.py"]

# Point-dependent Fisher pullback metric induced by the frozen CLIP classifier.
semantic_metric = "fisher_riemannian"
metric_pca_rank = 32
metric_pca_shrinkage = 0.1
metric_ridge = 1e-3
metric_fisher_temperature = 1.0

# Learnable Fisher-Riemannian deformation.  The semantic Fisher metric remains
# the anchor; W_theta is positive, diagonal, class-agnostic and identity-
# initialized, so training starts from the exact fixed Fisher baseline.
train = dict(type="GFS_VL_LearnableFisher_Trainer")
metric_fisher_learnable = True
metric_fisher_hidden_dim = 32
metric_fisher_residual_scale = 0.25
metric_fisher_max_log_weight = 2.0
metric_fisher_learn_temperature = True

# Metric supervision on annotated novel-class points.
metric_fisher_lr = 1e-3
metric_fisher_weight_decay = 0.0
metric_fisher_loss_weight = 0.1
metric_fisher_loss_temperature = 0.1
metric_fisher_reg_weight = 1e-3
metric_fisher_max_points = 4096

# Embedded evaluation is kept in-process; disable worker spawning to avoid
# SSH/pipe BrokenPipe failures after training.
num_worker_test = 0
