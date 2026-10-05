_base_ = ["./semseg-pt-v3m1-0-gfsregistrain_k1.py"]

# Steps 3-4: support-conditioned Fisher geometry with a learnable residual
# calibration of class competition. Step 5 will add the base episodic trainer.
semantic_metric = "learnable_fisher"
metric_pca_rank = 32
metric_pca_shrinkage = 0.1
metric_ridge = 1e-3
metric_fisher_temperature = 1.0

# Fixed support-conditioning term: q_tilde^(c) = q + gamma * a_c + r_phi.
metric_fisher_support_gamma = 0.2

# Shared residual calibrator g_phi: 6 -> 16 -> 8 -> 1.
# The final layer is zero-initialized, so the initial learnable metric exactly
# recovers fisher_support with the same gamma.
metric_fisher_residual_scale = 0.5
metric_fisher_calibrator_hidden_dim = 16
metric_fisher_calibrator_bottleneck_dim = 8

# Step-4 objective:
# L = L_cls + lambda_m L_margin + lambda_p L_preserve.
metric_fisher_metric_temperature = 0.1
metric_fisher_margin = 0.1
metric_fisher_margin_loss_weight = 0.5
metric_fisher_preserve_loss_weight = 0.1

num_worker_test = 0
