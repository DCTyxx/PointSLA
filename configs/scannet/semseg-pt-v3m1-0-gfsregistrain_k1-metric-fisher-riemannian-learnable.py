_base_ = ["./semseg-pt-v3m1-0-gfsregistrain_k1.py"]

# The legacy text-classifier Fisher is not used in this experiment.  The
# trainer replaces the temporary cosine matcher with the support-conditioned
# covariance Fisher geometry defined below.
semantic_metric = "cosine"
train = dict(type="GFS_VL_LearnableFisher_Trainer")

# G_c^LF = G_c^{F/2} [I + alpha A_c A_c^T] G_c^{F/2}
# G_c^F  = (Sigma_c + eps I)^(-1)
# A_c    = h_theta(mu_c, Lambda_c, R_c)
metric_fisher_learnable = True
metric_fisher_feat_dim = 512
metric_fisher_low_rank = 8
metric_fisher_hidden_dim = 128
metric_fisher_alpha = 0.25
metric_fisher_eps = 1e-3
metric_fisher_shrinkage = 0.1
metric_fisher_relation_topk = 3
metric_fisher_init_scale = 1e-3
metric_fisher_match_temperature = 1.0

# Optimization of h_theta.  The frozen 3D VLM is never unfrozen.
metric_fisher_lr = 1e-3
metric_fisher_weight_decay = 0.0
metric_fisher_loss_weight = 0.1
metric_fisher_loss_temperature = 1.0
metric_fisher_reg_weight = 1e-3
metric_fisher_max_points = 4096

# Keep embedded evaluation in-process.
num_worker_test = 0
