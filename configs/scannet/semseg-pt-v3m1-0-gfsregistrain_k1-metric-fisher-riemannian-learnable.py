_base_ = ["./semseg-pt-v3m1-0-gfsregistrain_k1.py"]

# Stage-1 experiment: cosine remains responsible for pseudo-label selection
# and adaptive infilling. Learnable Fisher geometry is built in the trainable
# point-cloud backbone feature space and directly regularizes query backbone
# representations through the Fisher metric CE.
semantic_metric = "cosine"
train = dict(type="GFS_VL_LearnableFisher_Trainer")

# G_c^LF = G_c^{F/2} [I + alpha A_c A_c^T] G_c^{F/2}
# G_c^F  = (Sigma_c + eps I)^(-1)
# A_c    = h_theta(mu_c, Lambda_c, R_c)
# PT-v3 decoder output is 64-D in the current RegisTrainSegmentor.
metric_fisher_learnable = True
metric_fisher_feat_dim = 64
metric_fisher_spectral_rank = 32
metric_fisher_low_rank = 8
metric_fisher_hidden_dim = 128
metric_fisher_alpha = 0.1
metric_fisher_eps = 1e-3
metric_fisher_shrinkage = 0.1
metric_fisher_relation_topk = 3
metric_fisher_init_scale = 1e-3
metric_fisher_match_temperature = 5.0

# Stage-1 optimization. The Fisher metric loss now backpropagates into the
# point-cloud backbone query features; support-side moments are detached.
metric_fisher_lr = 1e-4
metric_fisher_weight_decay = 0.0
metric_fisher_loss_weight = 0.05
metric_fisher_loss_temperature = 5.0
metric_fisher_reg_weight = 1e-2
metric_fisher_max_points = 4096
metric_fisher_grad_clip = 5.0

num_worker_test = 0
