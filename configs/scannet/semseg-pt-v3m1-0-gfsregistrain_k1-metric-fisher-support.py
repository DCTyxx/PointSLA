_base_ = ["./semseg-pt-v3m1-0-gfsregistrain_k1.py"]

# Step 2: fixed support-conditioned Fisher geometry.
semantic_metric = "fisher_support"
metric_pca_rank = 32
metric_pca_shrinkage = 0.1
metric_ridge = 1e-3
metric_fisher_temperature = 1.0

# q_tilde^(c) = q + gamma * a_c, where a_c are support-prototype text logits.
metric_fisher_support_gamma = 0.2

num_worker_test = 0
