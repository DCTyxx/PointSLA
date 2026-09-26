_base_ = ["./semseg-pt-v3m1-0-gfsregistrain_k1.py"]

# Point-dependent Fisher pullback metric induced by the frozen CLIP classifier.
semantic_metric = "fisher_riemannian"
metric_pca_rank = 32
metric_pca_shrinkage = 0.1
metric_ridge = 1e-3
metric_fisher_temperature = 1.0
