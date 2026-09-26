_base_ = ["./semseg-pt-v3m1-0-gfsregistrain_k1.py"]

# Global support-set PCA precision metric.
semantic_metric = "pca_mahalanobis"
metric_pca_rank = 32
metric_pca_shrinkage = 0.1
metric_ridge = 1e-3
metric_fisher_temperature = 1.0
