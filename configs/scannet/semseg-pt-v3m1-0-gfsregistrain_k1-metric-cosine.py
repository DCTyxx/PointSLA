_base_ = ["./semseg-pt-v3m1-0-gfsregistrain_k1.py"]

# Original GFS-VL matching.
semantic_metric = "cosine"
metric_pca_rank = 32
metric_pca_shrinkage = 0.1
metric_ridge = 1e-3
metric_fisher_temperature = 1.0

# Embedded evaluation is kept in-process; disable worker spawning to avoid
# SSH/pipe BrokenPipe failures after training.
num_worker_test = 0
