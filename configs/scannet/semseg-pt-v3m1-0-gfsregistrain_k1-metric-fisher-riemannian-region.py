_base_ = ["./semseg-pt-v3m1-0-gfsregistrain_k1-metric-fisher-riemannian-learnable.py"]

# Stage-3 experiment: start from the successful Stage-1 Fisher-to-backbone
# regularization, keep cosine pseudo-label selection/adaptive infilling exactly
# unchanged, and improve only the few-shot support geometry estimation.
train = dict(type="GFS_VL_LearnableFisherRegion_Trainer")

# Closed-form local support-region exploration.
# For retained/tangent covariance eigenvalues lambda and residual variance v:
#   lambda_tilde = (1 + eta_parallel^2) * lambda
#   v_tilde      = (1 + eta_perp^2) * v
# This is equivalent to marginalizing anisotropic zero-mean perturbations around
# support features, without explicitly sampling synthetic support points.
metric_fisher_region_exploration = True
metric_fisher_region_tangent_scale = 0.5
metric_fisher_region_residual_scale = 0.1
