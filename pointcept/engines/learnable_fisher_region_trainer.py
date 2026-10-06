from .learnable_fisher import LearnableCovarianceFisher
from .learnable_fisher_trainer import GFS_VL_LearnableFisher_Trainer
from .train import TRAINERS


@TRAINERS.register_module("GFS_VL_LearnableFisherRegion_Trainer")
class GFS_VL_LearnableFisherRegion_Trainer(GFS_VL_LearnableFisher_Trainer):
    """Stage-3 trainer: Stage-1 Fisher regularization + support-region exploration.

    The segmentation/pseudo-label pipeline is exactly the Stage-1 pipeline:
    pseudo-label selection and adaptive infilling remain cosine based, while the
    Fisher metric CE regularizes trainable backbone query features.

    The only Stage-3 change is the support geometry. Instead of treating the
    observed support points as the entire class distribution, each support
    feature is interpreted as the centre of a small anisotropic local region.
    The region covariance is integrated in closed form before constructing the
    Fisher precision, so no synthetic support points or extra random sampling
    are required.
    """

    def build_zero_shot_model(self):
        super().build_zero_shot_model()

        matcher = self.learnable_fisher_matcher
        if not isinstance(matcher, LearnableCovarianceFisher):
            return

        enabled = bool(
            getattr(self.cfg, "metric_fisher_region_exploration", True)
        )
        tangent_scale = float(
            getattr(self.cfg, "metric_fisher_region_tangent_scale", 0.5)
        )
        residual_scale = float(
            getattr(self.cfg, "metric_fisher_region_residual_scale", 0.1)
        )
        matcher.configure_region_exploration(
            enabled=enabled,
            tangent_scale=tangent_scale,
            residual_scale=residual_scale,
        )

        self.logger.info(
            "=> Stage-3 Fisher support-region exploration: "
            f"enabled={matcher.region_exploration}, "
            f"tangent_scale={matcher.region_tangent_scale}, "
            f"residual_scale={matcher.region_residual_scale}, "
            f"tangent_variance_factor={1.0 + matcher.region_tangent_scale ** 2:.4f}, "
            f"residual_variance_factor={1.0 + matcher.region_residual_scale ** 2:.4f}; "
            "pseudo-label selection/infilling remain cosine"
        )
