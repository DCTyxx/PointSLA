"""Conservative semantic-boundary Tangent-CDC.

This variant strengthens boundary supervision for a Step-2-safe reference:
- dense monotone radius scan (default 0.005 increments);
- multiple path probes inside every candidate radius;
- binary refinement after the first unsafe interval;
- a final safety guard is subtracted from the largest verified-safe lower bound.

The returned oracle radius is therefore deliberately a LOWER, conservative
estimate of the semantic boundary rather than an aggressive approximation.
"""

import math

import torch

from .builder import MODELS
from .boundary_semantic_tangent_cdc_base_geometry import (
    BoundarySemanticTangentCDCBaseGeometrySegmentor,
)


@MODELS.register_module()
class ConservativeBoundarySemanticTangentCDCBaseGeometrySegmentor(
    BoundarySemanticTangentCDCBaseGeometrySegmentor
):
    """Boundary Tangent-CDC with a conservative lower-bound oracle radius."""

    def __init__(self, safe_radius=None, **kwargs):
        safe_radius = {} if safe_radius is None else dict(safe_radius)
        super().__init__(safe_radius=safe_radius, **kwargs)

        self.boundary_step = float(safe_radius.get("boundary_step", 0.005))
        self.boundary_safety_margin = float(
            safe_radius.get("boundary_safety_margin", 0.0025)
        )
        self.step2_reference_quantile = float(
            safe_radius.get("step2_reference_quantile", 0.05)
        )

        if self.boundary_step <= 0.0:
            raise ValueError("safe_radius.boundary_step must be positive")
        if self.boundary_safety_margin < 0.0:
            raise ValueError("boundary_safety_margin must be non-negative")
        if not (0.0 <= self.step2_reference_quantile <= 1.0):
            raise ValueError("step2_reference_quantile must lie in [0, 1]")

        # Override the parent coarse grid with a dense uniform scan. The last
        # point is forced to rho_max to avoid floating-point accumulation drift.
        num_steps = int(math.ceil(self.safe_radius_max / self.boundary_step))
        grid = [
            min((i + 1) * self.boundary_step, self.safe_radius_max)
            for i in range(num_steps)
        ]
        if grid[-1] < self.safe_radius_max - 1e-8:
            grid.append(self.safe_radius_max)
        else:
            grid[-1] = self.safe_radius_max
        self.boundary_grid = tuple(float(v) for v in grid)

    @torch.no_grad()
    def oracle_safe_radius(self, feature, direction, labels):
        # Parent returns the largest radius that has been explicitly verified as
        # prefix-safe. Subtract an additional guard so Step 2 does not operate at
        # the measured boundary itself.
        lower_bound = super().oracle_safe_radius(feature, direction, labels)
        if self.boundary_safety_margin == 0.0:
            return lower_bound
        return (lower_bound - self.boundary_safety_margin).clamp(
            min=0.0, max=self.safe_radius_max
        )
