"""Boundary-supervised Semantic Tangent-CDC for Step-1.

This variant keeps the successful QR Tangent-CDC geometry objective and replaces
indirect ``semantic hinge + expansion reward`` radius learning with explicit
semantic-boundary supervision. For each sampled tangent direction, a frozen
Base-only semantic evaluator searches the largest safe radius rho*; the radius
head directly regresses rho*. Semantic boundary targets are detached, so:

- L_tan / L_disc update CDC geometry only;
- L_align updates the PTv3->CLIP semantic projector only;
- L_radius updates the safe-radius head only.

No novel labels, class names, support/query samples, or registration data are
used in Step-1.
"""

import torch
import torch.nn.functional as F

from pointcept.models.utils.structure import Point
from .builder import MODELS
from .semantic_tangent_cdc_base_geometry import (
    SemanticTangentCDCBaseGeometrySegmentor,
)


@MODELS.register_module()
class BoundarySemanticTangentCDCBaseGeometrySegmentor(
    SemanticTangentCDCBaseGeometrySegmentor
):
    """QR Tangent-CDC + explicit semantic-boundary radius regression."""

    def __init__(self, safe_radius=None, **kwargs):
        safe_radius = {} if safe_radius is None else dict(safe_radius)
        super().__init__(safe_radius=safe_radius, **kwargs)

        # Coarse search radii. The target is the largest prefix-safe radius.
        default_grid = (0.025, 0.050, 0.075, 0.100, 0.125, 0.150)
        self.boundary_grid = tuple(
            float(v) for v in safe_radius.get("boundary_grid", default_grid)
        )
        self.boundary_refine_steps = int(
            safe_radius.get("boundary_refine_steps", 2)
        )
        self.boundary_loss_weight = float(
            safe_radius.get("boundary_loss_weight", 1.0)
        )
        self.boundary_overshoot_weight = float(
            safe_radius.get("overshoot_loss_weight", 0.5)
        )
        self.boundary_smooth_l1_beta = float(
            safe_radius.get("smooth_l1_beta", 0.1)
        )

        if not self.boundary_grid:
            raise ValueError("safe_radius.boundary_grid must be non-empty")
        if any(v <= 0.0 or v > self.safe_radius_max for v in self.boundary_grid):
            raise ValueError(
                "Every boundary_grid radius must lie in (0, max_radius]"
            )
        if any(
            self.boundary_grid[i] >= self.boundary_grid[i + 1]
            for i in range(len(self.boundary_grid) - 1)
        ):
            raise ValueError("safe_radius.boundary_grid must be strictly increasing")
        if abs(self.boundary_grid[-1] - self.safe_radius_max) > 1e-6:
            raise ValueError(
                "The last boundary_grid value must equal safe_radius.max_radius"
            )
        if self.boundary_refine_steps < 0:
            raise ValueError("boundary_refine_steps must be >= 0")
        if self.boundary_smooth_l1_beta <= 0.0:
            raise ValueError("smooth_l1_beta must be positive")

    @torch.no_grad()
    def _semantic_safe_at_radius(
        self,
        clean,
        direction,
        labels,
        radius,
        clean_gt=None,
        clean_margin=None,
    ):
        """Hard semantic validity for a proposed radius, including path probes."""
        if clean_gt is None or clean_margin is None:
            clean_scores = self._text_scores(clean, detach_projector=True)
            clean_gt, clean_margin = self._gt_and_margin(clean_scores, labels)

        feature_norm = clean.norm(dim=-1).clamp_min(1e-6)
        valid = torch.ones(clean.shape[0], device=clean.device, dtype=torch.bool)
        for fraction in self.safe_radius_path_fractions:
            virtual = (
                clean
                + float(fraction)
                * radius.unsqueeze(-1)
                * feature_norm.unsqueeze(-1)
                * direction
            )
            scores = self._text_scores(virtual, detach_projector=True)
            virtual_gt, virtual_margin = self._gt_and_margin(scores, labels)
            valid = valid & (
                virtual_gt >= clean_gt - self.text_positive_tolerance
            ) & (
                virtual_margin >= clean_margin - self.text_margin_tolerance
            )
        return valid

    @torch.no_grad()
    def oracle_safe_radius(self, feature, direction, labels):
        """Search the largest prefix-safe semantic radius rho*.

        Search is Base-semantic only. A coarse monotone-prefix scan is followed
        by optional binary refinement inside the first unsafe interval. This
        avoids assuming that isolated far-away safe points define a valid path.
        """
        clean = feature.float()
        direction = F.normalize(direction.float(), dim=-1, eps=1e-6)
        labels = labels.long()

        clean_scores = self._text_scores(clean, detach_projector=True)
        clean_gt, clean_margin = self._gt_and_margin(clean_scores, labels)

        grid = clean.new_tensor(self.boundary_grid)
        n = clean.shape[0]
        prefix_alive = torch.ones(n, device=clean.device, dtype=torch.bool)
        low = clean.new_zeros(n)
        high = clean.new_full((n,), self.safe_radius_max)
        first_failure_found = torch.zeros(n, device=clean.device, dtype=torch.bool)

        for radius_value in grid:
            radius = torch.full_like(low, float(radius_value.item()))
            valid = self._semantic_safe_at_radius(
                clean,
                direction,
                labels,
                radius,
                clean_gt=clean_gt,
                clean_margin=clean_margin,
            )
            prefix_valid = prefix_alive & valid
            low = torch.where(prefix_valid, radius, low)

            newly_failed = prefix_alive & (~valid) & (~first_failure_found)
            high = torch.where(newly_failed, radius, high)
            first_failure_found = first_failure_found | newly_failed
            prefix_alive = prefix_valid

        # Samples safe at every coarse radius already have rho*=rho_max.
        refine_mask = first_failure_found
        for _ in range(self.boundary_refine_steps):
            if not bool(refine_mask.any().item()):
                break
            mid = 0.5 * (low + high)
            valid_mid = self._semantic_safe_at_radius(
                clean,
                direction,
                labels,
                mid,
                clean_gt=clean_gt,
                clean_margin=clean_margin,
            )
            accept = refine_mask & valid_mid
            reject = refine_mask & (~valid_mid)
            low = torch.where(accept, mid, low)
            high = torch.where(reject, mid, high)

        return low.clamp(min=0.0, max=self.safe_radius_max)

    def _boundary_radius_objective(self, anchors, labels, q):
        """Regress the explicit semantic-boundary target for sampled tangents."""
        directions = self._sample_tangent_directions(q)
        num_anchor, num_dir, dim = directions.shape

        clean = (
            anchors[:, None, :]
            .expand(num_anchor, num_dir, dim)
            .reshape(-1, dim)
            .detach()
        )
        flat_labels = (
            labels[:, None].expand(num_anchor, num_dir).reshape(-1).detach()
        )
        direction = directions.reshape(-1, dim).detach()

        # Treat +d and -d as separate directional queries. This lets one shared
        # radius head learn asymmetric semantic boundaries without basis-specific
        # outputs.
        signed_clean = torch.cat([clean, clean], dim=0)
        signed_labels = torch.cat([flat_labels, flat_labels], dim=0)
        signed_direction = torch.cat([direction, -direction], dim=0)

        with torch.no_grad():
            clean_semantic = self.semantic_projector(
                signed_clean, detach_parameters=True
            )
            class_text = self.text_embeddings.float()[signed_labels]
            oracle_radius = self.oracle_safe_radius(
                signed_clean,
                signed_direction,
                signed_labels,
            )

        pred_radius = self.safe_radius_head(
            signed_clean,
            signed_direction,
            clean_semantic,
            class_text,
        )

        pred_norm = pred_radius / self.safe_radius_max
        target_norm = oracle_radius / self.safe_radius_max
        regression_loss = F.smooth_l1_loss(
            pred_norm,
            target_norm,
            beta=self.boundary_smooth_l1_beta,
        )
        # Regression is the main target. A modest asymmetric overshoot penalty
        # makes errors conservative because overshooting rho* directly hurts PSVR.
        overshoot_loss = F.relu(pred_norm - target_norm).mean()
        radius_loss = (
            self.boundary_loss_weight * regression_loss
            + self.boundary_overshoot_weight * overshoot_loss
        )

        with torch.no_grad():
            predicted_valid = self._semantic_safe_at_radius(
                signed_clean,
                signed_direction,
                signed_labels,
                pred_radius.detach(),
            )
            oracle_valid = self._semantic_safe_at_radius(
                signed_clean,
                signed_direction,
                signed_labels,
                oracle_radius,
            )
            radius_mae = (pred_radius.detach() - oracle_radius).abs().mean()
            mean_pred = pred_radius.detach().mean()
            mean_oracle = oracle_radius.mean()
            radius_ratio = mean_pred / mean_oracle.clamp_min(1e-6)

        return dict(
            radius_loss=radius_loss,
            regression_loss=regression_loss,
            overshoot_loss=overshoot_loss,
            predicted_svr=predicted_valid.float().mean(),
            oracle_svr=oracle_valid.float().mean(),
            mean_safe_radius=mean_pred,
            oracle_mean_safe_radius=mean_oracle,
            radius_mae=radius_mae,
            radius_ratio=radius_ratio,
        )

    def forward(self, input_dict):
        if not bool(self._text_ready.item()):
            raise RuntimeError("Frozen Base text embeddings are not initialized")

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                point = self.backbone(Point(input_dict))
                feature = point.feat if isinstance(point, Point) else point

        labels = input_dict["segment"].long()
        valid = labels >= 0
        if not valid.any():
            raise RuntimeError("The batch contains no valid Base points")
        if int(labels[valid].max().item()) >= self.num_classes:
            raise RuntimeError("Non-Base label detected in Step-1")

        with torch.cuda.amp.autocast(enabled=False):
            clean = feature.detach().float()
            labels = labels.to(clean.device)

            geometry = self._tangent_objective_qr(clean, labels)

            text_idx = self._balanced_unique_indices(labels, self.text_max_points)
            text_align_loss, text_acc = self._semantic_alignment_loss(
                clean[text_idx], labels[text_idx]
            )

            radius_active = self.current_epoch >= self.safe_radius_warmup_epochs
            if radius_active:
                radius = self._boundary_radius_objective(
                    geometry["anchors"],
                    geometry["anchor_labels"],
                    geometry["q"],
                )
                radius_loss = radius["radius_loss"]
            else:
                zero = clean.new_zeros(())
                radius = dict(
                    radius_loss=zero,
                    regression_loss=zero,
                    overshoot_loss=zero,
                    predicted_svr=zero,
                    oracle_svr=zero,
                    mean_safe_radius=zero,
                    oracle_mean_safe_radius=zero,
                    radius_mae=zero,
                    radius_ratio=zero,
                )
                # Keep the radius head in the autograd graph during warm-up so
                # DDP remains valid even when find_unused_parameters=False later.
                dummy = next(self.safe_radius_head.parameters()).sum() * 0.0
                radius_loss = dummy

            loss = (
                self.tangent_loss_weight * geometry["tangent_loss"]
                + self.tangent_disc_weight * geometry["disc_loss"]
                + self.text_align_weight * text_align_loss
                + radius_loss
            )

            factor_energy = (
                geometry["factor"].square().sum(dim=(1, 2)) / self.feature_dim
            ).mean()
            qtq = torch.einsum("adr,adq->arq", geometry["q"], geometry["q"])
            eye = torch.eye(
                self.cdc_rank, device=clean.device, dtype=clean.dtype
            )[None]
            basis_orth_error = (qtq - eye).square().mean()

        return {
            "loss": loss,
            "tangent_loss": geometry["tangent_loss"].detach(),
            "discrimination_loss": geometry["disc_loss"].detach(),
            "train_tas": geometry["train_tas"],
            "train_gdg": geometry["train_gdg"],
            "text_align_loss": text_align_loss.detach(),
            "text_acc": text_acc.detach(),
            "radius_boundary_loss": radius["radius_loss"].detach(),
            "radius_regression_loss": radius["regression_loss"].detach(),
            "radius_overshoot_loss": radius["overshoot_loss"].detach(),
            "predicted_svr": radius["predicted_svr"].detach(),
            "oracle_svr": radius["oracle_svr"].detach(),
            "mean_safe_radius": radius["mean_safe_radius"].detach(),
            "oracle_mean_safe_radius": radius["oracle_mean_safe_radius"].detach(),
            "radius_mae": radius["radius_mae"].detach(),
            "radius_ratio": radius["radius_ratio"].detach(),
            "radius_active": clean.new_tensor(float(radius_active)),
            "basis_orth_error": basis_orth_error.detach(),
            "cdc_factor_energy": factor_energy.detach(),
        }
