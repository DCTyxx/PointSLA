"""Trainer extensions for Semantic Tangent-CDC safe-radius validation."""

import torch
import torch.nn.functional as F

from .train import TRAINERS
from .cdc_base_train import CDCBaseGeometryTrainer


@TRAINERS.register_module("SemanticTangentCDCTrainer")
class SemanticTangentCDCTrainer(CDCBaseGeometryTrainer):
    """Base-only trainer with PSVR/MSR held-out diagnostics."""

    def before_epoch(self):
        super().before_epoch()
        model = self._unwrap_model()
        if self.epoch == getattr(model, "safe_radius_warmup_epochs", -1):
            self.logger.info(
                "=> Semantic safe-radius learning is now enabled "
                f"(epoch {self.epoch + 1})."
            )

    @torch.no_grad()
    def _compute_predicted_radius_metrics(self, model, features, labels, eval_cfg):
        if not (
            hasattr(model, "get_tangent_basis")
            and hasattr(model, "predict_safe_radius")
        ):
            return None

        num_anchors = int(eval_cfg.get("num_anchors", 256))
        num_directions = int(eval_cfg.get("safe_radius_directions", 8))
        anchors = self._balanced_anchor_indices(labels, num_anchors)
        if anchors.numel() == 0:
            return None

        generator = torch.Generator(device="cpu")
        generator.manual_seed(20261008)
        coeff = torch.randn(
            num_directions,
            int(model.cdc_rank),
            generator=generator,
            dtype=torch.float32,
        ).to(features.device)
        coeff = F.normalize(coeff, dim=-1, eps=1e-6)

        valid_sum = 0.0
        radius_sum = 0.0
        total = 0

        for anchor_idx in anchors.tolist():
            z = features[anchor_idx]
            y = labels[anchor_idx]
            q = model.get_tangent_basis(
                z.unsqueeze(0), relative_h=model.cdc_probe_h
            )[0]
            direction = (q @ coeff.transpose(0, 1)).transpose(0, 1)
            direction = F.normalize(direction, dim=-1, eps=1e-6)
            signed_direction = torch.cat([direction, -direction], dim=0)

            expanded_z = z.unsqueeze(0).expand(signed_direction.shape[0], -1)
            expanded_y = y.view(1).expand(signed_direction.shape[0])
            rho = model.predict_safe_radius(
                expanded_z,
                signed_direction,
                labels=expanded_y,
            )
            virtual = (
                expanded_z
                + rho.unsqueeze(-1)
                * z.norm().clamp_min(1e-6)
                * signed_direction
            )

            clean_scores = model._text_scores(
                z.unsqueeze(0), detach_projector=True
            )
            clean_gt, clean_margin = model._gt_and_margin(
                clean_scores, y.view(1)
            )
            virtual_scores = model._text_scores(
                virtual, detach_projector=True
            )
            virtual_gt, virtual_margin = model._gt_and_margin(
                virtual_scores, expanded_y
            )
            valid = (
                (virtual_gt >= clean_gt - model.text_positive_tolerance)
                & (
                    virtual_margin
                    >= clean_margin - model.text_margin_tolerance
                )
            )

            valid_sum += float(valid.float().sum().item())
            radius_sum += float(rho.sum().item())
            total += int(rho.numel())

        if total == 0:
            return None
        return valid_sum / total, radius_sum / total

    @torch.no_grad()
    def _compute_geometry_metrics(self, model, features, labels, eval_cfg):
        result = super()._compute_geometry_metrics(
            model, features, labels, eval_cfg
        )
        self._last_radius_metrics = self._compute_predicted_radius_metrics(
            model, features, labels, eval_cfg
        )
        return result

    @torch.no_grad()
    def _evaluate_base_geometry(self):
        tas, gdg, svr = super()._evaluate_base_geometry()
        metrics = getattr(self, "_last_radius_metrics", None)
        if metrics is not None:
            psvr, msr = metrics
            model = self._unwrap_model()
            utilization = msr / max(float(model.safe_radius_max), 1e-12)
            self.logger.info(
                "Semantic radius eval (BASE-ONLY) | "
                f"PSVR: {psvr:.4f} | MSR: {msr:.4f} | "
                f"RadiusUtil: {utilization:.4f}"
            )
            if self.writer is not None:
                step = self.epoch + 1
                self.writer.add_scalar("semantic_radius/PSVR", psvr, step)
                self.writer.add_scalar("semantic_radius/MSR", msr, step)
                self.writer.add_scalar(
                    "semantic_radius/RadiusUtil", utilization, step
                )
        return tas, gdg, svr
