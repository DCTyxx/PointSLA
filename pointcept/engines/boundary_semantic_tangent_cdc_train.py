"""Trainer diagnostics for boundary-supervised Semantic Tangent-CDC."""

import torch
import torch.nn.functional as F

from .train import TRAINERS
from .cdc_base_train import CDCBaseGeometryTrainer


@TRAINERS.register_module("BoundarySemanticTangentCDCTrainer")
class BoundarySemanticTangentCDCTrainer(CDCBaseGeometryTrainer):
    """Base-only trainer with predicted/oracle semantic-radius diagnostics."""

    def before_epoch(self):
        super().before_epoch()
        model = self._unwrap_model()
        if self.epoch == getattr(model, "safe_radius_warmup_epochs", -1):
            self.logger.info(
                "=> Boundary-supervised semantic radius learning is now enabled "
                f"(epoch {self.epoch + 1})."
            )

    @torch.no_grad()
    def _compute_boundary_radius_metrics(self, model, features, labels, eval_cfg):
        required = (
            hasattr(model, "get_tangent_basis")
            and hasattr(model, "predict_safe_radius")
            and hasattr(model, "oracle_safe_radius")
            and hasattr(model, "_semantic_safe_at_radius")
        )
        if not required:
            return None

        num_anchors = int(eval_cfg.get("num_anchors", 256))
        num_directions = int(eval_cfg.get("safe_radius_directions", 8))
        anchors = self._balanced_anchor_indices(labels, num_anchors)
        if anchors.numel() == 0:
            return None

        # Fixed tangent coefficients make radius evaluation deterministic across
        # epochs while the learned tangent basis itself is allowed to change.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20261008)
        coeff = torch.randn(
            num_directions,
            int(model.cdc_rank),
            generator=generator,
            dtype=torch.float32,
        ).to(features.device)
        coeff = F.normalize(coeff, dim=-1, eps=1e-6)

        pred_valid_sum = 0.0
        oracle_valid_sum = 0.0
        pred_radius_sum = 0.0
        oracle_radius_sum = 0.0
        abs_error_sum = 0.0
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

            pred_radius = model.predict_safe_radius(
                expanded_z,
                signed_direction,
                labels=expanded_y,
            )
            oracle_radius = model.oracle_safe_radius(
                expanded_z,
                signed_direction,
                expanded_y,
            )

            pred_valid = model._semantic_safe_at_radius(
                expanded_z,
                signed_direction,
                expanded_y,
                pred_radius,
            )
            oracle_valid = model._semantic_safe_at_radius(
                expanded_z,
                signed_direction,
                expanded_y,
                oracle_radius,
            )

            pred_valid_sum += float(pred_valid.float().sum().item())
            oracle_valid_sum += float(oracle_valid.float().sum().item())
            pred_radius_sum += float(pred_radius.sum().item())
            oracle_radius_sum += float(oracle_radius.sum().item())
            abs_error_sum += float((pred_radius - oracle_radius).abs().sum().item())
            total += int(pred_radius.numel())

        if total == 0:
            return None

        psvr = pred_valid_sum / total
        oracle_psvr = oracle_valid_sum / total
        msr = pred_radius_sum / total
        oracle_msr = oracle_radius_sum / total
        radius_mae = abs_error_sum / total
        radius_ratio = msr / max(oracle_msr, 1e-12)
        return dict(
            psvr=psvr,
            oracle_psvr=oracle_psvr,
            msr=msr,
            oracle_msr=oracle_msr,
            radius_mae=radius_mae,
            radius_ratio=radius_ratio,
        )

    @torch.no_grad()
    def _compute_geometry_metrics(self, model, features, labels, eval_cfg):
        result = super()._compute_geometry_metrics(model, features, labels, eval_cfg)
        self._last_boundary_radius_metrics = self._compute_boundary_radius_metrics(
            model, features, labels, eval_cfg
        )
        return result

    @torch.no_grad()
    def _evaluate_base_geometry(self):
        # Explicitly call the CDC base implementation so TAS/GDG/SVR stay exactly
        # comparable to previous Tangent-CDC runs. Dynamic dispatch above still
        # records boundary-radius diagnostics on the same held-out feature bank.
        tas, gdg, svr = CDCBaseGeometryTrainer._evaluate_base_geometry(self)
        metrics = getattr(self, "_last_boundary_radius_metrics", None)
        if metrics is not None:
            model = self._unwrap_model()
            utilization = metrics["msr"] / max(float(model.safe_radius_max), 1e-12)
            oracle_utilization = metrics["oracle_msr"] / max(
                float(model.safe_radius_max), 1e-12
            )
            self.logger.info(
                "Boundary radius eval (BASE-ONLY) | "
                f"PSVR: {metrics['psvr']:.4f} | "
                f"OraclePSVR: {metrics['oracle_psvr']:.4f} | "
                f"MSR: {metrics['msr']:.4f} | "
                f"OracleMSR: {metrics['oracle_msr']:.4f} | "
                f"RadiusMAE: {metrics['radius_mae']:.4f} | "
                f"RadiusRatio: {metrics['radius_ratio']:.4f} | "
                f"RadiusUtil: {utilization:.4f} | "
                f"OracleUtil: {oracle_utilization:.4f}"
            )
            if self.writer is not None:
                step = self.epoch + 1
                for name, value in metrics.items():
                    self.writer.add_scalar(
                        f"boundary_radius/{name}", float(value), step
                    )
                self.writer.add_scalar(
                    "boundary_radius/RadiusUtil", utilization, step
                )
                self.writer.add_scalar(
                    "boundary_radius/OracleUtil", oracle_utilization, step
                )
        return tas, gdg, svr
