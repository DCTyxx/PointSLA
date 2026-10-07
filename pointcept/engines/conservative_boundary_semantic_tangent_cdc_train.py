"""Conservative boundary-radius diagnostics for Step-2 reference selection."""

import torch
import torch.nn.functional as F

from .train import TRAINERS
from .cdc_base_train import CDCBaseGeometryTrainer
from .boundary_semantic_tangent_cdc_train import (
    BoundarySemanticTangentCDCTrainer,
)


@TRAINERS.register_module("ConservativeBoundarySemanticTangentCDCTrainer")
class ConservativeBoundarySemanticTangentCDCTrainer(
    BoundarySemanticTangentCDCTrainer
):
    """Reports strict and robust conservative Base-val radius references."""

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

        generator = torch.Generator(device="cpu")
        generator.manual_seed(20261008)
        coeff = torch.randn(
            num_directions,
            int(model.cdc_rank),
            generator=generator,
            dtype=torch.float32,
        ).to(features.device)
        coeff = F.normalize(coeff, dim=-1, eps=1e-6)

        pred_values = []
        oracle_values = []
        pred_valid_values = []
        oracle_valid_values = []

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

            pred_values.append(pred_radius.detach().cpu())
            oracle_values.append(oracle_radius.detach().cpu())
            pred_valid_values.append(pred_valid.detach().float().cpu())
            oracle_valid_values.append(oracle_valid.detach().float().cpu())

        if not oracle_values:
            return None

        pred = torch.cat(pred_values).float()
        oracle = torch.cat(oracle_values).float()
        pred_valid = torch.cat(pred_valid_values).float()
        oracle_valid = torch.cat(oracle_valid_values).float()

        positive = oracle[oracle > 0.0]
        if positive.numel() > 0:
            positive_p01 = float(torch.quantile(positive, 0.01).item())
            positive_p05 = float(torch.quantile(positive, 0.05).item())
            positive_p10 = float(torch.quantile(positive, 0.10).item())
        else:
            positive_p01 = positive_p05 = positive_p10 = 0.0

        q_ref = float(getattr(model, "step2_reference_quantile", 0.05))
        step2_reference = (
            float(torch.quantile(positive, q_ref).item())
            if positive.numel() > 0
            else 0.0
        )

        return dict(
            psvr=float(pred_valid.mean().item()),
            oracle_psvr=float(oracle_valid.mean().item()),
            msr=float(pred.mean().item()),
            oracle_msr=float(oracle.mean().item()),
            radius_mae=float((pred - oracle).abs().mean().item()),
            radius_ratio=float(pred.mean().item() / max(oracle.mean().item(), 1e-12)),
            oracle_min=float(oracle.min().item()),
            oracle_p01=float(torch.quantile(oracle, 0.01).item()),
            oracle_p05=float(torch.quantile(oracle, 0.05).item()),
            oracle_p10=float(torch.quantile(oracle, 0.10).item()),
            positive_oracle_p01=positive_p01,
            positive_oracle_p05=positive_p05,
            positive_oracle_p10=positive_p10,
            zero_radius_rate=float((oracle <= 0.0).float().mean().item()),
            expandable_rate=float((oracle > 0.0).float().mean().item()),
            step2_reference_radius=step2_reference,
        )

    @torch.no_grad()
    def _evaluate_base_geometry(self):
        # Keep TAS/GDG/SVR exactly on the original held-out Base protocol.
        tas, gdg, svr = CDCBaseGeometryTrainer._evaluate_base_geometry(self)
        metrics = getattr(self, "_last_boundary_radius_metrics", None)
        if metrics is not None:
            model = self._unwrap_model()
            utilization = metrics["msr"] / max(float(model.safe_radius_max), 1e-12)
            oracle_utilization = metrics["oracle_msr"] / max(
                float(model.safe_radius_max), 1e-12
            )
            self.logger.info(
                "Conservative boundary radius eval (BASE-ONLY) | "
                f"PSVR: {metrics['psvr']:.4f} | "
                f"OraclePSVR: {metrics['oracle_psvr']:.4f} | "
                f"MSR: {metrics['msr']:.4f} | "
                f"OracleMSR: {metrics['oracle_msr']:.4f} | "
                f"RadiusMAE: {metrics['radius_mae']:.4f} | "
                f"RadiusRatio: {metrics['radius_ratio']:.4f} | "
                f"ZeroRate: {metrics['zero_radius_rate']:.4f} | "
                f"ExpandableRate: {metrics['expandable_rate']:.4f}"
            )
            self.logger.info(
                "Step-2 conservative radius reference | "
                f"OracleMin: {metrics['oracle_min']:.4f} | "
                f"OracleP01: {metrics['oracle_p01']:.4f} | "
                f"OracleP05: {metrics['oracle_p05']:.4f} | "
                f"OracleP10: {metrics['oracle_p10']:.4f} | "
                f"PositiveP01: {metrics['positive_oracle_p01']:.4f} | "
                f"PositiveP05: {metrics['positive_oracle_p05']:.4f} | "
                f"PositiveP10: {metrics['positive_oracle_p10']:.4f} | "
                f"RecommendedRef(q={getattr(model, 'step2_reference_quantile', 0.05):.2f}): "
                f"{metrics['step2_reference_radius']:.4f}"
            )
            self.logger.info(
                "Radius utilization | "
                f"Predicted: {utilization:.4f} | Oracle: {oracle_utilization:.4f}"
            )

            if self.writer is not None:
                step = self.epoch + 1
                for name, value in metrics.items():
                    self.writer.add_scalar(
                        f"conservative_radius/{name}", float(value), step
                    )
                self.writer.add_scalar(
                    "conservative_radius/RadiusUtil", utilization, step
                )
                self.writer.add_scalar(
                    "conservative_radius/OracleUtil", oracle_utilization, step
                )
        return tas, gdg, svr
