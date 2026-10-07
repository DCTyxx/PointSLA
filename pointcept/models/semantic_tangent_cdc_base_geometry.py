"""Joint Tangent-CDC + semantic safe-radius learning for Step-1."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.utils.structure import Point
from .builder import MODELS
from .tangent_cdc_base_geometry import TangentCDCBaseGeometrySegmentor


class SemanticSafeRadiusHead(nn.Module):
    def __init__(
        self,
        feature_dim,
        text_dim,
        hidden_dim=128,
        max_radius=0.15,
        initial_radius=0.05,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.text_dim = int(text_dim)
        self.max_radius = float(max_radius)
        initial_radius = float(initial_radius)
        if not (0.0 < initial_radius < self.max_radius):
            raise ValueError("initial_radius must be in (0, max_radius)")

        self.feature_proj = nn.Linear(self.feature_dim, hidden_dim)
        self.direction_proj = nn.Linear(self.feature_dim, hidden_dim)
        self.semantic_proj = nn.Linear(self.text_dim, hidden_dim)
        self.text_proj = nn.Linear(self.text_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.radius_out = nn.Linear(hidden_dim, 1)

        nn.init.zeros_(self.radius_out.weight)
        p = initial_radius / self.max_radius
        nn.init.constant_(self.radius_out.bias, math.log(p / (1.0 - p)))

    def forward(self, feature, direction, semantic_embedding, text_embedding):
        feature = F.layer_norm(feature, (self.feature_dim,))
        direction = F.normalize(direction, dim=-1, eps=1e-6)
        semantic_embedding = F.normalize(semantic_embedding, dim=-1, eps=1e-6)
        text_embedding = F.normalize(text_embedding, dim=-1, eps=1e-6)

        h = torch.cat(
            [
                F.gelu(self.feature_proj(feature)),
                F.gelu(self.direction_proj(direction)),
                F.gelu(self.semantic_proj(semantic_embedding)),
                F.gelu(self.text_proj(text_embedding)),
            ],
            dim=-1,
        )
        logit = self.radius_out(self.fusion(h)).squeeze(-1)
        return self.max_radius * torch.sigmoid(logit)


@MODELS.register_module()
class SemanticTangentCDCBaseGeometrySegmentor(TangentCDCBaseGeometrySegmentor):
    """Frozen PTv3 + orthonormal Tangent-CDC + semantic safe radius.

    Gradient routing is controlled by construction:
    - L_tan/L_disc -> CDC only;
    - L_align -> semantic projector only;
    - L_safe/L_expand -> safe-radius head only.
    """

    def __init__(self, safe_radius=None, **kwargs):
        super().__init__(**kwargs)
        safe_radius = {} if safe_radius is None else dict(safe_radius)

        self.safe_radius_max = float(safe_radius.get("max_radius", 0.15))
        self.safe_radius_num_directions = int(
            safe_radius.get("num_directions", 4)
        )
        self.safe_radius_path_fractions = tuple(
            float(v) for v in safe_radius.get("path_fractions", (0.5, 1.0))
        )
        self.safe_radius_loss_weight = float(
            safe_radius.get("safe_loss_weight", 1.0)
        )
        self.safe_radius_expansion_weight = float(
            safe_radius.get("expansion_loss_weight", 0.01)
        )
        self.safe_radius_warmup_epochs = int(
            safe_radius.get("warmup_epochs", 1)
        )
        self.safe_radius_qr_eps = float(safe_radius.get("qr_eps", 1e-6))

        if self.safe_radius_num_directions <= 0:
            raise ValueError("safe_radius.num_directions must be positive")
        if (
            not self.safe_radius_path_fractions
            or any(v <= 0.0 or v > 1.0 for v in self.safe_radius_path_fractions)
        ):
            raise ValueError("safe_radius.path_fractions must lie in (0, 1]")
        if self.safe_radius_max <= 0.0:
            raise ValueError("safe_radius.max_radius must be positive")

        self.safe_radius_head = SemanticSafeRadiusHead(
            feature_dim=self.feature_dim,
            text_dim=self.text_dim,
            hidden_dim=int(safe_radius.get("hidden_dim", 128)),
            max_radius=self.safe_radius_max,
            initial_radius=float(safe_radius.get("initial_radius", 0.05)),
        )

    def _orthonormalize_factor(self, factor):
        normalized = factor / factor.norm(dim=1, keepdim=True).clamp_min(
            self.safe_radius_qr_eps
        )
        q, _ = torch.linalg.qr(normalized, mode="reduced")
        return q

    @staticmethod
    def _orthonormal_projected_energy_ratio(q, residual):
        coeff = torch.einsum("akd,adr->akr", residual, q)
        projected_energy = coeff.square().sum(dim=(1, 2))
        total_energy = residual.square().sum(dim=(1, 2)).clamp_min(1e-12)
        return projected_energy / total_energy

    def _tangent_objective_qr(self, clean, labels):
        bank_idx = self._balanced_unique_indices(
            labels, self.tangent_max_bank_points
        )
        bank = clean[bank_idx]
        bank_labels = labels[bank_idx]

        anchor_idx = self._balanced_tangent_anchors(
            bank_labels,
            self.tangent_max_anchors,
            self.tangent_knn + 1,
        )
        if anchor_idx.numel() == 0:
            raise RuntimeError("No class has enough points for Tangent-CDC KNN")

        anchors = bank[anchor_idx]
        anchor_labels = bank_labels[anchor_idx]
        num_anchors = anchors.shape[0]

        with torch.no_grad():
            dist = torch.cdist(anchors, bank, p=2).square()

            same = anchor_labels[:, None].eq(bank_labels[None, :])
            row = torch.arange(num_anchors, device=clean.device)
            same[row, anchor_idx] = False
            nearest_intra = torch.topk(
                dist.masked_fill(~same, float("inf")),
                k=self.tangent_knn,
                dim=1,
                largest=False,
            ).indices
            intra_residual = bank[nearest_intra] - anchors[:, None, :]

            _, _, vh = torch.linalg.svd(
                intra_residual.detach(), full_matrices=False
            )
            q_emp = vh[:, : self.cdc_rank, :].transpose(1, 2).contiguous()

            different = ~anchor_labels[:, None].eq(bank_labels[None, :])
            min_different = int(different.sum(dim=1).min().item())
            if min_different <= 0:
                raise RuntimeError("At least two Base classes are required")
            k_inter = min(self.tangent_inter_knn, min_different)
            nearest_inter = torch.topk(
                dist.masked_fill(~different, float("inf")),
                k=k_inter,
                dim=1,
                largest=False,
            ).indices
            inter_residual = bank[nearest_inter] - anchors[:, None, :]

        relative_h = torch.full(
            (num_anchors,),
            self.cdc_probe_h,
            device=clean.device,
            dtype=clean.dtype,
        )
        feature_rms = anchors.square().mean(dim=-1).add(1e-8).sqrt()
        factor = self.cdc_head(anchors, relative_h * feature_rms)
        q = self._orthonormalize_factor(factor)

        overlap = torch.einsum("adr,adq->arq", q, q_emp)
        alignment = overlap.square().sum(dim=(1, 2)) / self.cdc_rank
        tangent_loss = 1.0 - alignment.mean()

        intra_evr = self._orthonormal_projected_energy_ratio(
            q, intra_residual.detach()
        )
        inter_evr = self._orthonormal_projected_energy_ratio(
            q, inter_residual.detach()
        )
        gap = intra_evr - inter_evr
        disc_loss = F.relu(self.tangent_disc_margin - gap).mean()

        return dict(
            tangent_loss=tangent_loss,
            disc_loss=disc_loss,
            train_tas=alignment.mean().detach(),
            train_gdg=gap.mean().detach(),
            factor=factor,
            q=q,
            anchors=anchors,
            anchor_labels=anchor_labels,
        )

    def _sample_tangent_directions(self, q):
        latent = torch.randn(
            q.shape[0],
            self.safe_radius_num_directions,
            self.cdc_rank,
            device=q.device,
            dtype=q.dtype,
        )
        latent = F.normalize(latent, dim=-1, eps=1e-6)
        direction = torch.einsum("adr,amr->amd", q.detach(), latent)
        return F.normalize(direction, dim=-1, eps=1e-6)

    def _safe_radius_objective(self, anchors, labels, q):
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

        with torch.no_grad():
            clean_semantic = self.semantic_projector(
                clean, detach_parameters=True
            )
            class_text = self.text_embeddings.float()[flat_labels]
            clean_scores = self._text_scores(
                clean, detach_projector=True
            )
            clean_gt, clean_margin = self._gt_and_margin(
                clean_scores, flat_labels
            )

        rho_pos = self.safe_radius_head(
            clean, direction, clean_semantic, class_text
        )
        rho_neg = self.safe_radius_head(
            clean, -direction, clean_semantic, class_text
        )
        feature_norm = clean.norm(dim=-1).clamp_min(1e-6)

        safety_losses = []
        endpoint_valid = []
        for signed_direction, rho in (
            (direction, rho_pos),
            (-direction, rho_neg),
        ):
            for fraction in self.safe_radius_path_fractions:
                virtual = (
                    clean
                    + float(fraction)
                    * rho.unsqueeze(-1)
                    * feature_norm.unsqueeze(-1)
                    * signed_direction
                )
                virtual_scores = self._text_scores(
                    virtual, detach_projector=True
                )
                virtual_gt, virtual_margin = self._gt_and_margin(
                    virtual_scores, flat_labels
                )
                positive_loss = F.relu(
                    clean_gt.detach()
                    - self.text_positive_tolerance
                    - virtual_gt
                )
                margin_loss = F.relu(
                    clean_margin.detach()
                    - self.text_margin_tolerance
                    - virtual_margin
                )
                safety_losses.append(positive_loss + margin_loss)

                if abs(float(fraction) - 1.0) < 1e-6:
                    endpoint_valid.append(
                        (
                            (
                                virtual_gt
                                >= clean_gt.detach()
                                - self.text_positive_tolerance
                            )
                            & (
                                virtual_margin
                                >= clean_margin.detach()
                                - self.text_margin_tolerance
                            )
                        )
                        .float()
                        .detach()
                    )

        safe_loss = torch.cat(safety_losses, dim=0).mean()
        mean_radius = 0.5 * (rho_pos.mean() + rho_neg.mean())
        expansion_loss = 1.0 - mean_radius / self.safe_radius_max
        predicted_svr = (
            torch.cat(endpoint_valid, dim=0).mean()
            if endpoint_valid
            else clean.new_zeros(())
        )
        return safe_loss, expansion_loss, predicted_svr, mean_radius

    @torch.no_grad()
    def get_tangent_basis(self, feature, relative_h=None):
        if relative_h is None:
            relative_h = self.cdc_probe_h
        feature = feature.float()
        relative = torch.full(
            (feature.shape[0],),
            float(relative_h),
            device=feature.device,
            dtype=feature.dtype,
        )
        feature_rms = feature.square().mean(dim=-1).add(1e-8).sqrt()
        factor = self.cdc_head(feature, relative * feature_rms)
        return self._orthonormalize_factor(factor)

    @torch.no_grad()
    def predict_safe_radius(
        self,
        feature,
        direction,
        labels=None,
        text_embedding=None,
    ):
        feature = feature.float()
        direction = F.normalize(direction.float(), dim=-1, eps=1e-6)
        semantic = self.semantic_projector(
            feature, detach_parameters=True
        )

        if text_embedding is None:
            if labels is None:
                raise ValueError("Provide labels or text_embedding")
            text_embedding = self.text_embeddings.float()[labels.long()]
        else:
            text_embedding = text_embedding.to(
                device=feature.device, dtype=feature.dtype
            )
            if text_embedding.ndim == 1:
                text_embedding = text_embedding.unsqueeze(0)
            if text_embedding.shape[0] == 1 and feature.shape[0] > 1:
                text_embedding = text_embedding.expand(feature.shape[0], -1)

        return self.safe_radius_head(
            feature, direction, semantic, text_embedding
        )

    def forward(self, input_dict):
        if not bool(self._text_ready.item()):
            raise RuntimeError(
                "Frozen Base text embeddings are not initialized"
            )

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

            text_idx = self._balanced_unique_indices(
                labels, self.text_max_points
            )
            text_align_loss, text_acc = self._semantic_alignment_loss(
                clean[text_idx], labels[text_idx]
            )

            radius_active = (
                self.current_epoch >= self.safe_radius_warmup_epochs
            )
            if radius_active:
                (
                    safe_loss,
                    expansion_loss,
                    predicted_svr,
                    mean_safe_radius,
                ) = self._safe_radius_objective(
                    geometry["anchors"],
                    geometry["anchor_labels"],
                    geometry["q"],
                )
                radius_loss = (
                    self.safe_radius_loss_weight * safe_loss
                    + self.safe_radius_expansion_weight * expansion_loss
                )
            else:
                zero = clean.new_zeros(())
                safe_loss = zero
                expansion_loss = zero
                predicted_svr = zero
                mean_safe_radius = zero
                radius_loss = zero

            loss = (
                self.tangent_loss_weight * geometry["tangent_loss"]
                + self.tangent_disc_weight * geometry["disc_loss"]
                + self.text_align_weight * text_align_loss
                + radius_loss
            )

            factor_energy = (
                geometry["factor"].square().sum(dim=(1, 2))
                / self.feature_dim
            ).mean()
            qtq = torch.einsum(
                "adr,adq->arq", geometry["q"], geometry["q"]
            )
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
            "safe_radius_loss": safe_loss.detach(),
            "radius_expansion_loss": expansion_loss.detach(),
            "predicted_svr": predicted_svr.detach(),
            "mean_safe_radius": mean_safe_radius.detach(),
            "radius_active": clean.new_tensor(float(radius_active)),
            "basis_orth_error": basis_orth_error.detach(),
            "cdc_factor_energy": factor_energy.detach(),
        }
