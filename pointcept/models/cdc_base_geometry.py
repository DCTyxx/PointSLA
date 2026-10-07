"""
CDC base-geometry pretraining for GFS-VL.

Step 1 learns a transferable local geometry prior in the frozen PTv3 feature
space. The CDC objective follows the low-rank conditional metric matching loss,
while frozen CLIP text prototypes constrain learned CDC directions to preserve
class semantics. No novel/few-shot logic is included here.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model


class BandwidthEmbedding(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, bandwidth):
        return self.net(bandwidth.clamp_min(1e-8).log().unsqueeze(-1))


class CDCMetricHead(nn.Module):
    """Predict low-rank CDC factor U(z,h), with Gamma = U U^T."""

    def __init__(self, feature_dim, rank=8, hidden_dim=128, bandwidth_dim=32):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.rank = int(rank)
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        self.bandwidth_embedding = BandwidthEmbedding(bandwidth_dim)
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.feature_dim + bandwidth_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.factor_projection = nn.Linear(hidden_dim, self.feature_dim * self.rank)

    def forward(self, feature, bandwidth):
        x = self.feature_norm(feature)
        h = self.bandwidth_embedding(bandwidth)
        x = self.shared_mlp(torch.cat([x, h], dim=-1))
        factor = self.factor_projection(x)
        return factor.view(-1, self.feature_dim, self.rank)


class SemanticProjector(nn.Module):
    """Project PTv3 features into the frozen CLIP text space."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.in_dim = int(in_dim)
        self.fc1 = nn.Linear(self.in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    @staticmethod
    def _linear_detached(x, layer):
        weight = layer.weight.detach()
        bias = None if layer.bias is None else layer.bias.detach()
        return F.linear(x, weight, bias)

    def forward(self, x, detach_parameters=False):
        x = F.layer_norm(x, (self.in_dim,))
        if detach_parameters:
            x = self._linear_detached(x, self.fc1)
            x = F.gelu(x)
            x = self._linear_detached(x, self.fc2)
        else:
            x = F.gelu(self.fc1(x))
            x = self.fc2(x)
        return F.normalize(x, dim=-1, eps=1e-6)


@MODELS.register_module()
class CDCBaseGeometrySegmentor(nn.Module):
    """Step-1 model: frozen GFS-VL backbone + trainable CDC/text bridge."""

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=True,
        freeze_seg_head=True,
        cdc=None,
        text_constraint=None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature_dim = int(backbone_out_channels)

        # Preserve original names so the official base checkpoint loads directly.
        self.seg_head = (
            nn.Linear(self.feature_dim, self.num_classes)
            if self.num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)

        cdc = {} if cdc is None else dict(cdc)
        text_constraint = {} if text_constraint is None else dict(text_constraint)

        self.cdc_rank = int(cdc.get("rank", 8))
        self.cdc_h_min = float(cdc.get("h_min", 0.01))
        self.cdc_h_max = float(cdc.get("h_max", 0.05))
        self.cdc_sampling_method = cdc.get("sampling_method", "loguniform")
        self.cdc_max_points = int(cdc.get("max_points", 4096))
        self.cdc_tikhonov = float(cdc.get("tikhonov_epsilon", 1e-4))
        self.cdc_loss_weight = float(cdc.get("loss_weight", 1.0))
        self.cdc_probe_h = float(cdc.get("probe_h", 0.02))
        self.cdc_head = CDCMetricHead(
            feature_dim=self.feature_dim,
            rank=self.cdc_rank,
            hidden_dim=int(cdc.get("hidden_dim", 128)),
            bandwidth_dim=int(cdc.get("bandwidth_dim", 32)),
        )

        self.text_dim = int(text_constraint.get("text_dim", 512))
        self.semantic_projector = SemanticProjector(
            in_dim=self.feature_dim,
            hidden_dim=int(text_constraint.get("projector_hidden_dim", 256)),
            out_dim=self.text_dim,
        )
        self.text_logit_scale = float(text_constraint.get("logit_scale", 20.0))
        self.text_align_weight = float(text_constraint.get("align_loss_weight", 1.0))
        self.text_direction_weight = float(
            text_constraint.get("direction_loss_weight", 0.2)
        )
        self.text_positive_tolerance = float(
            text_constraint.get("positive_tolerance", 0.02)
        )
        self.text_margin_tolerance = float(
            text_constraint.get("margin_tolerance", 0.02)
        )
        self.text_probe_radius = float(text_constraint.get("probe_radius", 0.05))
        self.text_num_direction_probes = int(
            text_constraint.get("num_direction_probes", 1)
        )
        self.text_max_points = int(text_constraint.get("max_points", 1024))
        self.text_warmup_epochs = int(text_constraint.get("warmup_epochs", 5))

        self.register_buffer(
            "text_embeddings",
            torch.zeros(self.num_classes, self.text_dim),
            persistent=True,
        )
        self.register_buffer("_text_ready", torch.tensor(False), persistent=False)
        self.current_epoch = 0

        self.freeze_backbone = bool(freeze_backbone)
        self.freeze_seg_head = bool(freeze_seg_head)
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        if self.freeze_seg_head:
            self.seg_head.requires_grad_(False)
            self.seg_head.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        if self.freeze_seg_head:
            self.seg_head.eval()
        return self

    @torch.no_grad()
    def set_text_embeddings(self, text_embeddings):
        text_embeddings = text_embeddings.to(
            device=self.text_embeddings.device,
            dtype=self.text_embeddings.dtype,
        )
        if tuple(text_embeddings.shape) != tuple(self.text_embeddings.shape):
            raise ValueError(
                "Text embedding shape mismatch: expected "
                f"{tuple(self.text_embeddings.shape)}, got {tuple(text_embeddings.shape)}"
            )
        self.text_embeddings.copy_(F.normalize(text_embeddings, dim=-1, eps=1e-6))
        self._text_ready.fill_(True)

    def set_training_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def getFeatures(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        return point.feat if isinstance(point, Point) else point

    def _sample_relative_h(self, count, device, dtype):
        if self.cdc_sampling_method == "uniform":
            return (
                torch.rand(count, device=device, dtype=dtype)
                * (self.cdc_h_max - self.cdc_h_min)
                + self.cdc_h_min
            )
        if self.cdc_sampling_method != "loguniform":
            raise ValueError(
                "cdc.sampling_method must be 'uniform' or 'loguniform', got "
                f"{self.cdc_sampling_method}"
            )
        log_min, log_max = math.log(self.cdc_h_min), math.log(self.cdc_h_max)
        return torch.exp(
            torch.rand(count, device=device, dtype=dtype) * (log_max - log_min)
            + log_min
        )

    @staticmethod
    def _balanced_indices(labels, max_points):
        valid = torch.nonzero(labels >= 0, as_tuple=False).flatten()
        if valid.numel() == 0 or max_points <= 0 or valid.numel() <= max_points:
            return valid

        valid_labels = labels[valid]
        classes = torch.unique(valid_labels)
        per_class = max(1, max_points // max(1, classes.numel()))
        selected = []
        for class_id in classes:
            idx = valid[valid_labels == class_id]
            if idx.numel() > per_class:
                idx = idx[
                    torch.randperm(idx.numel(), device=idx.device)[:per_class]
                ]
            selected.append(idx)
        selected = torch.cat(selected, dim=0)
        if selected.numel() < max_points:
            remain = max_points - selected.numel()
            extra = valid[
                torch.randint(0, valid.numel(), (remain,), device=valid.device)
            ]
            selected = torch.cat([selected, extra], dim=0)
        elif selected.numel() > max_points:
            selected = selected[
                torch.randperm(selected.numel(), device=selected.device)[:max_points]
            ]
        return selected

    @staticmethod
    def _low_rank_metric_matching_loss(factor, diff, tikhonov_epsilon):
        """Exact low-rank Frobenius objective used by smart metric matching."""
        gram = torch.einsum("bdr,bdq->brq", factor, factor)
        frob = gram.square().sum(dim=(1, 2))
        projected = torch.einsum("bdr,bd->br", factor, diff)
        quad = projected.square().sum(dim=-1)
        const = diff.square().sum(dim=-1).square()
        tikhonov = 0.0
        if tikhonov_epsilon != 0.0:
            tikhonov = (
                2.0 * tikhonov_epsilon * factor.square().sum(dim=(1, 2))
            )
        dim = factor.shape[1]
        return ((frob - 2.0 * quad + const + tikhonov) / (dim * dim)).mean()

    def _text_scores(self, feature, detach_projector=False):
        embedding = self.semantic_projector(
            feature, detach_parameters=detach_projector
        )
        text = F.normalize(self.text_embeddings.float(), dim=-1, eps=1e-6)
        return embedding @ text.transpose(0, 1)

    @staticmethod
    def _gt_and_margin(scores, labels):
        gt = scores.gather(1, labels.unsqueeze(1)).squeeze(1)
        wrong = scores.clone()
        wrong.scatter_(1, labels.unsqueeze(1), float("-inf"))
        max_wrong = wrong.max(dim=1).values
        return gt, gt - max_wrong

    def _semantic_alignment_loss(self, clean, labels):
        scores = self._text_scores(clean, detach_projector=False)
        logits = self.text_logit_scale * scores
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        return loss, acc

    def _semantic_direction_loss(self, clean, labels):
        if clean.shape[0] > self.text_max_points:
            perm = torch.randperm(clean.shape[0], device=clean.device)[: self.text_max_points]
            clean, labels = clean[perm], labels[perm]

        relative_h = torch.full(
            (clean.shape[0],),
            self.cdc_probe_h,
            device=clean.device,
            dtype=clean.dtype,
        )
        feature_rms = clean.square().mean(dim=-1).add(1e-8).sqrt()
        bandwidth = relative_h * feature_rms
        factor = self.cdc_head(clean, bandwidth)

        with torch.no_grad():
            clean_scores = self._text_scores(clean, detach_projector=True)
            clean_gt, clean_margin = self._gt_and_margin(clean_scores, labels)

        losses, margins, valid_flags = [], [], []
        radius = self.text_probe_radius * clean.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        for _ in range(self.text_num_direction_probes):
            latent = torch.randn(
                clean.shape[0], self.cdc_rank, device=clean.device, dtype=clean.dtype
            )
            direction = torch.einsum("bdr,br->bd", factor, latent)
            direction = F.normalize(direction, dim=-1, eps=1e-6)
            delta = radius * direction
            signed_losses = []

            for virtual in (clean + delta, clean - delta):
                scores = self._text_scores(virtual, detach_projector=True)
                virtual_gt, virtual_margin = self._gt_and_margin(scores, labels)
                positive_loss = F.relu(
                    clean_gt.detach() - self.text_positive_tolerance - virtual_gt
                )
                margin_loss = F.relu(
                    clean_margin.detach() - self.text_margin_tolerance - virtual_margin
                )
                signed_losses.append(positive_loss + margin_loss)
                margins.append(virtual_margin.detach())
                valid_flags.append(
                    (
                        (virtual_gt >= clean_gt.detach() - self.text_positive_tolerance)
                        & (
                            virtual_margin
                            >= clean_margin.detach() - self.text_margin_tolerance
                        )
                    ).float()
                )
            losses.append(0.5 * (signed_losses[0] + signed_losses[1]))

        return (
            torch.stack(losses, dim=0).mean(),
            torch.cat(margins, dim=0).mean(),
            torch.cat(valid_flags, dim=0).mean(),
        )

    @torch.no_grad()
    def get_cdc_factor(self, feature, relative_h=None):
        """Inference API reserved for the later few-shot expansion stage."""
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
        return self.cdc_head(feature, relative * feature_rms)

    def forward(self, input_dict):
        if not bool(self._text_ready.item()):
            raise RuntimeError(
                "Frozen text embeddings are not initialized. "
                "Use CDCBaseGeometryTrainer for Step-1 training."
            )

        with torch.no_grad():
            point = self.backbone(Point(input_dict))
            feature = point.feat if isinstance(point, Point) else point

        labels = input_dict["segment"].long()
        indices = self._balanced_indices(labels, self.cdc_max_points)
        if indices.numel() == 0:
            raise RuntimeError("The current batch contains no valid base-class points.")

        # Keep Gram/quadratic CDC terms in fp32 even when the outer trainer uses AMP.
        with torch.cuda.amp.autocast(enabled=False):
            clean = feature.detach().float()[indices]
            sampled_labels = labels[indices]

            relative_h = self._sample_relative_h(
                clean.shape[0], clean.device, clean.dtype
            )
            feature_rms = clean.square().mean(dim=-1).add(1e-8).sqrt()
            h_abs = relative_h * feature_rms
            noise = torch.randn_like(clean) * h_abs.unsqueeze(-1)
            noisy = clean + noise
            diff = -noise / (
                math.sqrt(2.0) * h_abs.unsqueeze(-1).clamp_min(1e-8)
            )

            factor = self.cdc_head(noisy, h_abs)
            cdc_loss = self._low_rank_metric_matching_loss(
                factor, diff, self.cdc_tikhonov
            )
            text_align_loss, text_acc = self._semantic_alignment_loss(
                clean, sampled_labels
            )

            if self.current_epoch >= self.text_warmup_epochs:
                (
                    text_direction_loss,
                    text_virtual_margin,
                    text_virtual_valid_rate,
                ) = self._semantic_direction_loss(clean, sampled_labels)
                direction_weight = self.text_direction_weight
            else:
                text_direction_loss = clean.new_zeros(())
                text_virtual_margin = clean.new_zeros(())
                text_virtual_valid_rate = clean.new_zeros(())
                direction_weight = 0.0

            loss = (
                self.cdc_loss_weight * cdc_loss
                + self.text_align_weight * text_align_loss
                + direction_weight * text_direction_loss
            )
            factor_energy = (
                factor.square().sum(dim=(1, 2)) / self.feature_dim
            ).mean()

        return {
            "loss": loss,
            "cdc_loss": cdc_loss.detach(),
            "text_align_loss": text_align_loss.detach(),
            "text_direction_loss": text_direction_loss.detach(),
            "text_acc": text_acc.detach(),
            "text_virtual_margin": text_virtual_margin.detach(),
            "text_virtual_valid_rate": text_virtual_valid_rate.detach(),
            "cdc_factor_energy": factor_energy.detach(),
            "relative_h": relative_h.mean().detach(),
        }
