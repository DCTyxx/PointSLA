import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model


class ConditionalCDCMetricHead(nn.Module):
    """Low-rank conditional CDC estimator.

    Given a noisy feature y and its noise variance epsilon, predict a low-rank
    factor U(y, epsilon) such that Gamma(y, epsilon) = U^T U approximates the
    conditional carre du champ (CDC) matrix.
    """

    def __init__(self, feat_dim, rank=4, hidden_dim=128):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.rank = int(rank)
        self.net = nn.Sequential(
            nn.Linear(self.feat_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.rank * self.feat_dim),
        )

    def forward(self, y, epsilon):
        if epsilon.ndim == 1:
            epsilon = epsilon.unsqueeze(-1)
        # A log-noise coordinate is substantially better conditioned when the
        # training noise range spans several orders of magnitude.
        log_eps = torch.log(epsilon.clamp_min(1e-12))
        inp = torch.cat([y, log_eps], dim=-1)
        u = self.net(inp)
        return u.view(-1, self.rank, self.feat_dim)


class GeometryResidualAdapter(nn.Module):
    """Inject the learned local CDC geometry into point representations."""

    def __init__(self, feat_dim, hidden_dim=128, alpha=0.1):
        super().__init__()
        self.alpha = float(alpha)
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feat_dim),
        )
        # Start exactly from the baseline representation. Geometry refinement
        # is learned gradually by the segmentation objective.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat, gamma_feat):
        delta = self.net(torch.cat([feat, gamma_feat], dim=-1))
        return feat + self.alpha * delta


@MODELS.register_module()
class RegisTrainRMMCDCSegmentor(nn.Module):
    """GFS-VL registration segmentor with conditional-CDC RMM geometry.

    The baseline PTv3 and base/novel segmentation heads are preserved.  The
    added branch learns local geometry by Gaussian perturbation followed by
    conditional CDC regression, and uses the learned geometry to refine the
    point representation before the original segmentation heads.
    """

    def __init__(
        self,
        num_base_classes,
        num_novel_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        rmm_rank=4,
        rmm_hidden_dim=128,
        rmm_adapter_hidden_dim=128,
        rmm_epsilon_min=1e-4,
        rmm_epsilon_max=1e-2,
        rmm_ref_epsilon=1e-4,
        rmm_max_points=4096,
        rmm_refine_alpha=0.1,
        rmm_normalize_feature=True,
        rmm_detach_cdc_target=True,
    ):
        super().__init__()
        self.feat_dim = int(backbone_out_channels)
        self.seg_head = (
            nn.Linear(self.feat_dim, num_base_classes)
            if num_base_classes > 0
            else nn.Identity()
        )
        self.novel_seg_head = nn.Linear(
            self.feat_dim, num_novel_classes, bias=False
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

        if rmm_epsilon_min <= 0 or rmm_epsilon_max <= 0:
            raise ValueError("RMM epsilon values must be positive")
        if rmm_epsilon_min > rmm_epsilon_max:
            raise ValueError("rmm_epsilon_min must be <= rmm_epsilon_max")
        if rmm_ref_epsilon <= 0:
            raise ValueError("rmm_ref_epsilon must be positive")

        self.rmm_epsilon_min = float(rmm_epsilon_min)
        self.rmm_epsilon_max = float(rmm_epsilon_max)
        self.rmm_ref_epsilon = float(rmm_ref_epsilon)
        self.rmm_max_points = int(rmm_max_points)
        self.rmm_normalize_feature = bool(rmm_normalize_feature)
        self.rmm_detach_cdc_target = bool(rmm_detach_cdc_target)

        self.rmm_head = ConditionalCDCMetricHead(
            feat_dim=self.feat_dim,
            rank=rmm_rank,
            hidden_dim=rmm_hidden_dim,
        )
        self.geometry_adapter = GeometryResidualAdapter(
            feat_dim=self.feat_dim,
            hidden_dim=rmm_adapter_hidden_dim,
            alpha=rmm_refine_alpha,
        )

    def _normalize_geometry_feature(self, feat):
        if not self.rmm_normalize_feature:
            return feat
        return F.layer_norm(feat, (feat.shape[-1],))

    def _sample_epsilon(self, n, device, dtype):
        if self.rmm_epsilon_min == self.rmm_epsilon_max:
            return torch.full(
                (n, 1), self.rmm_epsilon_min, device=device, dtype=dtype
            )
        lo = math.log(self.rmm_epsilon_min)
        hi = math.log(self.rmm_epsilon_max)
        log_eps = torch.empty((n, 1), device=device, dtype=dtype).uniform_(lo, hi)
        return log_eps.exp()

    def _subsample_for_cdc(self, feat):
        if self.rmm_max_points <= 0 or feat.shape[0] <= self.rmm_max_points:
            return feat
        idx = torch.randperm(feat.shape[0], device=feat.device)[: self.rmm_max_points]
        return feat[idx]

    def compute_conditional_cdc_loss(self, feat):
        """Conditional CDC regression under Gaussian feature perturbations.

        For X from the PTv3 feature distribution, sample
            Y = X + sqrt(epsilon) * xi, xi ~ N(0, I),
        and regress Gamma_phi(Y, epsilon)=U^T U toward the sample-wise
        conditional CDC target
            (X-Y)(X-Y)^T / (2 epsilon).

        Expanding the Frobenius regression and dropping the target-only
        constant gives
            ||U U^T||_F^2 - ||U(X-Y)||_2^2 / epsilon.
        """
        feat = self._subsample_for_cdc(feat)
        x = self._normalize_geometry_feature(feat)
        if self.rmm_detach_cdc_target:
            # The original RMM treats X as observed data rather than an
            # optimizable target.  Detaching stabilizes the moving PTv3 feature
            # distribution while the segmentation path still trains PTv3.
            x = x.detach()

        # Keep the geometric regression in fp32 even when segmentation uses AMP.
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            epsilon = self._sample_epsilon(x.shape[0], x.device, x.dtype)
            noise = torch.randn_like(x)
            y = x + torch.sqrt(epsilon) * noise
            displacement = x - y

            u = self.rmm_head(y, epsilon)
            uut = torch.bmm(u, u.transpose(1, 2))
            cdc_energy = uut.square().sum(dim=(1, 2))
            u_disp = torch.bmm(u, displacement.unsqueeze(-1)).squeeze(-1)
            conditional_term = u_disp.square().sum(dim=1) / epsilon.squeeze(-1)
            loss = (cdc_energy - conditional_term).mean()
        return loss

    def geometry_refine(self, feat):
        """Refine features using the local CDC operator without changing cosine matching."""
        geom_x = self._normalize_geometry_feature(feat)
        eps = torch.full(
            (geom_x.shape[0], 1),
            self.rmm_ref_epsilon,
            device=geom_x.device,
            dtype=geom_x.dtype,
        )
        u = self.rmm_head(geom_x, eps)
        # Gamma z = U^T(U z) uses the learned local PSD geometry without
        # explicitly materializing the D x D matrix Gamma.
        coeff = torch.bmm(u, geom_x.unsqueeze(-1)).squeeze(-1)
        gamma_feat = torch.bmm(u.transpose(1, 2), coeff.unsqueeze(-1)).squeeze(-1)
        gamma_feat = self._normalize_geometry_feature(gamma_feat)
        return self.geometry_adapter(feat, gamma_feat)

    def getFeatures(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        feat = point.feat if isinstance(point, Point) else point
        return self.geometry_refine(feat)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        feat = point.feat if isinstance(point, Point) else point

        refined_feat = self.geometry_refine(feat)
        base_seg_logits = self.seg_head(refined_feat)
        novel_seg_logits = self.novel_seg_head(refined_feat)
        seg_logits = torch.cat([base_seg_logits, novel_seg_logits], dim=1)

        output = dict(
            seg_logits=seg_logits,
            backbone_feat=feat,
            geometry_feat=refined_feat,
        )
        if self.training:
            output["rmm_cdc_loss"] = self.compute_conditional_cdc_loss(feat)
        elif "segment" in input_dict:
            output["loss"] = self.criteria(seg_logits, input_dict["segment"])
        return output
