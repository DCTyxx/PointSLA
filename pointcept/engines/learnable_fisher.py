import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableFisherRiemannianMatcher(nn.Module):
    """Task-learnable deformation of the frozen semantic Fisher metric.

    The fixed Fisher pullback induced by the frozen text classifier is

        G_F(z) = T^T [Diag(p(z)) - p(z)p(z)^T] T.

    We preserve this statistical geometry and learn only a positive diagonal
    deformation in classifier space,

        G_theta(z) = T^T W_theta(z) F(z) W_theta(z) T + ridge * I,

    where W_theta(z) is initialized to the identity.  Consequently the module
    exactly recovers the original Fisher-Riemannian metric at initialization,
    while task supervision can learn which semantic competition directions
    should receive more or less curvature.
    """

    def __init__(
        self,
        text_embed,
        metric_ridge=1e-3,
        fisher_temperature=1.0,
        hidden_dim=32,
        residual_scale=0.25,
        learn_temperature=True,
        max_log_weight=2.0,
    ):
        super().__init__()
        if text_embed is None:
            raise ValueError(
                "LearnableFisherRiemannianMatcher requires frozen text embeddings."
            )

        self.register_buffer("text_embed", text_embed.detach().float())
        self.metric = "fisher_riemannian"
        self.metric_ridge = float(metric_ridge)
        self.residual_scale = float(residual_scale)
        self.max_log_weight = float(max_log_weight)
        self.learn_temperature = bool(learn_temperature)

        init_temp = max(float(fisher_temperature), 1e-4)
        log_temp = torch.tensor(math.log(init_temp), dtype=torch.float32)
        if self.learn_temperature:
            self.log_temperature = nn.Parameter(log_temp)
        else:
            self.register_buffer("log_temperature", log_temp)

        # The same small calibrator is shared by all semantic classes.  This is
        # deliberately class-agnostic so the learned rule transfers to novel
        # classes rather than memorising a per-class metric.
        self.weight_net = nn.Sequential(
            nn.Linear(3, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

        # Zero-init the residual head: W_theta(z)=I at step 0, i.e. the exact
        # fixed Fisher baseline before learning starts.
        nn.init.zeros_(self.weight_net[-1].weight)
        nn.init.zeros_(self.weight_net[-1].bias)

    @property
    def needs_support_statistics(self):
        return False

    def reset_support_statistics(self, *args, **kwargs):
        return None

    def update_support_statistics(self, *args, **kwargs):
        return None

    def finalize_support_statistics(self):
        return None

    def temperature(self):
        return self.log_temperature.exp().clamp(0.05, 20.0)

    def _fisher_state(self, query):
        q = query.float()
        text = self.text_embed.to(q.device, dtype=q.dtype)
        q_logits = q @ text.transpose(0, 1)

        temperature = self.temperature().to(q.dtype)
        prob = torch.softmax(q_logits / temperature, dim=-1)

        mean = q_logits.mean(dim=-1, keepdim=True)
        std = q_logits.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        normalized_logits = (q_logits - mean) / std

        # Three class-agnostic descriptors: semantic evidence, posterior mass,
        # and local categorical uncertainty.  The resulting correction remains
        # point-dependent while sharing parameters across base/novel classes.
        descriptor = torch.stack(
            [normalized_logits, prob, prob * (1.0 - prob)], dim=-1
        )
        delta = self.weight_net(descriptor).squeeze(-1)
        log_weight = (self.residual_scale * delta).clamp(
            -self.max_log_weight, self.max_log_weight
        )
        weight = torch.exp(log_weight)
        return q_logits, prob, weight

    def similarity(self, query, prototypes, class_indices=None):
        """Return Fisher-metric cosine similarities [N_query, N_proto]."""
        del class_indices  # Kept for API compatibility with SemanticMetricMatcher.
        if query.ndim == 1:
            query = query.unsqueeze(0)
        if prototypes.ndim == 1:
            prototypes = prototypes.unsqueeze(0)

        q = query.float()
        p = prototypes.float()
        text = self.text_embed.to(q.device, dtype=q.dtype)

        q_logits, prob, weight = self._fisher_state(q)
        p_logits = p @ text.transpose(0, 1)

        # W_theta is diagonal and strictly positive, so W F W remains PSD.
        q_tangent = q_logits * weight
        p_tangent = p_logits.unsqueeze(0) * weight.unsqueeze(1)

        eq = (prob * q_tangent).sum(dim=-1)
        ep = (prob.unsqueeze(1) * p_tangent).sum(dim=-1)
        eqp = (
            prob.unsqueeze(1)
            * q_tangent.unsqueeze(1)
            * p_tangent
        ).sum(dim=-1)
        fisher_inner = eqp - eq.unsqueeze(1) * ep

        q_var = (prob * q_tangent.square()).sum(dim=-1) - eq.square()
        p_var = (prob.unsqueeze(1) * p_tangent.square()).sum(dim=-1) - ep.square()

        euclidean_inner = q @ p.transpose(0, 1)
        inner = fisher_inner + self.metric_ridge * euclidean_inner

        q_norm = q_var + self.metric_ridge * q.square().sum(dim=-1)
        p_norm = p_var + self.metric_ridge * p.square().sum(dim=-1).unsqueeze(0)
        denom = torch.sqrt(
            q_norm.clamp_min(1e-12).unsqueeze(1)
            * p_norm.clamp_min(1e-12)
        )
        return (inner / denom).clamp(-1.0, 1.0)

    def deformation_regularization(self, query):
        """Keep the learned geometry close to the fixed Fisher anchor."""
        if query.numel() == 0:
            return self.log_temperature.new_zeros(())
        _, _, weight = self._fisher_state(query)
        identity_penalty = (weight - 1.0).square().mean()
        return identity_penalty

    def parameter_summary(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "trainable_parameters": int(total),
            "temperature": float(self.temperature().detach().cpu()),
        }
