import unittest

import torch
import torch.nn.functional as F

from pointcept.engines.learnable_fisher import LearnableFisherRiemannianMatcher
from pointcept.engines.semantic_metric import SemanticMetricMatcher


class LearnableFisherMatcherTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.dim = 16
        self.num_text_classes = 7
        self.num_proto = 3
        self.query = torch.randn(11, self.dim)
        self.prototypes = torch.randn(self.num_proto, self.dim)
        self.text_embed = torch.randn(self.num_text_classes, self.dim)

    def test_identity_initialization_matches_fixed_fisher(self):
        fixed = SemanticMetricMatcher(
            metric="fisher_riemannian",
            text_embed=self.text_embed,
            metric_ridge=1e-3,
            fisher_temperature=1.0,
        )
        learned = LearnableFisherRiemannianMatcher(
            text_embed=self.text_embed,
            metric_ridge=1e-3,
            fisher_temperature=1.0,
            hidden_dim=8,
            residual_scale=0.25,
            learn_temperature=True,
        )

        expected = fixed.similarity(self.query, self.prototypes)
        actual = learned.similarity(self.query, self.prototypes)
        self.assertTrue(
            torch.allclose(actual, expected, atol=1e-6, rtol=1e-6),
            msg=(actual - expected).abs().max().item(),
        )

    def test_supervised_metric_loss_backpropagates(self):
        matcher = LearnableFisherRiemannianMatcher(
            text_embed=self.text_embed,
            hidden_dim=8,
            residual_scale=0.25,
            learn_temperature=True,
        )
        target = torch.randint(0, self.num_proto, (self.query.shape[0],))
        similarities = matcher.similarity(self.query, self.prototypes)
        loss = F.cross_entropy(similarities / 0.1, target)
        loss.backward()

        grads = [
            p.grad
            for p in matcher.parameters()
            if p.requires_grad and p.grad is not None
        ]
        self.assertTrue(len(grads) > 0)
        total_grad = sum(g.abs().sum().item() for g in grads)
        self.assertGreater(total_grad, 0.0)

    def test_geometry_remains_finite_after_update(self):
        matcher = LearnableFisherRiemannianMatcher(
            text_embed=self.text_embed,
            hidden_dim=8,
            residual_scale=0.25,
        )
        optimizer = torch.optim.AdamW(matcher.parameters(), lr=1e-3)
        target = torch.randint(0, self.num_proto, (self.query.shape[0],))

        for _ in range(3):
            optimizer.zero_grad()
            sim = matcher.similarity(self.query, self.prototypes)
            loss = F.cross_entropy(sim / 0.1, target)
            loss = loss + 1e-3 * matcher.deformation_regularization(self.query)
            loss.backward()
            optimizer.step()

        sim = matcher.similarity(self.query, self.prototypes)
        self.assertTrue(torch.isfinite(sim).all())
        self.assertLessEqual(sim.max().item(), 1.00001)
        self.assertGreaterEqual(sim.min().item(), -1.00001)


if __name__ == "__main__":
    unittest.main()
