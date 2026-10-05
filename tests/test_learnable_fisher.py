import unittest

import torch
import torch.nn.functional as F

from pointcept.engines.learnable_fisher import LearnableCovarianceFisher


class LearnableCovarianceFisherTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.dim = 16
        self.num_classes = 3
        self.query = torch.randn(13, self.dim)

        sums = []
        outers = []
        counts = []
        for c in range(self.num_classes):
            x = torch.randn(48, self.dim) * (0.6 + 0.2 * c)
            x = x + 0.35 * c
            sums.append(x.sum(dim=0))
            outers.append(x.transpose(0, 1) @ x)
            counts.append(x.shape[0])
        self.sums = torch.stack(sums)
        self.outers = torch.stack(outers)
        self.counts = torch.tensor(counts, dtype=torch.long)

    def _build(self):
        matcher = LearnableCovarianceFisher(
            feat_dim=self.dim,
            low_rank=4,
            hidden_dim=24,
            alpha=0.25,
            eps=1e-3,
            shrinkage=0.1,
            relation_topk=2,
            init_scale=1e-3,
        )
        matcher.prepare_support_from_moments(
            self.sums, self.outers, self.counts
        )
        return matcher

    def test_initial_geometry_is_close_to_fixed_fisher(self):
        matcher = self._build()
        fixed = matcher.fixed_fisher_distances(self.query)
        learned = matcher.distances(self.query)
        relative = (learned - fixed).abs().mean() / fixed.abs().mean().clamp_min(1e-8)
        self.assertLess(relative.item(), 1e-3)

    def test_low_rank_deformation_backpropagates(self):
        matcher = self._build()
        target = torch.randint(0, self.num_classes, (self.query.shape[0],))
        logits = matcher.logits(self.query, temperature=1.0)
        loss = F.cross_entropy(logits, target)
        loss = loss + 1e-3 * matcher.deformation_regularization()
        loss.backward()

        grads = [
            p.grad
            for p in matcher.parameters()
            if p.requires_grad and p.grad is not None
        ]
        self.assertTrue(len(grads) > 0)
        self.assertGreater(sum(g.abs().sum().item() for g in grads), 0.0)

    def test_metric_is_positive_and_probabilities_are_valid(self):
        matcher = self._build()
        distances = matcher.distances(self.query)
        probs = matcher.probabilities(self.query)
        self.assertTrue(torch.isfinite(distances).all())
        self.assertTrue(torch.all(distances >= -1e-7))
        self.assertTrue(torch.isfinite(probs).all())
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(probs.shape[0]), atol=1e-6))

    def test_inter_class_relation_shape(self):
        matcher = self._build()
        self.assertEqual(
            matcher.support_stats["relations"].shape,
            (self.num_classes, 2 * matcher.relation_topk),
        )


if __name__ == "__main__":
    unittest.main()
