import unittest

import torch
import torch.nn.functional as F

from pointcept.engines.semantic_metric import SemanticMetricMatcher


class SemanticMetricMatcherTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.dim = 16
        self.num_novel = 3
        self.query = torch.randn(9, self.dim)
        self.prototypes = torch.randn(self.num_novel, self.dim)
        self.text_embed = torch.randn(7, self.dim)

    def _fit_pca(self, matcher):
        matcher.reset_support_statistics(
            self.num_novel, self.dim, device=self.query.device
        )
        for class_idx in range(self.num_novel):
            features = (
                torch.randn(64, self.dim)
                + 0.25 * class_idx
                + torch.arange(self.dim).float()[None, :] * 0.01
            )
            matcher.update_support_statistics(class_idx, features)
        matcher.finalize_support_statistics()

    def _assert_valid_similarity(self, sim):
        self.assertEqual(sim.shape, (9, 3))
        self.assertTrue(torch.isfinite(sim).all())
        self.assertLessEqual(sim.max().item(), 1.00001)
        self.assertGreaterEqual(sim.min().item(), -1.00001)

    def test_cosine_matches_original(self):
        matcher = SemanticMetricMatcher(metric="cosine")
        actual = matcher.similarity(self.query, self.prototypes)
        expected = F.normalize(self.query, dim=-1) @ F.normalize(
            self.prototypes, dim=-1
        ).transpose(0, 1)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_pca_mahalanobis(self):
        matcher = SemanticMetricMatcher(
            metric="pca_mahalanobis", pca_rank=8
        )
        self._fit_pca(matcher)
        sim = matcher.similarity(self.query, self.prototypes)
        self._assert_valid_similarity(sim)

    def test_local_pca(self):
        matcher = SemanticMetricMatcher(metric="local_pca", pca_rank=8)
        self._fit_pca(matcher)
        sim = matcher.similarity(
            self.query, self.prototypes, class_indices=[0, 1, 2]
        )
        self._assert_valid_similarity(sim)

    def test_fisher_riemannian(self):
        matcher = SemanticMetricMatcher(
            metric="fisher_riemannian",
            text_embed=self.text_embed,
            fisher_temperature=1.0,
        )
        sim = matcher.similarity(self.query, self.prototypes)
        self._assert_valid_similarity(sim)


if __name__ == "__main__":
    unittest.main()
