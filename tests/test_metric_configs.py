import os
import unittest

from pointcept.utils.config import Config


class MetricAblationConfigTest(unittest.TestCase):
    def test_metric_configs(self):
        root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "configs", "scannet")
        )
        expected = {
            "semseg-pt-v3m1-0-gfsregistrain_k1-metric-cosine.py": "cosine",
            "semseg-pt-v3m1-0-gfsregistrain_k1-metric-pca-mahalanobis.py": "pca_mahalanobis",
            "semseg-pt-v3m1-0-gfsregistrain_k1-metric-local-pca.py": "local_pca",
            "semseg-pt-v3m1-0-gfsregistrain_k1-metric-fisher-riemannian.py": "fisher_riemannian",
            "semseg-pt-v3m1-0-gfsregistrain_k1-metric-fisher-support.py": "fisher_support",
            "semseg-pt-v3m1-0-gfsregistrain_k1-metric-learnable-fisher.py": "learnable_fisher",
        }
        for filename, metric in expected.items():
            cfg = Config.fromfile(os.path.join(root, filename))
            self.assertEqual(cfg.semantic_metric, metric)
            self.assertEqual(cfg.k_shot, 1)
            self.assertEqual(cfg.epoch, 20)
            self.assertEqual(cfg.model.type, "RegisTrainSegmentor")
            self.assertEqual(cfg.metric_pca_rank, 32)


if __name__ == "__main__":
    unittest.main()
