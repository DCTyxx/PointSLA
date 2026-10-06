import torch

from pointcept.engines.learnable_fisher import LearnableCovarianceFisher


def _moments(features, labels, num_classes):
    feat_dim = features.shape[1]
    sums = torch.zeros(num_classes, feat_dim, dtype=torch.float32)
    outers = torch.zeros(num_classes, feat_dim, feat_dim, dtype=torch.float32)
    counts = torch.zeros(num_classes, dtype=torch.long)
    for c in range(num_classes):
        x = features[labels == c]
        sums[c] = x.sum(dim=0)
        outers[c] = x.transpose(0, 1) @ x
        counts[c] = x.shape[0]
    return sums, outers, counts


def _build_matcher():
    return LearnableCovarianceFisher(
        feat_dim=4,
        spectral_rank=2,
        low_rank=2,
        hidden_dim=8,
        alpha=0.1,
        eps=1e-3,
        shrinkage=0.1,
        relation_topk=1,
        init_scale=1e-3,
    )


def test_zero_region_scale_recovers_stage1_statistics():
    torch.manual_seed(7)
    features = torch.randn(24, 4)
    labels = torch.arange(24) % 3
    sums, outers, counts = _moments(features, labels, 3)

    stage1 = _build_matcher()
    stage1_stats = stage1.prepare_support_from_moments(sums, outers, counts)

    region = _build_matcher().configure_region_exploration(
        enabled=True, tangent_scale=0.0, residual_scale=0.0
    )
    region_stats = region.prepare_support_from_moments(sums, outers, counts)

    assert torch.allclose(region_stats["evals"], stage1_stats["evals"])
    assert torch.allclose(
        region_stats["residual_precs"], stage1_stats["residual_precs"]
    )


def test_region_exploration_expands_anisotropic_support_variance():
    torch.manual_seed(11)
    features = torch.randn(30, 4)
    labels = torch.arange(30) % 3
    sums, outers, counts = _moments(features, labels, 3)

    matcher = _build_matcher().configure_region_exploration(
        enabled=True, tangent_scale=0.5, residual_scale=0.1
    )
    stats = matcher.prepare_support_from_moments(sums, outers, counts)

    retained_ratio = stats["evals"] / stats["raw_evals"].clamp_min(1e-12)
    assert torch.allclose(
        retained_ratio,
        torch.full_like(retained_ratio, 1.25),
        rtol=1e-5,
        atol=1e-5,
    )

    explored_residual_var = 1.0 / stats["residual_precs"] - matcher.eps
    expected_residual_var = stats["raw_residual_vars"] * 1.01
    assert torch.allclose(
        explored_residual_var,
        expected_residual_var,
        rtol=1e-4,
        atol=1e-6,
    )


def test_region_exploration_reduces_precision_in_explored_directions():
    torch.manual_seed(13)
    features = torch.randn(30, 4)
    labels = torch.arange(30) % 3
    sums, outers, counts = _moments(features, labels, 3)

    base = _build_matcher()
    base_stats = base.prepare_support_from_moments(sums, outers, counts)

    region = _build_matcher().configure_region_exploration(
        enabled=True, tangent_scale=0.5, residual_scale=0.1
    )
    region_stats = region.prepare_support_from_moments(sums, outers, counts)

    # Larger local variance must not increase Fisher precision.
    assert torch.all(region_stats["evals"] >= base_stats["evals"])
    assert torch.all(
        region_stats["residual_precs"] <= base_stats["residual_precs"] + 1e-6
    )
