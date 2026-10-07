import torch

from pointcept.models.rmm_cdc import (
    ConditionalCDCMetricHead,
    GeometryResidualAdapter,
)


def test_rmm_factor_defines_psd_cdc_matrix():
    torch.manual_seed(0)
    head = ConditionalCDCMetricHead(feat_dim=8, rank=3, hidden_dim=16)
    y = torch.randn(5, 8)
    epsilon = torch.full((5, 1), 1e-3)
    u = head(y, epsilon)
    gamma = torch.bmm(u.transpose(1, 2), u)
    eigvals = torch.linalg.eigvalsh(gamma)
    assert u.shape == (5, 3, 8)
    assert torch.all(eigvals >= -1e-5)


def test_conditional_cdc_low_rank_expansion_matches_matrix_regression_gradient_term():
    torch.manual_seed(1)
    n, d, r = 4, 6, 2
    u = torch.randn(n, r, d)
    x = torch.randn(n, d)
    epsilon = torch.full((n, 1), 5e-3)
    y = x + torch.sqrt(epsilon) * torch.randn_like(x)
    v = x - y

    gamma = torch.bmm(u.transpose(1, 2), u)
    target = torch.bmm(v.unsqueeze(-1), v.unsqueeze(1)) / (2.0 * epsilon.view(-1, 1, 1))
    full = (gamma - target).square().sum(dim=(1, 2))

    uut = torch.bmm(u, u.transpose(1, 2))
    reduced = uut.square().sum(dim=(1, 2)) - (
        torch.bmm(u, v.unsqueeze(-1)).squeeze(-1).square().sum(dim=1)
        / epsilon.squeeze(-1)
    )
    target_constant = target.square().sum(dim=(1, 2))

    assert torch.allclose(full, reduced + target_constant, atol=1e-5, rtol=1e-5)


def test_geometry_adapter_starts_from_baseline_feature():
    torch.manual_seed(2)
    adapter = GeometryResidualAdapter(feat_dim=8, hidden_dim=16, alpha=0.1)
    feat = torch.randn(7, 8)
    gamma_feat = torch.randn(7, 8)
    refined = adapter(feat, gamma_feat)
    assert torch.allclose(refined, feat)
