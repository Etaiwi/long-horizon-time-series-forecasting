from __future__ import annotations

import copy

import pytest
import torch

from ts_project.models import (
    CrossPeriodDailyBranch,
    CrossPeriodDailyDLinear,
    DailyLowRankDLinear,
    DLinear,
    LowRankDLinear,
)


def test_cross_period_branch_interleaves_daily_phases() -> None:
    branch = CrossPeriodDailyBranch(
        input_length=48,
        prediction_length=24,
        period=24,
        zero_init=False,
    )
    with torch.no_grad():
        branch.projection.weight.copy_(torch.tensor([[0.0, 1.0]]))
    series = torch.arange(48, dtype=torch.float32).reshape(1, 48, 1)

    forecast = branch(series)

    torch.testing.assert_close(forecast, series[:, 24:, :])


def test_daily_residual_starts_at_exact_backbone_forecast() -> None:
    torch.manual_seed(41)
    backbone = DLinear(input_length=48, prediction_length=24, channels=3)
    reference = copy.deepcopy(backbone)
    candidate = CrossPeriodDailyDLinear(backbone)
    series = torch.randn(2, 48, 3)

    torch.testing.assert_close(candidate(series), reference(series))


def test_low_rank_model_starts_as_shared_dlinear() -> None:
    torch.manual_seed(43)
    baseline = DLinear(input_length=48, prediction_length=24, channels=3)
    torch.manual_seed(43)
    candidate = LowRankDLinear(
        input_length=48,
        prediction_length=24,
        channels=3,
        rank=1,
    )
    series = torch.randn(2, 48, 3)

    torch.testing.assert_close(candidate(series), baseline(series))


def test_low_rank_adapter_receives_gradient_from_zero_initialization() -> None:
    model = LowRankDLinear(
        input_length=48,
        prediction_length=24,
        channels=3,
        rank=1,
    )
    model(torch.randn(2, 48, 3)).square().mean().backward()

    assert model.seasonal_adapter.up.grad is not None
    assert torch.count_nonzero(model.seasonal_adapter.up.grad) > 0


def test_combined_model_preserves_forecast_shape() -> None:
    backbone = LowRankDLinear(
        input_length=48,
        prediction_length=24,
        channels=3,
        rank=1,
    )
    forecast = DailyLowRankDLinear(backbone)(torch.randn(2, 48, 3))
    assert forecast.shape == (2, 24, 3)


def test_daily_branch_requires_complete_cycles() -> None:
    with pytest.raises(ValueError, match="divisible by period"):
        CrossPeriodDailyBranch(input_length=47, prediction_length=24)
