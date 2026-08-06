from __future__ import annotations

import torch

from ts_project.models import (
    DLinear,
    GatedSliceAwareDLinear,
    RevINDLinear,
    SliceAwareDLinear,
    slice_statistics_loss,
)


def backbone() -> DLinear:
    return DLinear(input_length=48, prediction_length=24, channels=3, moving_average=3)


def test_revin_dlinear_preserves_forecast_shape() -> None:
    forecast = RevINDLinear(backbone())(torch.randn(2, 48, 3))
    assert forecast.shape == (2, 24, 3)


def test_slice_aware_predicts_one_future_daily_mean_per_channel() -> None:
    model = SliceAwareDLinear(backbone(), slice_length=24)
    forecast, statistics = model.forward_with_statistics(torch.randn(2, 48, 3))
    assert forecast.shape == (2, 24, 3)
    assert statistics.mean.shape == (2, 24, 3)
    assert statistics.scale is None


def test_scale_variant_produces_positive_future_scales() -> None:
    model = SliceAwareDLinear(backbone(), slice_length=24, predict_scale=True)
    _, statistics = model.forward_with_statistics(torch.randn(2, 48, 3))
    assert statistics.scale is not None
    assert torch.all(statistics.scale > 0)


def test_gate_is_channelwise_and_bounded() -> None:
    model = GatedSliceAwareDLinear(backbone(), SliceAwareDLinear(backbone()))
    forecast = model(torch.randn(2, 48, 3))
    assert forecast.shape == (2, 24, 3)
    assert model.gates.shape == (3,)
    assert torch.all((model.gates >= 0) & (model.gates <= 1))


def test_slice_statistics_loss_is_finite() -> None:
    model = SliceAwareDLinear(backbone(), slice_length=24, predict_scale=True)
    _, statistics = model.forward_with_statistics(torch.randn(2, 48, 3))
    loss = slice_statistics_loss(statistics, torch.randn(2, 24, 3), slice_length=24)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
