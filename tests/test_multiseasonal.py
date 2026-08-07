from __future__ import annotations

import pytest
import torch

from ts_project.models import DLinear, MultiSeasonalDecomposition, PeriodAwareDLinear


def test_multiseasonal_components_reconstruct_input() -> None:
    torch.manual_seed(31)
    series = torch.randn(2, 336, 3)

    irregular, daily, weekly, trend = MultiSeasonalDecomposition(
        input_length=336
    )(series)

    torch.testing.assert_close(irregular + daily + weekly + trend, series)


def test_daily_and_weekly_components_repeat_at_their_periods() -> None:
    series = torch.randn(2, 336, 3)
    _, daily, weekly, _ = MultiSeasonalDecomposition(input_length=336)(series)

    torch.testing.assert_close(daily[:, :-24], daily[:, 24:])
    torch.testing.assert_close(weekly[:, :168], weekly[:, 168:])


def test_period_aware_dlinear_produces_direct_multichannel_forecast() -> None:
    model = PeriodAwareDLinear(
        input_length=336,
        prediction_length=96,
        channels=7,
    )

    forecast = model(torch.randn(4, 336, 7))

    assert forecast.shape == (4, 96, 7)


def test_shared_period_aware_model_adds_only_channelwise_strengths() -> None:
    baseline = DLinear(input_length=336, prediction_length=96, channels=7)
    period_aware = PeriodAwareDLinear(
        input_length=336,
        prediction_length=96,
        channels=7,
    )

    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    period_aware_parameters = sum(
        parameter.numel() for parameter in period_aware.parameters()
    )

    assert period_aware_parameters == baseline_parameters + 14


def test_multiseasonal_decomposition_requires_complete_weeks() -> None:
    with pytest.raises(ValueError, match="divisible by weekly_period"):
        MultiSeasonalDecomposition(input_length=335)
