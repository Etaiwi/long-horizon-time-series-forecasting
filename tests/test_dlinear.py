from __future__ import annotations

import pytest
import torch

from ts_project.models.dlinear import DLinear, MovingAverage, SeriesDecomposition


def test_moving_average_preserves_a_constant_series() -> None:
    series = torch.full((2, 8, 3), 4.5)

    averaged = MovingAverage(kernel_size=5)(series)

    torch.testing.assert_close(averaged, series)


def test_decomposition_components_reconstruct_the_input() -> None:
    torch.manual_seed(7)
    series = torch.randn(2, 12, 3)

    seasonal, trend = SeriesDecomposition(kernel_size=5)(series)

    torch.testing.assert_close(seasonal + trend, series)


def test_dlinear_produces_direct_multichannel_forecast() -> None:
    model = DLinear(
        input_length=8,
        prediction_length=3,
        channels=2,
        moving_average=3,
    )
    series = torch.randn(4, 8, 2)

    forecast = model(series)

    assert forecast.shape == (4, 3, 2)


def test_shared_dlinear_does_not_mix_channel_values() -> None:
    torch.manual_seed(11)
    model = DLinear(
        input_length=8,
        prediction_length=3,
        channels=2,
        moving_average=3,
        individual=False,
    )
    original = torch.zeros(1, 8, 2)
    changed = original.clone()
    changed[:, :, 0] = torch.arange(8, dtype=torch.float32)

    original_forecast = model(original)
    changed_forecast = model(changed)

    torch.testing.assert_close(original_forecast[:, :, 1], changed_forecast[:, :, 1])


def test_individual_mode_has_separate_parameters_per_channel() -> None:
    shared = DLinear(input_length=4, prediction_length=2, channels=3)
    individual = DLinear(
        input_length=4,
        prediction_length=2,
        channels=3,
        individual=True,
    )

    shared_parameters = sum(parameter.numel() for parameter in shared.parameters())
    individual_parameters = sum(parameter.numel() for parameter in individual.parameters())

    assert shared_parameters == 20
    assert individual_parameters == 60


def test_dlinear_rejects_wrong_input_length() -> None:
    model = DLinear(input_length=8, prediction_length=3, channels=2)

    with pytest.raises(ValueError, match="Expected 8 input time steps"):
        model(torch.zeros(1, 7, 2))
