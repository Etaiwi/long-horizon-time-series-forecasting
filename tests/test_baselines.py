from __future__ import annotations

import pytest
import torch

from ts_project.baselines import last_value_forecast, seasonal_naive_forecast


def test_last_value_forecast_repeats_latest_observation() -> None:
    history = torch.tensor(
        [
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ]
        ]
    )

    forecast = last_value_forecast(history, prediction_length=4)

    expected = torch.tensor(
        [
            [
                [3.0, 30.0],
                [3.0, 30.0],
                [3.0, 30.0],
                [3.0, 30.0],
            ]
        ]
    )
    torch.testing.assert_close(forecast, expected)


def test_seasonal_naive_repeats_latest_complete_season() -> None:
    history = torch.arange(1, 7, dtype=torch.float32).reshape(1, 6, 1)

    forecast = seasonal_naive_forecast(
        history,
        prediction_length=5,
        season_length=3,
    )

    expected = torch.tensor([[[4.0], [5.0], [6.0], [4.0], [5.0]]])
    torch.testing.assert_close(forecast, expected)


def test_seasonal_naive_requires_enough_history() -> None:
    history = torch.zeros(2, 12, 7)

    with pytest.raises(ValueError, match="season_length is 24"):
        seasonal_naive_forecast(history, prediction_length=96)
