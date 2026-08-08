from __future__ import annotations

import numpy as np
import pandas as pd

from ts_project.data import (
    build_weather_window_datasets,
    prepare_weather,
    weather_split_bounds,
)


def test_weather_split_and_scaler_use_training_only(tmp_path) -> None:
    rows = 100
    dates = pd.date_range("2020-01-01", periods=rows, freq="10min")
    frame = pd.DataFrame(
        {
            "date": dates,
            "x1": np.arange(rows, dtype=float),
            "x2": 2.0 * np.arange(rows, dtype=float),
        }
    )
    path = tmp_path / "weather.csv"
    frame.to_csv(path, index=False)

    data = prepare_weather(path, strict_benchmark=False)
    bounds = weather_split_bounds(rows)
    assert bounds.train == (0, 70)
    assert bounds.validation == (70, 80)
    assert bounds.test == (80, 100)
    np.testing.assert_allclose(data.scaled.iloc[:70].mean().to_numpy(), 0.0, atol=1e-7)
    assert not np.allclose(data.scaled.iloc[70:].mean().to_numpy(), 0.0)

    datasets = build_weather_window_datasets(
        data,
        input_length=12,
        prediction_length=4,
    )
    first_validation_origin = datasets["validation"].origin_at(0)
    first_test_origin = datasets["test"].origin_at(0)
    assert first_validation_origin == 70
    assert first_test_origin == 80
    assert datasets["validation"][0][1].shape == (4, 2)
    assert datasets["test"][0][1].shape == (4, 2)
