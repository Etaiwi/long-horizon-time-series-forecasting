from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ts_project.data.etth1 import (
    BENCHMARK_ROWS,
    EXPECTED_RAW_ROWS,
    FEATURE_COLUMNS,
    TRAIN_ROWS,
    load_etth1,
    prepare_etth1,
)
from ts_project.data.windows import build_window_datasets


def _write_synthetic_etth1(path) -> pd.DataFrame:
    row = np.arange(EXPECTED_RAW_ROWS, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2016-07-01", periods=EXPECTED_RAW_ROWS, freq="h"),
            **{
                column: row * (index + 1) + index
                for index, column in enumerate(FEATURE_COLUMNS)
            },
        }
    )
    frame.to_csv(path, index=False)
    return frame


def test_preparation_uses_only_training_statistics(tmp_path):
    path = tmp_path / "ETTh1.csv"
    original = _write_synthetic_etth1(path)

    data = prepare_etth1(path)

    expected_mean = original.loc[: TRAIN_ROWS - 1, FEATURE_COLUMNS].mean().to_numpy()
    np.testing.assert_allclose(data.scaler.mean_, expected_mean)
    np.testing.assert_allclose(
        data.split("train", scaled=True).mean().to_numpy(),
        np.zeros(len(FEATURE_COLUMNS)),
        atol=1e-12,
    )
    assert len(data.benchmark) == BENCHMARK_ROWS
    assert len(data.raw) == EXPECTED_RAW_ROWS


def test_windows_respect_temporal_partition_boundaries(tmp_path):
    path = tmp_path / "ETTh1.csv"
    _write_synthetic_etth1(path)
    data = prepare_etth1(path)

    windows = build_window_datasets(
        data, input_length=336, prediction_length=96
    )

    assert windows["train"].origin_at(0) == 336
    assert windows["train"].origin_at(-1) + 96 == TRAIN_ROWS
    assert windows["validation"].origin_at(0) == TRAIN_ROWS
    assert windows["test"].origin_at(-1) + 96 == BENCHMARK_ROWS

    inputs, targets = windows["validation"][0]
    assert inputs.shape == (336, len(FEATURE_COLUMNS))
    assert targets.shape == (96, len(FEATURE_COLUMNS))
    np.testing.assert_allclose(
        inputs[-1].numpy(), data.scaled.iloc[TRAIN_ROWS - 1].to_numpy()
    )
    np.testing.assert_allclose(
        targets[0].numpy(), data.scaled.iloc[TRAIN_ROWS].to_numpy()
    )


def test_loader_rejects_a_missing_hour(tmp_path):
    path = tmp_path / "ETTh1.csv"
    frame = _write_synthetic_etth1(path)
    frame.loc[100, "date"] = frame.loc[99, "date"]
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate timestamps"):
        load_etth1(path)
