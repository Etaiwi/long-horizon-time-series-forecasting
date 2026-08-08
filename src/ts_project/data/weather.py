"""Leakage-safe loading and chronological splits for the Weather benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ts_project.data.windows import ForecastWindowDataset

TIMESTAMP_COLUMN = "date"
EXPECTED_ROWS = 52_696
EXPECTED_CHANNELS = 21


@dataclass(frozen=True)
class WeatherSplitBounds:
    """Non-overlapping target bounds in the complete Weather series."""

    train: tuple[int, int]
    validation: tuple[int, int]
    test: tuple[int, int]


@dataclass
class WeatherData:
    """Validated Weather data and its training-fitted standardized copy."""

    raw: pd.DataFrame
    scaled: pd.DataFrame
    scaler: StandardScaler
    bounds: WeatherSplitBounds

    @property
    def channel_names(self) -> list[str]:
        return list(self.raw.columns)

    def split(self, name: str, *, scaled: bool = False) -> pd.DataFrame:
        if name not in {"train", "validation", "test"}:
            raise KeyError("name must be 'train', 'validation', or 'test'.")
        start, end = getattr(self.bounds, name)
        source = self.scaled if scaled else self.raw
        return source.iloc[start:end]


def weather_split_bounds(rows: int) -> WeatherSplitBounds:
    """Reproduce the official custom-data 70/10/20 chronological split."""

    if rows <= 0:
        raise ValueError("rows must be positive.")
    train_end = int(rows * 0.7)
    test_rows = int(rows * 0.2)
    validation_end = rows - test_rows
    return WeatherSplitBounds(
        train=(0, train_end),
        validation=(train_end, validation_end),
        test=(validation_end, rows),
    )


def load_weather(path: str | Path, *, strict_benchmark: bool = True) -> pd.DataFrame:
    """Load Weather while retaining its supplied benchmark row ordering.

    The public file contains one duplicate timestamp and one gap. They are
    documented but intentionally retained so the reconstruction sees the same
    row sequence as the official DLinear benchmark.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Weather was not found at {path}. Place weather.csv in data/raw/."
        )

    frame = pd.read_csv(path)
    if TIMESTAMP_COLUMN not in frame.columns:
        raise ValueError("Weather must contain a 'date' column.")

    timestamps = pd.to_datetime(frame.pop(TIMESTAMP_COLUMN), errors="raise")
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    if strict_benchmark and len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} rows; received {len(frame):,}.")
    if strict_benchmark and frame.shape[1] != EXPECTED_CHANNELS:
        raise ValueError(
            f"Expected {EXPECTED_CHANNELS} variables; received {frame.shape[1]}."
        )
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Weather timestamps are not in non-decreasing order.")
    if frame.isna().any().any():
        raise ValueError("Weather contains missing numeric values.")
    if not np.isfinite(frame.to_numpy(dtype=np.float64, copy=False)).all():
        raise ValueError("Weather contains non-finite numeric values.")

    frame.index = pd.DatetimeIndex(timestamps, name=TIMESTAMP_COLUMN)
    return frame


def prepare_weather(path: str | Path, *, strict_benchmark: bool = True) -> WeatherData:
    """Validate Weather and standardize it using training observations only."""

    raw = load_weather(path, strict_benchmark=strict_benchmark)
    bounds = weather_split_bounds(len(raw))
    train_start, train_end = bounds.train

    scaler = StandardScaler()
    scaler.fit(raw.iloc[train_start:train_end].to_numpy())
    scaled = pd.DataFrame(
        scaler.transform(raw.to_numpy()),
        index=raw.index,
        columns=raw.columns,
    )
    return WeatherData(raw=raw, scaled=scaled, scaler=scaler, bounds=bounds)


def build_weather_window_datasets(
    data: WeatherData,
    *,
    input_length: int = 336,
    prediction_length: int = 96,
) -> dict[str, ForecastWindowDataset]:
    """Create windows whose targets stay inside their chronological split."""

    values = data.scaled.to_numpy(copy=False)
    return {
        name: ForecastWindowDataset(
            values,
            target_start=start,
            target_end=end,
            input_length=input_length,
            prediction_length=prediction_length,
        )
        for name, (start, end) in {
            "train": data.bounds.train,
            "validation": data.bounds.validation,
            "test": data.bounds.test,
        }.items()
    }


__all__ = [
    "EXPECTED_CHANNELS",
    "EXPECTED_ROWS",
    "WeatherData",
    "WeatherSplitBounds",
    "build_weather_window_datasets",
    "load_weather",
    "prepare_weather",
    "weather_split_bounds",
]
