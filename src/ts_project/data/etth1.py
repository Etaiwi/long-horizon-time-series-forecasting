"""Loading and leakage-safe preprocessing for the ETTh1 benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TIMESTAMP_COLUMN = "date"
FEATURE_COLUMNS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
EXPECTED_COLUMNS = (TIMESTAMP_COLUMN, *FEATURE_COLUMNS)
EXPECTED_RAW_ROWS = 17_420

TRAIN_ROWS = 12 * 30 * 24
VALIDATION_ROWS = 4 * 30 * 24
TEST_ROWS = 4 * 30 * 24
BENCHMARK_ROWS = TRAIN_ROWS + VALIDATION_ROWS + TEST_ROWS

SPLIT_BOUNDS = {
    "train": (0, TRAIN_ROWS),
    "validation": (TRAIN_ROWS, TRAIN_ROWS + VALIDATION_ROWS),
    "test": (TRAIN_ROWS + VALIDATION_ROWS, BENCHMARK_ROWS),
}


def load_etth1(path: str | Path) -> pd.DataFrame:
    """Load ETTh1 and fail clearly if its temporal structure is unexpected."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"ETTh1 was not found at {path}. Run: python scripts/download_etth1.py"
        )

    frame = pd.read_csv(path)
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected ETTh1 columns: {tuple(frame.columns)}; "
            f"expected {EXPECTED_COLUMNS}."
        )
    if len(frame) != EXPECTED_RAW_ROWS:
        raise ValueError(
            f"Unexpected ETTh1 row count: {len(frame)}; "
            f"expected {EXPECTED_RAW_ROWS}."
        )

    timestamps = pd.to_datetime(frame.pop(TIMESTAMP_COLUMN), errors="raise")
    if timestamps.duplicated().any():
        raise ValueError("ETTh1 contains duplicate timestamps.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("ETTh1 timestamps are not in chronological order.")
    expected_timestamps = pd.date_range(
        start=timestamps.iloc[0], periods=len(timestamps), freq="h"
    )
    if not timestamps.reset_index(drop=True).equals(pd.Series(expected_timestamps)):
        raise ValueError("ETTh1 timestamps are not a complete hourly sequence.")

    if frame.isna().any().any():
        missing = int(frame.isna().sum().sum())
        raise ValueError(f"ETTh1 contains {missing} missing numeric values.")
    if not all(pd.api.types.is_numeric_dtype(frame[column]) for column in frame):
        raise ValueError("All ETTh1 feature columns must be numeric.")
    values = frame.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("ETTh1 contains non-finite numeric values.")

    frame.index = pd.DatetimeIndex(timestamps, name=TIMESTAMP_COLUMN)
    return frame


@dataclass
class ETTh1Data:
    """Validated raw data plus benchmark-only, train-fitted scaling."""

    raw: pd.DataFrame
    benchmark: pd.DataFrame
    scaled: pd.DataFrame
    scaler: StandardScaler

    def split(self, name: str, *, scaled: bool = False) -> pd.DataFrame:
        """Return a non-overlapping target partition."""

        if name not in SPLIT_BOUNDS:
            raise KeyError(f"Unknown split {name!r}; choose from {tuple(SPLIT_BOUNDS)}.")
        start, end = SPLIT_BOUNDS[name]
        source = self.scaled if scaled else self.benchmark
        return source.iloc[start:end]


def prepare_etth1(path: str | Path) -> ETTh1Data:
    """Validate ETTh1 and scale it using training statistics only."""

    raw = load_etth1(path)
    benchmark = raw.iloc[:BENCHMARK_ROWS].copy()

    scaler = StandardScaler()
    scaler.fit(benchmark.iloc[:TRAIN_ROWS].to_numpy())
    scaled_values = scaler.transform(benchmark.to_numpy())
    scaled = pd.DataFrame(
        scaled_values,
        index=benchmark.index,
        columns=benchmark.columns,
    )
    return ETTh1Data(raw=raw, benchmark=benchmark, scaled=scaled, scaler=scaler)
