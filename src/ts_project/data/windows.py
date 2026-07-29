"""Leakage-safe sliding windows for long-horizon forecasting."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from ts_project.data.etth1 import ETTh1Data, SPLIT_BOUNDS


class ForecastWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Map a continuous multivariate series to input/forecast tensor pairs."""

    def __init__(
        self,
        values: np.ndarray,
        *,
        target_start: int,
        target_end: int,
        input_length: int,
        prediction_length: int,
    ) -> None:
        if values.ndim != 2:
            raise ValueError("values must have shape [time, variables].")
        if input_length <= 0 or prediction_length <= 0:
            raise ValueError("Window lengths must be positive.")
        if not 0 <= target_start < target_end <= len(values):
            raise ValueError("Target bounds are outside the supplied time series.")

        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.target_start = target_start
        self.target_end = target_end
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.first_origin = max(input_length, target_start)
        self.last_origin = target_end - prediction_length
        if self.first_origin > self.last_origin:
            raise ValueError("The requested partition is too short for these windows.")

    def __len__(self) -> int:
        return self.last_origin - self.first_origin + 1

    def origin_at(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.first_origin + index

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        origin = self.origin_at(index)
        inputs = self.values[origin - self.input_length : origin]
        targets = self.values[origin : origin + self.prediction_length]
        return inputs, targets


def build_window_datasets(
    data: ETTh1Data,
    *,
    input_length: int = 336,
    prediction_length: int = 96,
) -> Mapping[str, ForecastWindowDataset]:
    """Build train, validation, and test windows from one scaled series."""

    values = data.scaled.to_numpy(copy=False)
    return {
        name: ForecastWindowDataset(
            values,
            target_start=start,
            target_end=end,
            input_length=input_length,
            prediction_length=prediction_length,
        )
        for name, (start, end) in SPLIT_BOUNDS.items()
    }
