"""Faithful, concise reconstruction of the DLinear forecasting architecture."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _validate_series(series: Tensor) -> None:
    if not isinstance(series, Tensor):
        raise TypeError("series must be a PyTorch tensor.")
    if series.ndim != 3:
        raise ValueError(
            "series must have shape [batch, time, channels]; "
            f"received {tuple(series.shape)}."
        )
    if series.shape[1] == 0 or series.shape[2] == 0:
        raise ValueError("series must contain at least one time step and channel.")
    if not torch.is_floating_point(series):
        raise TypeError("series must use floating-point values.")


class MovingAverage(nn.Module):
    """Centered moving average with repeated endpoint padding."""

    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, series: Tensor) -> Tensor:
        _validate_series(series)

        front = series[:, :1, :].expand(-1, self.padding, -1)
        end = series[:, -1:, :].expand(-1, self.padding, -1)
        padded = torch.cat([front, series, end], dim=1)
        averaged = self.pool(padded.transpose(1, 2))
        return averaged.transpose(1, 2)


class SeriesDecomposition(nn.Module):
    """Split a series into a moving-average trend and its remainder."""

    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)

    def forward(self, series: Tensor) -> tuple[Tensor, Tensor]:
        trend = self.moving_average(series)
        seasonal_remainder = series - trend
        return seasonal_remainder, trend


class DLinear(nn.Module):
    """Decompose each channel, forecast both components linearly, and add them."""

    def __init__(
        self,
        *,
        input_length: int,
        prediction_length: int,
        channels: int,
        moving_average: int = 25,
        individual: bool = False,
    ) -> None:
        super().__init__()
        if input_length <= 0 or prediction_length <= 0 or channels <= 0:
            raise ValueError("input_length, prediction_length, and channels must be positive.")

        self.input_length = input_length
        self.prediction_length = prediction_length
        self.channels = channels
        self.individual = individual
        self.decomposition = SeriesDecomposition(moving_average)

        if individual:
            self.seasonal_projection = nn.ModuleList(
                nn.Linear(input_length, prediction_length) for _ in range(channels)
            )
            self.trend_projection = nn.ModuleList(
                nn.Linear(input_length, prediction_length) for _ in range(channels)
            )
        else:
            self.seasonal_projection = nn.Linear(input_length, prediction_length)
            self.trend_projection = nn.Linear(input_length, prediction_length)

    def forward(self, series: Tensor) -> Tensor:
        _validate_series(series)
        if series.shape[1] != self.input_length:
            raise ValueError(
                f"Expected {self.input_length} input time steps; "
                f"received {series.shape[1]}."
            )
        if series.shape[2] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels; received {series.shape[2]}."
            )

        seasonal, trend = self.decomposition(series)
        seasonal = seasonal.transpose(1, 2)
        trend = trend.transpose(1, 2)

        if self.individual:
            seasonal_forecast = torch.stack(
                [
                    projection(seasonal[:, channel, :])
                    for channel, projection in enumerate(self.seasonal_projection)
                ],
                dim=1,
            )
            trend_forecast = torch.stack(
                [
                    projection(trend[:, channel, :])
                    for channel, projection in enumerate(self.trend_projection)
                ],
                dim=1,
            )
        else:
            seasonal_forecast = self.seasonal_projection(seasonal)
            trend_forecast = self.trend_projection(trend)

        return (seasonal_forecast + trend_forecast).transpose(1, 2)


__all__ = ["DLinear", "MovingAverage", "SeriesDecomposition"]
