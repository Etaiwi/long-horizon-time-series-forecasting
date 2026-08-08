"""Static and dynamic per-variable multiscale improvements to DLinear."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ts_project.models.dlinear import DLinear, MovingAverage, _validate_series


def _validate_kernels(kernel_sizes: Sequence[int]) -> tuple[int, ...]:
    kernels = tuple(int(kernel) for kernel in kernel_sizes)
    if not kernels:
        raise ValueError("At least one moving-average kernel is required.")
    if any(kernel <= 0 or kernel % 2 == 0 for kernel in kernels):
        raise ValueError("Every moving-average kernel must be a positive odd integer.")
    if len(set(kernels)) != len(kernels):
        raise ValueError("Moving-average kernels must be unique.")
    return kernels


class PerVariableMultiScaleDLinear(nn.Module):
    """V1: learn one fixed convex trend-scale mixture per variable."""

    def __init__(
        self,
        *,
        input_length: int,
        prediction_length: int,
        channels: int,
        kernel_sizes: Sequence[int] = (25, 73, 145),
    ) -> None:
        super().__init__()
        if input_length <= 0 or prediction_length <= 0 or channels <= 0:
            raise ValueError("Input length, prediction length, and channels must be positive.")

        self.input_length = int(input_length)
        self.prediction_length = int(prediction_length)
        self.channels = int(channels)
        self.kernel_sizes = _validate_kernels(kernel_sizes)
        self.moving_averages = nn.ModuleList(
            MovingAverage(kernel) for kernel in self.kernel_sizes
        )
        self.seasonal_projection = nn.Linear(input_length, prediction_length)
        self.trend_projection = nn.Linear(input_length, prediction_length)
        self.scale_logits = nn.Parameter(
            torch.zeros(channels, len(self.kernel_sizes))
        )

    def scale_weights(self) -> Tensor:
        return torch.softmax(self.scale_logits, dim=-1)

    def decompose(self, series: Tensor) -> tuple[Tensor, Tensor]:
        _validate_series(series)
        if series.shape[1:] != (self.input_length, self.channels):
            raise ValueError(
                f"Expected [batch, {self.input_length}, {self.channels}]; "
                f"received {tuple(series.shape)}."
            )

        trends = torch.stack(
            [moving_average(series) for moving_average in self.moving_averages],
            dim=0,
        )  # [scales, batch, time, channels]
        weights = self.scale_weights().transpose(0, 1)[:, None, None, :]
        trend = torch.sum(weights * trends, dim=0)
        return series - trend, trend

    def forward(self, series: Tensor) -> Tensor:
        remainder, trend = self.decompose(series)
        remainder_forecast = self.seasonal_projection(remainder.transpose(1, 2))
        trend_forecast = self.trend_projection(trend.transpose(1, 2))
        return (remainder_forecast + trend_forecast).transpose(1, 2)


class DynamicPerVariableMultiScaleDLinear(nn.Module):
    """V2A: condition each variable's trend scales on the current window.

    For every input window and variable, a small shared gate receives the
    standardized window mean, population standard deviation, last value, and
    last-minus-first change. Its correction is added to a variable-specific
    static prior before a softmax produces the three convex scale weights.
    """

    def __init__(
        self,
        *,
        input_length: int,
        prediction_length: int,
        channels: int,
        kernel_sizes: Sequence[int] = (25, 73, 145),
        hidden_dimension: int = 8,
    ) -> None:
        super().__init__()
        if input_length <= 0 or prediction_length <= 0 or channels <= 0:
            raise ValueError("Input length, prediction length, and channels must be positive.")
        if hidden_dimension <= 0:
            raise ValueError("hidden_dimension must be positive.")

        self.input_length = int(input_length)
        self.prediction_length = int(prediction_length)
        self.channels = int(channels)
        self.kernel_sizes = _validate_kernels(kernel_sizes)
        self.hidden_dimension = int(hidden_dimension)

        self.moving_averages = nn.ModuleList(
            MovingAverage(kernel) for kernel in self.kernel_sizes
        )
        self.seasonal_projection = nn.Linear(input_length, prediction_length)
        self.trend_projection = nn.Linear(input_length, prediction_length)

        self.base_scale_logits = nn.Parameter(
            torch.zeros(channels, len(self.kernel_sizes))
        )
        self.gate_hidden = nn.Linear(4, hidden_dimension)
        self.gate_output = nn.Linear(hidden_dimension, len(self.kernel_sizes))
        nn.init.zeros_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)

    @staticmethod
    def summary_features(series: Tensor) -> Tensor:
        return torch.stack(
            [
                series.mean(dim=1),
                series.std(dim=1, unbiased=False),
                series[:, -1, :],
                series[:, -1, :] - series[:, 0, :],
            ],
            dim=-1,
        )

    def scale_weights(self, series: Tensor) -> Tensor:
        features = self.summary_features(series)
        dynamic_logits = self.gate_output(
            torch.nn.functional.gelu(self.gate_hidden(features))
        )
        return torch.softmax(
            self.base_scale_logits.unsqueeze(0) + dynamic_logits,
            dim=-1,
        )

    def decompose(self, series: Tensor) -> tuple[Tensor, Tensor]:
        _validate_series(series)
        if series.shape[1:] != (self.input_length, self.channels):
            raise ValueError(
                f"Expected [batch, {self.input_length}, {self.channels}]; "
                f"received {tuple(series.shape)}."
            )

        trends = torch.stack(
            [moving_average(series) for moving_average in self.moving_averages],
            dim=1,
        )  # [batch, scales, time, channels]
        weights = self.scale_weights(series).permute(0, 2, 1).unsqueeze(2)
        trend = torch.sum(weights * trends, dim=1)
        return series - trend, trend

    def forward(self, series: Tensor) -> Tensor:
        remainder, trend = self.decompose(series)
        remainder_forecast = self.seasonal_projection(remainder.transpose(1, 2))
        trend_forecast = self.trend_projection(trend.transpose(1, 2))
        return (remainder_forecast + trend_forecast).transpose(1, 2)


def initialize_projections_from_dlinear(
    improved: PerVariableMultiScaleDLinear | DynamicPerVariableMultiScaleDLinear,
    original: DLinear,
) -> None:
    """Copy the paired shared DLinear temporal projections."""

    if original.individual:
        raise ValueError("The source DLinear model must use individual=False.")
    if not isinstance(original.seasonal_projection, nn.Linear):
        raise TypeError("Expected a shared DLinear seasonal projection.")
    if not isinstance(original.trend_projection, nn.Linear):
        raise TypeError("Expected a shared DLinear trend projection.")
    improved.seasonal_projection.load_state_dict(
        original.seasonal_projection.state_dict()
    )
    improved.trend_projection.load_state_dict(original.trend_projection.state_dict())


__all__ = [
    "DynamicPerVariableMultiScaleDLinear",
    "PerVariableMultiScaleDLinear",
    "initialize_projections_from_dlinear",
]
