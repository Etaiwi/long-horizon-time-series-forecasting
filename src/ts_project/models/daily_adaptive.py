"""Daily cross-period and channel-adaptive extensions for DLinear."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ts_project.models.dlinear import DLinear, SeriesDecomposition, _validate_series


class CrossPeriodDailyBranch(nn.Module):
    """Forecast each hour-of-day across days with one shared linear mapping."""

    def __init__(
        self,
        *,
        input_length: int,
        prediction_length: int,
        period: int = 24,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if input_length <= 0 or prediction_length <= 0 or period <= 0:
            raise ValueError("input_length, prediction_length, and period must be positive.")
        if input_length % period or prediction_length % period:
            raise ValueError("input_length and prediction_length must be divisible by period.")

        self.input_length = input_length
        self.prediction_length = prediction_length
        self.period = period
        self.input_cycles = input_length // period
        self.output_cycles = prediction_length // period
        self.projection = nn.Linear(self.input_cycles, self.output_cycles, bias=False)
        if zero_init:
            nn.init.zeros_(self.projection.weight)

    def forward(self, series: Tensor) -> Tensor:
        _validate_series(series)
        if series.shape[1] != self.input_length:
            raise ValueError(
                f"Expected {self.input_length} input time steps; received {series.shape[1]}."
            )

        batch, _, channels = series.shape
        phases = series.reshape(batch, self.input_cycles, self.period, channels)
        phases = phases.permute(0, 2, 3, 1)
        forecast = self.projection(phases)
        return (
            forecast.permute(0, 3, 1, 2)
            .contiguous()
            .reshape(batch, self.prediction_length, channels)
        )


class CrossPeriodDailyDLinear(nn.Module):
    """Add a zero-initialized cross-day residual forecast to DLinear."""

    def __init__(self, backbone: DLinear, *, period: int = 24) -> None:
        super().__init__()
        self.backbone = backbone
        self.daily_branch = CrossPeriodDailyBranch(
            input_length=backbone.input_length,
            prediction_length=backbone.prediction_length,
            period=period,
            zero_init=True,
        )

    def forward(self, series: Tensor) -> Tensor:
        return self.backbone(series) + self.daily_branch(series)


class ChannelLowRankAdapter(nn.Module):
    """Channel-specific low-rank correction to a shared temporal projection."""

    def __init__(
        self,
        *,
        input_length: int,
        prediction_length: int,
        channels: int,
        rank: int = 1,
    ) -> None:
        super().__init__()
        if input_length <= 0 or prediction_length <= 0 or channels <= 0 or rank <= 0:
            raise ValueError("All dimensions and rank must be positive.")
        self.down = nn.Parameter(torch.empty(channels, rank, input_length))
        self.up = nn.Parameter(torch.zeros(channels, prediction_length, rank))
        nn.init.normal_(self.down, mean=0.0, std=0.02)

    def forward(self, component: Tensor) -> Tensor:
        latent = torch.einsum("bci,cri->bcr", component, self.down)
        return torch.einsum("bcr,cpr->bcp", latent, self.up)


class LowRankDLinear(nn.Module):
    """DLinear with rank-limited channel-specific temporal corrections."""

    def __init__(
        self,
        *,
        input_length: int,
        prediction_length: int,
        channels: int,
        moving_average: int = 25,
        rank: int = 1,
    ) -> None:
        super().__init__()
        if input_length <= 0 or prediction_length <= 0 or channels <= 0:
            raise ValueError("input_length, prediction_length, and channels must be positive.")

        self.input_length = input_length
        self.prediction_length = prediction_length
        self.channels = channels
        self.rank = rank
        self.decomposition = SeriesDecomposition(moving_average)

        # These are constructed in the same order as shared DLinear so paired
        # seeded runs begin from the same shared temporal weights.
        self.seasonal_projection = nn.Linear(input_length, prediction_length)
        self.trend_projection = nn.Linear(input_length, prediction_length)
        self.seasonal_adapter = ChannelLowRankAdapter(
            input_length=input_length,
            prediction_length=prediction_length,
            channels=channels,
            rank=rank,
        )
        self.trend_adapter = ChannelLowRankAdapter(
            input_length=input_length,
            prediction_length=prediction_length,
            channels=channels,
            rank=rank,
        )

    def forward(self, series: Tensor) -> Tensor:
        _validate_series(series)
        if series.shape[1] != self.input_length:
            raise ValueError(
                f"Expected {self.input_length} input time steps; received {series.shape[1]}."
            )
        if series.shape[2] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels; received {series.shape[2]}."
            )

        seasonal, trend = self.decomposition(series)
        seasonal = seasonal.transpose(1, 2)
        trend = trend.transpose(1, 2)
        forecast = (
            self.seasonal_projection(seasonal)
            + self.seasonal_adapter(seasonal)
            + self.trend_projection(trend)
            + self.trend_adapter(trend)
        )
        return forecast.transpose(1, 2)


class DailyLowRankDLinear(nn.Module):
    """Combine low-rank channel adaptation with a daily residual branch."""

    def __init__(self, backbone: LowRankDLinear, *, period: int = 24) -> None:
        super().__init__()
        self.backbone = backbone
        self.daily_branch = CrossPeriodDailyBranch(
            input_length=backbone.input_length,
            prediction_length=backbone.prediction_length,
            period=period,
            zero_init=True,
        )

    def forward(self, series: Tensor) -> Tensor:
        return self.backbone(series) + self.daily_branch(series)


__all__ = [
    "ChannelLowRankAdapter",
    "CrossPeriodDailyBranch",
    "CrossPeriodDailyDLinear",
    "DailyLowRankDLinear",
    "LowRankDLinear",
]
