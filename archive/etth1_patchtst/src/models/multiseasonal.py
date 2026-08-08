"""Period-aware DLinear with explicit daily and weekly seasonal components."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ts_project.models.dlinear import SeriesDecomposition, _validate_series


class MultiSeasonalDecomposition(nn.Module):
    """Split a series into irregular, daily, weekly, and trend components.

    The daily template is the mean detrended value at each hour of the day. The
    weekly template is the mean remaining value at each hour of the week after
    removing that daily template. Consequently, the four returned components
    add back to the input exactly.
    """

    def __init__(
        self,
        *,
        input_length: int,
        moving_average: int = 25,
        daily_period: int = 24,
        weekly_period: int = 168,
    ) -> None:
        super().__init__()
        if input_length <= 0:
            raise ValueError("input_length must be positive.")
        if daily_period <= 0 or weekly_period <= 0:
            raise ValueError("daily_period and weekly_period must be positive.")
        if weekly_period % daily_period:
            raise ValueError("weekly_period must be divisible by daily_period.")
        if input_length % weekly_period:
            raise ValueError("input_length must be divisible by weekly_period.")

        self.input_length = input_length
        self.daily_period = daily_period
        self.weekly_period = weekly_period
        self.trend_decomposition = SeriesDecomposition(moving_average)

    @staticmethod
    def _phase_average(component: Tensor, period: int) -> tuple[Tensor, Tensor]:
        batch, length, channels = component.shape
        template = component.reshape(batch, length // period, period, channels).mean(dim=1)
        expanded = template.repeat(1, length // period, 1)
        return expanded, template

    def forward(self, series: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        _validate_series(series)
        if series.shape[1] != self.input_length:
            raise ValueError(
                f"Expected {self.input_length} input time steps; received {series.shape[1]}."
            )

        detrended, trend = self.trend_decomposition(series)
        daily, _ = self._phase_average(detrended, self.daily_period)
        daily_adjusted = detrended - daily
        weekly, _ = self._phase_average(daily_adjusted, self.weekly_period)
        irregular = daily_adjusted - weekly
        return irregular, daily, weekly, trend


class PeriodAwareDLinear(nn.Module):
    """Forecast trend/remainder linearly and extrapolate daily/weekly templates."""

    def __init__(
        self,
        *,
        input_length: int,
        prediction_length: int,
        channels: int,
        moving_average: int = 25,
        daily_period: int = 24,
        weekly_period: int = 168,
        individual: bool = False,
    ) -> None:
        super().__init__()
        if input_length <= 0 or prediction_length <= 0 or channels <= 0:
            raise ValueError("input_length, prediction_length, and channels must be positive.")

        self.input_length = input_length
        self.prediction_length = prediction_length
        self.channels = channels
        self.daily_period = daily_period
        self.weekly_period = weekly_period
        self.individual = individual
        self.decomposition = MultiSeasonalDecomposition(
            input_length=input_length,
            moving_average=moving_average,
            daily_period=daily_period,
            weekly_period=weekly_period,
        )

        if individual:
            self.irregular_projection = nn.ModuleList(
                nn.Linear(input_length, prediction_length) for _ in range(channels)
            )
            self.trend_projection = nn.ModuleList(
                nn.Linear(input_length, prediction_length) for _ in range(channels)
            )
        else:
            self.irregular_projection = nn.Linear(input_length, prediction_length)
            self.trend_projection = nn.Linear(input_length, prediction_length)

        # Start from ordinary additive seasonal extrapolation while allowing
        # training to suppress or rescale either period independently by channel.
        self.daily_strength = nn.Parameter(torch.ones(channels))
        self.weekly_strength = nn.Parameter(torch.ones(channels))

    def _project(self, component: Tensor, projections: nn.Module) -> Tensor:
        component = component.transpose(1, 2)
        if self.individual:
            return torch.stack(
                [
                    projection(component[:, channel, :])
                    for channel, projection in enumerate(projections)
                ],
                dim=1,
            ).transpose(1, 2)
        return projections(component).transpose(1, 2)

    def _future_template(self, component: Tensor, period: int) -> Tensor:
        template = component[:, :period, :]
        future_phases = (
            torch.arange(self.prediction_length, device=component.device)
            + self.input_length
        ) % period
        return template[:, future_phases, :]

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

        irregular, daily, weekly, trend = self.decomposition(series)
        irregular_forecast = self._project(irregular, self.irregular_projection)
        trend_forecast = self._project(trend, self.trend_projection)
        daily_forecast = self._future_template(daily, self.daily_period)
        weekly_forecast = self._future_template(weekly, self.weekly_period)

        return (
            irregular_forecast
            + trend_forecast
            + daily_forecast * self.daily_strength.view(1, 1, -1)
            + weekly_forecast * self.weekly_strength.view(1, 1, -1)
        )


__all__ = ["MultiSeasonalDecomposition", "PeriodAwareDLinear"]
