"""Frequency-Adaptive Normalization (FAN) with a DLinear backbone."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ts_project.models.dlinear import DLinear, _validate_series


def dominant_frequency_component(series: Tensor, top_k: int) -> tuple[Tensor, Tensor]:
    """Return the residual and instance-wise top-k Fourier reconstruction.

    Frequencies are selected independently for every sample and channel, matching
    FAN's frequency residual learning operation.
    """

    _validate_series(series)
    frequency_count = series.shape[1] // 2 + 1
    if not 1 <= top_k <= frequency_count:
        raise ValueError(
            f"top_k must be between 1 and {frequency_count}; received {top_k}."
        )

    spectrum = torch.fft.rfft(series, dim=1)
    indices = torch.topk(spectrum.abs(), top_k, dim=1).indices
    mask = torch.zeros_like(spectrum)
    mask.scatter_(1, indices, 1)
    dominant = torch.fft.irfft(spectrum * mask, n=series.shape[1], dim=1)
    return series - dominant, dominant


class FrequencyPredictor(nn.Module):
    """Forecast removed frequency components with FAN's shared MLP."""

    def __init__(self, *, input_length: int, prediction_length: int) -> None:
        super().__init__()
        self.main_projection = nn.Sequential(
            nn.Linear(input_length, 64),
            nn.ReLU(),
        )
        self.output_projection = nn.Sequential(
            nn.Linear(input_length + 64, 128),
            nn.ReLU(),
            nn.Linear(128, prediction_length),
        )

    def forward(self, dominant: Tensor, original: Tensor) -> Tensor:
        dominant = dominant.transpose(1, 2)
        original = original.transpose(1, 2)
        encoded = self.main_projection(dominant)
        forecast = self.output_projection(torch.cat([encoded, original], dim=-1))
        return forecast.transpose(1, 2)


class FANDLinear(nn.Module):
    """Forecast residual dynamics with DLinear and evolving frequencies with FAN."""

    def __init__(
        self,
        *,
        input_length: int,
        prediction_length: int,
        channels: int,
        top_k: int = 4,
        moving_average: int = 25,
        individual: bool = False,
    ) -> None:
        super().__init__()
        if top_k > input_length // 2 + 1 or top_k > prediction_length // 2 + 1:
            raise ValueError("top_k must fit both input and prediction Fourier spectra.")

        self.input_length = input_length
        self.prediction_length = prediction_length
        self.channels = channels
        self.top_k = top_k

        # Build DLinear first so a paired seed starts from the same backbone
        # weights as the original implementation.
        self.backbone = DLinear(
            input_length=input_length,
            prediction_length=prediction_length,
            channels=channels,
            moving_average=moving_average,
            individual=individual,
        )
        self.frequency_predictor = FrequencyPredictor(
            input_length=input_length,
            prediction_length=prediction_length,
        )

    def forecast_components(self, series: Tensor) -> tuple[Tensor, Tensor]:
        _validate_series(series)
        if series.shape[1] != self.input_length:
            raise ValueError(
                f"Expected {self.input_length} input time steps; received {series.shape[1]}."
            )
        if series.shape[2] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels; received {series.shape[2]}."
            )

        residual, dominant = dominant_frequency_component(series, self.top_k)
        residual_forecast = self.backbone(residual)
        dominant_forecast = self.frequency_predictor(dominant, series)
        return residual_forecast, dominant_forecast

    def forward(self, series: Tensor) -> Tensor:
        residual, dominant = self.forecast_components(series)
        return residual + dominant

    def training_loss(
        self,
        series: Tensor,
        target: Tensor,
        *,
        prior_weight: float = 1.0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Paper objective: forecast MSE plus dominant-component guidance MSE."""

        if prior_weight < 0:
            raise ValueError("prior_weight must be non-negative.")
        residual_forecast, dominant_forecast = self.forecast_components(series)
        forecast = residual_forecast + dominant_forecast
        _, target_dominant = dominant_frequency_component(target, self.top_k)
        forecast_mse = nn.functional.mse_loss(forecast, target)
        dominant_mse = nn.functional.mse_loss(dominant_forecast, target_dominant)
        total = forecast_mse + prior_weight * dominant_mse
        return total, {
            "forecast_mse": forecast_mse.detach(),
            "dominant_mse": dominant_mse.detach(),
        }


__all__ = ["FANDLinear", "FrequencyPredictor", "dominant_frequency_component"]
