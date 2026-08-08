"""Reversible and slice-aware extensions that retain DLinear as the backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ts_project.models.dlinear import DLinear


@dataclass(frozen=True)
class PredictedStatistics:
    """Predicted future location and optional scale for auxiliary supervision."""

    mean: Tensor
    scale: Tensor | None = None


def _window_statistics(series: Tensor) -> tuple[Tensor, Tensor]:
    mean = series.mean(dim=1, keepdim=True).detach()
    scale = torch.sqrt(series.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    return mean, scale


class RevINDLinear(nn.Module):
    """Ordinary RevIN surrounding an otherwise unchanged DLinear model."""

    def __init__(self, backbone: DLinear) -> None:
        super().__init__()
        self.backbone = backbone
        channels = backbone.channels
        self.gamma = nn.Parameter(torch.ones(1, 1, channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, channels))

    def forward(self, series: Tensor) -> Tensor:
        mean, scale = _window_statistics(series)
        normalized = (series - mean) / scale
        affine = normalized * self.gamma + self.beta
        forecast = self.backbone(affine)
        return ((forecast - self.beta) / (self.gamma + 1e-8)) * scale + mean


class SliceStatisticsHead(nn.Module):
    """Map observed slice statistics to statistics for future slices per channel."""

    def __init__(
        self,
        *,
        input_length: int,
        prediction_length: int,
        channels: int,
        slice_length: int = 24,
        predict_scale: bool = False,
    ) -> None:
        super().__init__()
        if input_length % slice_length or prediction_length % slice_length:
            raise ValueError("input_length and prediction_length must divide by slice_length.")
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.channels = channels
        self.slice_length = slice_length
        self.input_slices = input_length // slice_length
        self.output_slices = prediction_length // slice_length
        self.predict_scale = predict_scale
        self.mean_layers = nn.ModuleList(
            nn.Linear(self.input_slices, self.output_slices) for _ in range(channels)
        )
        self.scale_layers = (
            nn.ModuleList(
                nn.Linear(self.input_slices, self.output_slices) for _ in range(channels)
            )
            if predict_scale
            else None
        )

    def forward(self, series: Tensor, mean: Tensor, scale: Tensor) -> PredictedStatistics:
        batch = series.shape[0]
        slices = series.reshape(
            batch, self.input_slices, self.slice_length, self.channels
        )
        observed_means = slices.mean(dim=2)
        standardized_means = (
            observed_means - mean.squeeze(1).unsqueeze(1)
        ) / scale.squeeze(1).unsqueeze(1)
        future_mean_z = torch.stack(
            [layer(standardized_means[:, :, c]) for c, layer in enumerate(self.mean_layers)],
            dim=2,
        )
        future_means = (
            mean.squeeze(1).unsqueeze(1)
            + scale.squeeze(1).unsqueeze(1) * future_mean_z
        ).repeat_interleave(self.slice_length, dim=1)

        if self.scale_layers is None:
            return PredictedStatistics(mean=future_means)

        observed_scales = torch.sqrt(slices.var(dim=2, unbiased=False) + 1e-5)
        observed_log_ratios = torch.log(
            observed_scales / scale.squeeze(1).unsqueeze(1).clamp_min(1e-5)
        )
        future_log_ratios = torch.stack(
            [layer(observed_log_ratios[:, :, c]) for c, layer in enumerate(self.scale_layers)],
            dim=2,
        )
        future_scales = (
            scale.squeeze(1).unsqueeze(1)
            * torch.exp(future_log_ratios.clamp(-3.0, 3.0))
        ).repeat_interleave(self.slice_length, dim=1)
        return PredictedStatistics(mean=future_means, scale=future_scales)


class SliceAwareDLinear(nn.Module):
    """Forecast normalized patterns with DLinear and future baselines by slice."""

    def __init__(self, backbone: DLinear, *, slice_length: int = 24, predict_scale: bool = False) -> None:
        super().__init__()
        self.backbone = backbone
        self.statistics_head = SliceStatisticsHead(
            input_length=backbone.input_length,
            prediction_length=backbone.prediction_length,
            channels=backbone.channels,
            slice_length=slice_length,
            predict_scale=predict_scale,
        )

    def forward_with_statistics(self, series: Tensor) -> tuple[Tensor, PredictedStatistics]:
        mean, scale = _window_statistics(series)
        pattern = self.backbone((series - mean) / scale)
        predicted = self.statistics_head(series, mean, scale)
        future_scale = predicted.scale if predicted.scale is not None else scale
        return pattern * future_scale + predicted.mean, predicted

    def forward(self, series: Tensor) -> Tensor:
        return self.forward_with_statistics(series)[0]


class GatedSliceAwareDLinear(nn.Module):
    """Learn a per-channel blend of raw and slice-aware DLinear predictions."""

    def __init__(self, raw: DLinear, slice_aware: SliceAwareDLinear) -> None:
        super().__init__()
        if raw.channels != slice_aware.backbone.channels:
            raise ValueError("Raw and slice-aware backbones must have the same channels.")
        self.raw = raw
        self.slice_aware = slice_aware
        self.gate_logits = nn.Parameter(torch.zeros(raw.channels))

    @property
    def gates(self) -> Tensor:
        return torch.sigmoid(self.gate_logits)

    def forward_with_statistics(self, series: Tensor) -> tuple[Tensor, PredictedStatistics]:
        slice_forecast, predicted = self.slice_aware.forward_with_statistics(series)
        gate = self.gates.view(1, 1, -1)
        forecast = (1.0 - gate) * self.raw(series) + gate * slice_forecast
        return forecast, predicted

    def forward(self, series: Tensor) -> Tensor:
        return self.forward_with_statistics(series)[0]


def slice_statistics_loss(
    predicted: PredictedStatistics,
    target: Tensor,
    *,
    slice_length: int = 24,
) -> Tensor:
    """Supervise predicted future slice means and, when present, scales."""

    if target.shape[1] % slice_length:
        raise ValueError("The target horizon must divide by slice_length.")
    slices = target.reshape(
        target.shape[0], target.shape[1] // slice_length, slice_length, target.shape[2]
    )
    target_means = slices.mean(dim=2).repeat_interleave(slice_length, dim=1)
    loss = torch.mean((predicted.mean - target_means) ** 2)
    if predicted.scale is not None:
        target_scales = torch.sqrt(slices.var(dim=2, unbiased=False) + 1e-5)
        target_scales = target_scales.repeat_interleave(slice_length, dim=1)
        loss = loss + torch.mean(
            (
                torch.log(predicted.scale.clamp_min(1e-5))
                - torch.log(target_scales.clamp_min(1e-5))
            )
            ** 2
        )
    return loss


__all__ = [
    "GatedSliceAwareDLinear",
    "PredictedStatistics",
    "RevINDLinear",
    "SliceAwareDLinear",
    "SliceStatisticsHead",
    "slice_statistics_loss",
]
