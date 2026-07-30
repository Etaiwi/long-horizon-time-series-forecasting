"""Simple forecasting baselines that require no learned parameters."""

from __future__ import annotations

import torch
from torch import Tensor


def _validate_history(history: Tensor, prediction_length: int) -> None:
    """Validate a batch of multivariate forecasting inputs."""

    if not isinstance(history, Tensor):
        raise TypeError("history must be a PyTorch tensor.")
    if history.ndim != 3:
        raise ValueError(
            "history must have shape [batch, time, variables]; "
            f"received {tuple(history.shape)}."
        )
    if history.shape[1] == 0 or history.shape[2] == 0:
        raise ValueError("history must contain at least one time step and variable.")
    if prediction_length <= 0:
        raise ValueError("prediction_length must be positive.")
    if not torch.is_floating_point(history):
        raise TypeError("history must use floating-point values.")
    if not torch.isfinite(history).all():
        raise ValueError("history must contain only finite values.")


def last_value_forecast(history: Tensor, prediction_length: int) -> Tensor:
    """Repeat each variable's most recently observed value across the horizon."""

    _validate_history(history, prediction_length)
    latest = history[:, -1:, :]
    return latest.expand(-1, prediction_length, -1).clone()


def seasonal_naive_forecast(
    history: Tensor,
    prediction_length: int,
    *,
    season_length: int = 24,
) -> Tensor:
    """Repeat the most recently observed season until the horizon is filled."""

    _validate_history(history, prediction_length)
    if season_length <= 0:
        raise ValueError("season_length must be positive.")
    if history.shape[1] < season_length:
        raise ValueError(
            f"history contains {history.shape[1]} time steps, "
            f"but season_length is {season_length}."
        )

    recent_season = history[:, -season_length:, :]
    repetitions = (prediction_length + season_length - 1) // season_length
    repeated = recent_season.repeat(1, repetitions, 1)
    return repeated[:, :prediction_length, :].clone()


__all__ = ["last_value_forecast", "seasonal_naive_forecast"]
