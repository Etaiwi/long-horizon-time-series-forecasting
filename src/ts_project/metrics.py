"""Forecasting metrics shared by every baseline and learned model."""

from __future__ import annotations

import torch
from torch import Tensor


def _validate_pair(predictions: Tensor, targets: Tensor) -> None:
    """Fail clearly when two tensors cannot represent matching forecasts."""

    if not isinstance(predictions, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("predictions and targets must both be PyTorch tensors.")
    if predictions.shape != targets.shape:
        raise ValueError(
            "predictions and targets must have the same shape; "
            f"received {tuple(predictions.shape)} and {tuple(targets.shape)}."
        )
    if predictions.numel() == 0:
        raise ValueError("predictions and targets must not be empty.")
    if not torch.is_floating_point(predictions) or not torch.is_floating_point(targets):
        raise TypeError("predictions and targets must use floating-point values.")
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError("predictions and targets must contain only finite values.")


def mse(predictions: Tensor, targets: Tensor) -> Tensor:
    """Return mean squared error; larger mistakes receive a quadratic penalty."""

    _validate_pair(predictions, targets)
    return torch.mean((predictions - targets) ** 2)


def mae(predictions: Tensor, targets: Tensor) -> Tensor:
    """Return mean absolute error in the same scale as the supplied values."""

    _validate_pair(predictions, targets)
    return torch.mean(torch.abs(predictions - targets))


def rmse(predictions: Tensor, targets: Tensor) -> Tensor:
    """Return root mean squared error in the same scale as the supplied values."""

    return torch.sqrt(mse(predictions, targets))


__all__ = ["mae", "mse", "rmse"]
