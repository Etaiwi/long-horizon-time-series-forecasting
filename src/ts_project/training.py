"""Compact, validation-based training utilities for forecasting models."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainingResult:
    """Training history and the validation-selected stopping point."""

    history: list[dict[str, float | int]]
    best_epoch: int
    best_validation_mse: float
    stopped_early: bool


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_mse_sum(predictions: Tensor, targets: Tensor) -> tuple[float, int]:
    squared_errors = (predictions - targets) ** 2
    return squared_errors.sum().item(), squared_errors.numel()


@torch.inference_mode()
def evaluate_mse(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
) -> float:
    """Evaluate elementwise MSE without updating the model."""

    model.eval()
    squared_error_sum = 0.0
    element_count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_sum, batch_count = _batch_mse_sum(model(inputs), targets)
        squared_error_sum += batch_sum
        element_count += batch_count

    if element_count == 0:
        raise ValueError("Cannot evaluate an empty data loader.")
    return squared_error_sum / element_count


def train_forecaster(
    model: nn.Module,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    *,
    device: torch.device,
    learning_rate: float = 0.005,
    max_epochs: int = 10,
    patience: int = 3,
) -> TrainingResult:
    """Train with Adam/MSE and restore the best validation checkpoint.

    The learning rate follows the official DLinear ``type1`` schedule: it is
    halved after every completed epoch. The test set is intentionally absent
    from this function so it cannot influence model selection.
    """

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if max_epochs <= 0 or patience <= 0:
        raise ValueError("max_epochs and patience must be positive.")
    if len(train_loader) == 0 or len(validation_loader) == 0:
        raise ValueError("Training and validation loaders must not be empty.")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    history: list[dict[str, float | int]] = []
    best_state: dict[str, Tensor] | None = None
    best_validation_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        squared_error_sum = 0.0
        element_count = 0
        current_learning_rate = float(optimizer.param_groups[0]["lr"])

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            batch_sum, batch_count = _batch_mse_sum(predictions.detach(), targets)
            squared_error_sum += batch_sum
            element_count += batch_count

        train_mse = squared_error_sum / element_count
        validation_mse = evaluate_mse(
            model,
            validation_loader,
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                "learning_rate": current_learning_rate,
                "train_mse": train_mse,
                "validation_mse": validation_mse,
            }
        )

        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

        next_learning_rate = learning_rate * (0.5**epoch)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = next_learning_rate

    if best_state is None:
        raise RuntimeError("Training completed without producing a checkpoint.")
    model.load_state_dict(best_state)

    return TrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_validation_mse=best_validation_mse,
        stopped_early=len(history) < max_epochs,
    )


__all__ = [
    "TrainingResult",
    "evaluate_mse",
    "seed_everything",
    "train_forecaster",
]
