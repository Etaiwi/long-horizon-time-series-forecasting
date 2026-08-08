from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ts_project.training import evaluate_mse, seed_everything, train_forecaster


def _loader() -> DataLoader:
    inputs = torch.arange(32, dtype=torch.float32).reshape(8, 4, 1) / 32
    targets = inputs[:, -2:, :] * 2
    return DataLoader(TensorDataset(inputs, targets), batch_size=4)


def test_evaluate_mse_aggregates_all_forecast_values() -> None:
    class ZeroForecast(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.zeros(inputs.shape[0], 2, 1)

    loader = _loader()
    targets = torch.cat([batch_targets for _, batch_targets in loader])
    expected = torch.mean(targets**2).item()

    actual = evaluate_mse(ZeroForecast(), loader, device=torch.device("cpu"))

    assert actual == expected


def test_training_uses_official_halving_schedule_and_restores_best_model() -> None:
    seed_everything(2021)
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2), nn.Unflatten(1, (2, 1)))
    loader = _loader()

    result = train_forecaster(
        model,
        loader,
        loader,
        device=torch.device("cpu"),
        learning_rate=0.005,
        max_epochs=3,
        patience=3,
    )

    assert [row["learning_rate"] for row in result.history] == [
        0.005,
        0.005,
        0.0025,
    ]
    assert 1 <= result.best_epoch <= 3
    assert evaluate_mse(model, loader, device=torch.device("cpu")) == (
        result.best_validation_mse
    )


def test_seed_everything_repeats_torch_random_values() -> None:
    seed_everything(17)
    first = torch.rand(4)
    seed_everything(17)
    second = torch.rand(4)

    torch.testing.assert_close(first, second)
