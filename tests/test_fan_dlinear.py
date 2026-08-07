from __future__ import annotations

import pytest
import torch

from ts_project.models.fan_dlinear import FANDLinear, dominant_frequency_component


def test_frequency_components_reconstruct_input() -> None:
    torch.manual_seed(4)
    series = torch.randn(3, 24, 2)

    residual, dominant = dominant_frequency_component(series, top_k=3)

    torch.testing.assert_close(residual + dominant, series)


def test_frequency_selection_is_instance_and_channel_specific() -> None:
    time = torch.arange(24, dtype=torch.float32)
    series = torch.stack(
        [
            torch.sin(2 * torch.pi * time / 24),
            torch.sin(2 * torch.pi * time / 6),
        ],
        dim=-1,
    ).unsqueeze(0)

    residual, _ = dominant_frequency_component(series, top_k=1)

    assert residual.square().mean().item() < 1e-10


def test_fan_dlinear_outputs_direct_multichannel_forecast() -> None:
    model = FANDLinear(
        input_length=24,
        prediction_length=12,
        channels=2,
        top_k=3,
        moving_average=3,
    )

    forecast = model(torch.randn(4, 24, 2))

    assert forecast.shape == (4, 12, 2)


def test_training_loss_backpropagates_through_both_branches() -> None:
    model = FANDLinear(
        input_length=24,
        prediction_length=12,
        channels=2,
        top_k=3,
        moving_average=3,
    )
    loss, parts = model.training_loss(
        torch.randn(4, 24, 2),
        torch.randn(4, 12, 2),
    )

    loss.backward()

    assert set(parts) == {"forecast_mse", "dominant_mse"}
    assert model.backbone.seasonal_projection.weight.grad is not None
    assert model.frequency_predictor.output_projection[-1].weight.grad is not None


def test_top_k_must_fit_input_and_output_spectra() -> None:
    with pytest.raises(ValueError, match="fit both"):
        FANDLinear(
            input_length=24,
            prediction_length=4,
            channels=2,
            top_k=4,
        )
