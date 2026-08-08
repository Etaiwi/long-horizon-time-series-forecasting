from __future__ import annotations

import torch

from ts_project.models import (
    DLinear,
    DynamicPerVariableMultiScaleDLinear,
    PerVariableMultiScaleDLinear,
    initialize_projections_from_dlinear,
)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_final_weather_parameter_counts() -> None:
    original = DLinear(
        input_length=336,
        prediction_length=96,
        channels=21,
        individual=False,
    )
    v1 = PerVariableMultiScaleDLinear(
        input_length=336,
        prediction_length=96,
        channels=21,
    )
    v2a = DynamicPerVariableMultiScaleDLinear(
        input_length=336,
        prediction_length=96,
        channels=21,
    )
    assert parameter_count(original) == 64_704
    assert parameter_count(v1) == 64_767
    assert parameter_count(v2a) == 64_834


def test_dynamic_weights_and_gradient_flow() -> None:
    torch.manual_seed(2021)
    original = DLinear(
        input_length=48,
        prediction_length=12,
        channels=3,
        individual=False,
    )
    model = DynamicPerVariableMultiScaleDLinear(
        input_length=48,
        prediction_length=12,
        channels=3,
        kernel_sizes=(5, 13, 25),
    )
    initialize_projections_from_dlinear(model, original)
    torch.testing.assert_close(
        model.seasonal_projection.weight,
        original.seasonal_projection.weight,
    )
    torch.testing.assert_close(
        model.trend_projection.weight,
        original.trend_projection.weight,
    )

    inputs = torch.randn(4, 48, 3)
    weights = model.scale_weights(inputs)
    assert weights.shape == (4, 3, 3)
    assert torch.all(weights > 0)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(4, 3))

    predictions = model(inputs)
    assert predictions.shape == (4, 12, 3)
    predictions.square().mean().backward()
    assert model.base_scale_logits.grad is not None
    assert model.gate_output.weight.grad is not None


def test_static_weights_are_convex() -> None:
    model = PerVariableMultiScaleDLinear(
        input_length=48,
        prediction_length=12,
        channels=3,
        kernel_sizes=(5, 13, 25),
    )
    weights = model.scale_weights()
    assert torch.all(weights > 0)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(3))
