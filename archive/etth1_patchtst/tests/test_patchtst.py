from __future__ import annotations

import pytest
import torch

from ts_project.models import PatchTST


def _model() -> PatchTST:
    return PatchTST(
        input_length=336,
        prediction_length=96,
        channels=7,
    )


def test_patchtst_42_configuration_and_output_shape() -> None:
    model = _model()
    inputs = torch.randn(2, 336, 7)

    forecast = model(inputs)

    assert model.patch_count == 42
    assert forecast.shape == (2, 96, 7)


def test_patchtst_matches_official_parameter_count() -> None:
    model = _model()

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert trainable_parameters == 81_728


def test_patchtst_is_channel_independent() -> None:
    torch.manual_seed(2021)
    model = _model().eval()
    original = torch.zeros(1, 336, 7)
    changed = original.clone()
    changed[:, :, 0] = torch.linspace(-1, 1, 336)

    with torch.no_grad():
        original_forecast = model(original)
        changed_forecast = model(changed)

    torch.testing.assert_close(
        original_forecast[:, :, 1:],
        changed_forecast[:, :, 1:],
    )


def test_patchtst_rejects_wrong_input_length() -> None:
    with pytest.raises(ValueError, match="Expected 336 input time steps"):
        _model()(torch.zeros(1, 335, 7))
