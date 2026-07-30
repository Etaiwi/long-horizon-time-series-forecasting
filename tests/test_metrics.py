from __future__ import annotations

import pytest
import torch

from ts_project.metrics import mae, mse, rmse


def test_metrics_match_hand_calculation() -> None:
    predictions = torch.tensor([0.0, 2.0])
    targets = torch.tensor([1.0, 0.0])

    assert mse(predictions, targets).item() == pytest.approx(2.5)
    assert mae(predictions, targets).item() == pytest.approx(1.5)
    assert rmse(predictions, targets).item() == pytest.approx(2.5**0.5)


def test_metrics_preserve_gradient_information() -> None:
    predictions = torch.tensor([0.0, 2.0], requires_grad=True)
    targets = torch.tensor([1.0, 0.0])

    loss = mse(predictions, targets)
    loss.backward()

    torch.testing.assert_close(predictions.grad, torch.tensor([-1.0, 2.0]))


def test_metrics_reject_mismatched_shapes() -> None:
    predictions = torch.zeros(2, 3)
    targets = torch.zeros(2, 4)

    with pytest.raises(ValueError, match="same shape"):
        mse(predictions, targets)
