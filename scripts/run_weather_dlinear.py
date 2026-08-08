"""Train and evaluate the final Weather DLinear models reproducibly."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ts_project.data import build_weather_window_datasets, prepare_weather
from ts_project.models import (
    DLinear,
    DynamicPerVariableMultiScaleDLinear,
    PerVariableMultiScaleDLinear,
    initialize_projections_from_dlinear,
)
from ts_project.training import seed_everything, train_forecaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("dlinear", "v1", "v2a"), required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/raw/weather.csv")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    squared_sum = 0.0
    absolute_sum = 0.0
    elements = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        errors = model(inputs) - targets
        squared_sum += errors.square().sum().item()
        absolute_sum += errors.abs().sum().item()
        elements += errors.numel()
    if elements == 0:
        raise RuntimeError("The evaluation loader is empty.")
    return {"mse": squared_sum / elements, "mae": absolute_sum / elements}


def build_model(name: str, channels: int, seed: int) -> nn.Module:
    seed_everything(seed)
    original = DLinear(
        input_length=336,
        prediction_length=96,
        channels=channels,
        moving_average=25,
        individual=False,
    )
    if name == "dlinear":
        return original

    seed_everything(seed)
    if name == "v1":
        improved: nn.Module = PerVariableMultiScaleDLinear(
            input_length=336,
            prediction_length=96,
            channels=channels,
            kernel_sizes=(25, 73, 145),
        )
    else:
        improved = DynamicPerVariableMultiScaleDLinear(
            input_length=336,
            prediction_length=96,
            channels=channels,
            kernel_sizes=(25, 73, 145),
            hidden_dimension=8,
        )
    initialize_projections_from_dlinear(improved, original)
    return improved


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    data = prepare_weather(args.data)
    datasets = build_weather_window_datasets(
        data,
        input_length=336,
        prediction_length=96,
    )
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=16, shuffle=True, drop_last=True),
        "validation": DataLoader(
            datasets["validation"], batch_size=16, shuffle=True, drop_last=True
        ),
        "test": DataLoader(datasets["test"], batch_size=16, shuffle=False, drop_last=False),
    }

    model = build_model(args.model, len(data.channel_names), args.seed)
    seed_everything(args.seed)
    training = train_forecaster(
        model,
        loaders["train"],
        loaders["validation"],
        device=device,
        learning_rate=1e-4,
        learning_rate_schedule="type1",
        max_epochs=10,
        patience=3,
        verbose=True,
    )

    validation = evaluate(model, loaders["validation"], device)
    result: dict[str, object] = {
        "model": args.model,
        "seed": args.seed,
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": training.best_epoch,
        "validation": validation,
    }
    if args.evaluate_test:
        result["test"] = evaluate(model, loaders["test"], device)

    output_dir = PROJECT_ROOT / "outputs/weather_dlinear"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.model}_seed_{args.seed}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
