"""Train paired DLinear and daily/weekly period-aware DLinear on ETTh1."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from ts_project.data import FEATURE_COLUMNS, build_window_datasets, prepare_etth1
from ts_project.models import DLinear, PeriodAwareDLinear
from ts_project.training import seed_everything, train_forecaster

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "multiseasonal_dlinear.yaml",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "ETTh1.csv",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    channels = len(FEATURE_COLUMNS)
    squared = torch.zeros(channels, dtype=torch.float64)
    absolute = torch.zeros(channels, dtype=torch.float64)
    count = 0
    for inputs, targets in loader:
        predictions = model(inputs.to(device)).cpu()
        errors = predictions - targets
        squared += (errors**2).sum(dim=(0, 1), dtype=torch.float64)
        absolute += errors.abs().sum(dim=(0, 1), dtype=torch.float64)
        count += targets.shape[0] * targets.shape[1]
    if count == 0:
        raise ValueError("Cannot evaluate an empty data loader.")

    per_feature_mse = squared / count
    per_feature_mae = absolute / count
    return {
        "MSE": per_feature_mse.mean().item(),
        "MAE": per_feature_mae.mean().item(),
        "per_feature": {
            feature: {
                "MSE": per_feature_mse[index].item(),
                "MAE": per_feature_mae[index].item(),
            }
            for index, feature in enumerate(FEATURE_COLUMNS)
        },
    }


def make_loaders(
    datasets: Any,
    *,
    batch_size: int,
    seed: int,
) -> dict[str, DataLoader[Any]]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        ),
        "validation": DataLoader(datasets["validation"], batch_size=batch_size),
        "test": DataLoader(datasets["test"], batch_size=batch_size),
    }


def train_variant(
    name: str,
    builder: Callable[[], nn.Module],
    datasets: Any,
    *,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    experiment = config["experiment"]
    training = config["training"]
    seed_everything(experiment["seed"])
    loaders = make_loaders(
        datasets,
        batch_size=training["batch_size"],
        seed=experiment["seed"],
    )
    model = builder()
    started = time.perf_counter()
    training_result = train_forecaster(
        model,
        loaders["train"],
        loaders["validation"],
        device=device,
        learning_rate=training["learning_rate"],
        learning_rate_schedule=training["learning_rate_schedule"],
        max_epochs=training["max_epochs"],
        patience=training["early_stopping_patience"],
        verbose=True,
    )
    runtime_seconds = time.perf_counter() - started
    result: dict[str, Any] = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": training_result.best_epoch,
        "runtime_seconds": runtime_seconds,
        "history": training_result.history,
        "validation": evaluate(model, loaders["validation"], device=device),
        "test": evaluate(model, loaders["test"], device=device),
    }
    if isinstance(model, PeriodAwareDLinear):
        result["seasonal_strengths"] = {
            feature: {
                "daily": model.daily_strength[index].detach().cpu().item(),
                "weekly": model.weekly_strength[index].detach().cpu().item(),
            }
            for index, feature in enumerate(FEATURE_COLUMNS)
        }
    print(
        f"{name}: validation MSE={result['validation']['MSE']:.6f}, "
        f"test MSE={result['test']['MSE']:.6f}, "
        f"test MAE={result['test']['MAE']:.6f}",
        flush=True,
    )
    return model, result


def write_results(
    output: Path,
    *,
    config: dict[str, Any],
    device: torch.device,
    results: dict[str, dict[str, Any]],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    stored_config = {**config, "runtime": {"device": str(device)}}
    (output / "config.yaml").write_text(
        yaml.safe_dump(stored_config, sort_keys=False), encoding="utf-8"
    )

    metrics = {
        name: {key: value for key, value in result.items() if key != "history"}
        for name, result in results.items()
    }
    baseline_mse = metrics["DLinear"]["test"]["MSE"]
    candidate_mse = metrics["PeriodAwareDLinear"]["test"]["MSE"]
    metrics["comparison"] = {
        "test_MSE_improvement_percent": 100.0 * (1.0 - candidate_mse / baseline_mse)
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("model", "epoch", "learning_rate", "train_mse", "validation_mse"),
        )
        writer.writeheader()
        for name, result in results.items():
            for row in result["history"]:
                writer.writerow({"model": name, **row})


def main() -> None:
    arguments = parse_arguments()
    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    device = select_device(arguments.device)
    data_config = config["data"]
    model_config = config["model"]
    seed = config["experiment"]["seed"]
    print(f"Device: {device}", flush=True)

    data = prepare_etth1(arguments.data)
    datasets = build_window_datasets(
        data,
        input_length=data_config["input_length"],
        prediction_length=data_config["prediction_length"],
    )

    common = {
        "input_length": data_config["input_length"],
        "prediction_length": data_config["prediction_length"],
        "channels": data_config["channels"],
        "moving_average": model_config["moving_average"],
        "individual": model_config["individual"],
    }
    builders: dict[str, Callable[[], nn.Module]] = {
        "DLinear": lambda: DLinear(**common),
        "PeriodAwareDLinear": lambda: PeriodAwareDLinear(
            **common,
            daily_period=model_config["daily_period"],
            weekly_period=model_config["weekly_period"],
        ),
    }
    results: dict[str, dict[str, Any]] = {}
    for name, builder in builders.items():
        _, results[name] = train_variant(
            name,
            builder,
            datasets,
            config=config,
            device=device,
        )

    output = (
        PROJECT_ROOT
        / "results"
        / "dlinear"
        / "multiseasonal"
        / "etth1"
        / f"horizon_{data_config['prediction_length']:03d}"
        / f"seed_{seed}"
    )
    write_results(output, config=config, device=device, results=results)
    print(f"Wrote results to {output}", flush=True)


if __name__ == "__main__":
    main()
