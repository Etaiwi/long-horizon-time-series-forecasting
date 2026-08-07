"""Run staged daily-adaptation DLinear experiments on ETTh1."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ts_project.data import FEATURE_COLUMNS, build_window_datasets, prepare_etth1
from ts_project.models import (
    CrossPeriodDailyDLinear,
    DailyLowRankDLinear,
    DLinear,
    LowRankDLinear,
    RevINDLinear,
)
from ts_project.training import seed_everything, train_forecaster

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "daily_adaptation_sequence.yaml"
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "ETTh1.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "dlinear" / "daily_adaptation_sequence"


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


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


def _metrics_from_sums(squared: Tensor, absolute: Tensor, count: int) -> dict[str, Any]:
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


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader[Any], *, device: torch.device) -> dict[str, Any]:
    model.eval()
    squared = torch.zeros(len(FEATURE_COLUMNS), dtype=torch.float64)
    absolute = torch.zeros(len(FEATURE_COLUMNS), dtype=torch.float64)
    count = 0
    for inputs, targets in loader:
        errors = model(inputs.to(device)).cpu() - targets
        squared += (errors**2).sum(dim=(0, 1), dtype=torch.float64)
        absolute += errors.abs().sum(dim=(0, 1), dtype=torch.float64)
        count += targets.shape[0] * targets.shape[1]
    if count == 0:
        raise ValueError("Cannot evaluate an empty loader.")
    return _metrics_from_sums(squared, absolute, count)


@torch.inference_mode()
def evaluate_masked(
    raw: nn.Module,
    normalized: nn.Module,
    loader: DataLoader[Any],
    *,
    mask: Tensor,
    device: torch.device,
) -> dict[str, Any]:
    raw.eval()
    normalized.eval()
    mask = mask.view(1, 1, -1)
    squared = torch.zeros(len(FEATURE_COLUMNS), dtype=torch.float64)
    absolute = torch.zeros(len(FEATURE_COLUMNS), dtype=torch.float64)
    count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        raw_prediction = raw(inputs).cpu()
        normalized_prediction = normalized(inputs).cpu()
        prediction = torch.where(mask, normalized_prediction, raw_prediction)
        errors = prediction - targets
        squared += (errors**2).sum(dim=(0, 1), dtype=torch.float64)
        absolute += errors.abs().sum(dim=(0, 1), dtype=torch.float64)
        count += targets.shape[0] * targets.shape[1]
    return _metrics_from_sums(squared, absolute, count)


def train_variant(
    name: str,
    builder: Callable[[], nn.Module],
    loaders: dict[str, DataLoader[Any]],
    *,
    seed: int,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    training = config["training"]
    seed_everything(seed)
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
        verbose=False,
    )
    runtime_seconds = time.perf_counter() - started
    result = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": training_result.best_epoch,
        "runtime_seconds": runtime_seconds,
        "history": training_result.history,
        "validation": evaluate(model, loaders["validation"], device=device),
        "test": evaluate(model, loaders["test"], device=device),
    }
    print(
        f"seed={seed} {name:28s} val={result['validation']['MSE']:.6f} "
        f"test={result['test']['MSE']:.6f}/{result['test']['MAE']:.6f}",
        flush=True,
    )
    return model, result


def builders_for_horizon(
    *,
    input_length: int,
    prediction_length: int,
    channels: int,
    moving_average: int,
    daily_period: int,
    adapter_rank: int,
) -> dict[str, Callable[[], nn.Module]]:
    common = {
        "input_length": input_length,
        "prediction_length": prediction_length,
        "channels": channels,
        "moving_average": moving_average,
    }

    def baseline() -> DLinear:
        return DLinear(**common, individual=False)

    def low_rank() -> LowRankDLinear:
        return LowRankDLinear(**common, rank=adapter_rank)

    return {
        "DLinear": baseline,
        "RevIN-DLinear": lambda: RevINDLinear(baseline()),
        "CrossPeriodDaily": lambda: CrossPeriodDailyDLinear(
            baseline(), period=daily_period
        ),
        "LowRankDLinear": low_rank,
        "DailyLowRank": lambda: DailyLowRankDLinear(
            low_rank(), period=daily_period
        ),
    }


def validation_mask(
    raw_result: dict[str, Any], normalized_result: dict[str, Any]
) -> Tensor:
    return torch.tensor(
        [
            normalized_result["validation"]["per_feature"][feature]["MSE"]
            < raw_result["validation"]["per_feature"][feature]["MSE"]
            for feature in FEATURE_COLUMNS
        ],
        dtype=torch.bool,
    )


def add_masked_result(
    records: dict[str, dict[str, Any]],
    models: dict[str, nn.Module],
    loaders: dict[str, DataLoader[Any]],
    *,
    device: torch.device,
) -> None:
    mask = validation_mask(records["DLinear"], records["RevIN-DLinear"])
    raw = models["DLinear"]
    normalized = models["RevIN-DLinear"]
    records["ValidationSelectedRawRevIN"] = {
        "parameters": records["DLinear"]["parameters"]
        + records["RevIN-DLinear"]["parameters"],
        "best_epoch": None,
        "runtime_seconds": records["DLinear"]["runtime_seconds"]
        + records["RevIN-DLinear"]["runtime_seconds"],
        "history": [],
        "selected_normalized_features": [
            feature for feature, selected in zip(FEATURE_COLUMNS, mask.tolist()) if selected
        ],
        "validation": evaluate_masked(
            raw, normalized, loaders["validation"], mask=mask, device=device
        ),
        "test": evaluate_masked(raw, normalized, loaders["test"], mask=mask, device=device),
    }


def metrics_with_feature_selection(
    raw_metrics: dict[str, Any],
    normalized_metrics: dict[str, Any],
    selected_normalized_features: list[str],
) -> dict[str, Any]:
    """Combine already-evaluated feature metrics using one frozen feature mask."""

    selected = set(selected_normalized_features)
    per_feature = {
        feature: copy.deepcopy(
            normalized_metrics["per_feature"][feature]
            if feature in selected
            else raw_metrics["per_feature"][feature]
        )
        for feature in FEATURE_COLUMNS
    }
    return {
        "MSE": statistics.mean(row["MSE"] for row in per_feature.values()),
        "MAE": statistics.mean(row["MAE"] for row in per_feature.values()),
        "per_feature": per_feature,
    }


def aggregate_validation_features(
    development_records: dict[str, dict[str, dict[str, Any]]]
) -> list[str]:
    """Select one normalization mask from mean validation MSE across seeds."""

    return [
        feature
        for feature in FEATURE_COLUMNS
        if statistics.mean(
            records["RevIN-DLinear"]["validation"]["per_feature"][feature]["MSE"]
            for records in development_records.values()
        )
        < statistics.mean(
            records["DLinear"]["validation"]["per_feature"][feature]["MSE"]
            for records in development_records.values()
        )
    ]


def add_consensus_result(
    records: dict[str, dict[str, Any]],
    selected_normalized_features: list[str],
) -> None:
    raw = records["DLinear"]
    normalized = records["RevIN-DLinear"]
    records["ConsensusRawRevIN"] = {
        "parameters": raw["parameters"] + normalized["parameters"],
        "best_epoch": None,
        "runtime_seconds": raw["runtime_seconds"] + normalized["runtime_seconds"],
        "history": [],
        "selected_normalized_features": selected_normalized_features,
        "validation": metrics_with_feature_selection(
            raw["validation"], normalized["validation"], selected_normalized_features
        ),
        "test": metrics_with_feature_selection(
            raw["test"], normalized["test"], selected_normalized_features
        ),
    }


def recompute_consensus_results(results: dict[str, Any]) -> dict[str, Any]:
    """Derive one frozen raw/RevIN mask from already completed paired runs."""

    development_horizon = results.get("selection", {}).get("development_horizon", 96)
    development_key = f"horizon_{development_horizon:03d}"
    development_records = results["runs"][development_key]
    for seed_records in development_records.values():
        seed_records.pop("ValidationSelectedRawRevIN", None)
    frozen_features = aggregate_validation_features(development_records)
    for seed_records in development_records.values():
        add_consensus_result(seed_records, frozen_features)

    baseline_mean = mean_validation(development_records, "DLinear")
    daily_mean = mean_validation(development_records, "CrossPeriodDaily")
    low_rank_mean = mean_validation(development_records, "LowRankDLinear")
    combine_ran = all("DailyLowRank" in row for row in development_records.values())
    candidate_names = ["CrossPeriodDaily", "LowRankDLinear", "ConsensusRawRevIN"]
    if combine_ran:
        candidate_names.append("DailyLowRank")
    candidate_validation = {
        name: mean_validation(development_records, name) for name in candidate_names
    }
    selected_candidate = min(candidate_validation, key=candidate_validation.get)
    results["selection"] = {
        "development_horizon": development_horizon,
        "baseline_mean_validation_MSE": baseline_mean,
        "candidate_mean_validation_MSE": candidate_validation,
        "daily_mean_validation_MSE": daily_mean,
        "low_rank_mean_validation_MSE": low_rank_mean,
        "combine_ran": combine_ran,
        "frozen_normalized_features": frozen_features,
        "selected_candidate": selected_candidate,
        "selected_candidate_beats_baseline": candidate_validation[selected_candidate]
        < baseline_mean,
    }

    if selected_candidate == "ConsensusRawRevIN":
        for horizon_key, horizon_records in results["runs"].items():
            if horizon_key == development_key:
                continue
            for seed_records in horizon_records.values():
                if "DLinear" not in seed_records or "RevIN-DLinear" not in seed_records:
                    raise ValueError(
                        f"Completed paired raw/RevIN runs are missing for {horizon_key}."
                    )
                add_consensus_result(seed_records, frozen_features)
    return results


def mean_validation(
    horizon_records: dict[str, dict[str, dict[str, Any]]], model_name: str
) -> float:
    return statistics.mean(
        seed_records[model_name]["validation"]["MSE"]
        for seed_records in horizon_records.values()
    )


def summary_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon_key, horizon_records in results["runs"].items():
        horizon = int(horizon_key.split("_")[-1])
        model_names = sorted(
            {model_name for seed_records in horizon_records.values() for model_name in seed_records}
        )
        for model_name in model_names:
            available = [
                seed_records[model_name]
                for seed_records in horizon_records.values()
                if model_name in seed_records
            ]
            rows.append(
                {
                    "horizon": horizon,
                    "model": model_name,
                    "seeds": len(available),
                    "validation_MSE_mean": statistics.mean(
                        row["validation"]["MSE"] for row in available
                    ),
                    "validation_MSE_std": statistics.stdev(
                        row["validation"]["MSE"] for row in available
                    )
                    if len(available) > 1
                    else 0.0,
                    "test_MSE_mean": statistics.mean(row["test"]["MSE"] for row in available),
                    "test_MSE_std": statistics.stdev(row["test"]["MSE"] for row in available)
                    if len(available) > 1
                    else 0.0,
                    "test_MAE_mean": statistics.mean(row["test"]["MAE"] for row in available),
                    "test_MAE_std": statistics.stdev(row["test"]["MAE"] for row in available)
                    if len(available) > 1
                    else 0.0,
                    "parameters": available[0]["parameters"],
                }
            )
    return rows


def per_feature_rows(results: dict[str, Any], horizon: int = 96) -> list[dict[str, Any]]:
    records = results["runs"][f"horizon_{horizon:03d}"]
    model_names = sorted(
        {model_name for seed_records in records.values() for model_name in seed_records}
    )
    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        available = [row[model_name] for row in records.values() if model_name in row]
        for feature in FEATURE_COLUMNS:
            rows.append(
                {
                    "horizon": horizon,
                    "model": model_name,
                    "feature": feature,
                    "test_MSE_mean": statistics.mean(
                        row["test"]["per_feature"][feature]["MSE"] for row in available
                    ),
                    "test_MAE_mean": statistics.mean(
                        row["test"]["per_feature"][feature]["MAE"] for row in available
                    ),
                }
            )
    return rows


def write_outputs(output: Path, results: dict[str, Any], config: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "sequence_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    summaries = summary_rows(results)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    features = per_feature_rows(results)
    with (output / "per_feature_h096.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=features[0].keys())
        writer.writeheader()
        writer.writerows(features)


def run_sequence(
    *,
    config_path: Path = DEFAULT_CONFIG,
    data_path: Path = DEFAULT_DATA,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "auto",
    reuse_completed_runs: bool = False,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    completed_path = output / "sequence_metrics.json"
    if reuse_completed_runs:
        if not completed_path.exists():
            raise FileNotFoundError(
                "reuse_completed_runs=True requires an existing sequence_metrics.json."
            )
        results = json.loads(completed_path.read_text(encoding="utf-8"))
        results = recompute_consensus_results(results)
        write_outputs(output, results, config)
        print(f"Reused completed paired runs and wrote consensus results to {output}")
        return results

    device = select_device(device_name)
    data = prepare_etth1(data_path)
    data_config = config["data"]
    model_config = config["models"]
    training = config["training"]
    seeds = config["experiment"]["seeds"]
    development_horizon = config["selection"]["development_horizon"]
    print(f"Device: {device}", flush=True)

    results: dict[str, Any] = {
        "device": str(device),
        "selection": {},
        "runs": {},
    }
    development_key = f"horizon_{development_horizon:03d}"
    development_records: dict[str, dict[str, Any]] = {}
    datasets = build_window_datasets(
        data,
        input_length=data_config["input_length"],
        prediction_length=development_horizon,
    )
    builders = builders_for_horizon(
        input_length=data_config["input_length"],
        prediction_length=development_horizon,
        channels=data_config["channels"],
        moving_average=model_config["moving_average"],
        daily_period=model_config["daily_period"],
        adapter_rank=model_config["adapter_rank"],
    )
    initial_names = ("DLinear", "RevIN-DLinear", "CrossPeriodDaily", "LowRankDLinear")
    for seed in seeds:
        loaders = make_loaders(datasets, batch_size=training["batch_size"], seed=seed)
        seed_records: dict[str, Any] = {}
        for name in initial_names:
            _, seed_records[name] = train_variant(
                name,
                builders[name],
                loaders,
                seed=seed,
                config=config,
                device=device,
            )
        seed_mask = validation_mask(seed_records["DLinear"], seed_records["RevIN-DLinear"])
        selected = [
            feature
            for feature, is_selected in zip(FEATURE_COLUMNS, seed_mask.tolist())
            if is_selected
        ]
        print(f"seed={seed} validation-selected RevIN features: {selected}", flush=True)
        development_records[str(seed)] = seed_records

    baseline_mean = mean_validation(development_records, "DLinear")
    daily_mean = mean_validation(development_records, "CrossPeriodDaily")
    low_rank_mean = mean_validation(development_records, "LowRankDLinear")
    combine_ran = daily_mean < baseline_mean and low_rank_mean < baseline_mean
    if combine_ran:
        for seed in seeds:
            loaders = make_loaders(datasets, batch_size=training["batch_size"], seed=seed)
            _, development_records[str(seed)]["DailyLowRank"] = train_variant(
                "DailyLowRank",
                builders["DailyLowRank"],
                loaders,
                seed=seed,
                config=config,
                device=device,
            )

    frozen_normalized_features = aggregate_validation_features(development_records)
    for seed_records in development_records.values():
        add_consensus_result(seed_records, frozen_normalized_features)

    candidate_names = [
        "CrossPeriodDaily",
        "LowRankDLinear",
        "ConsensusRawRevIN",
    ]
    if combine_ran:
        candidate_names.append("DailyLowRank")
    candidate_validation = {
        name: mean_validation(development_records, name) for name in candidate_names
    }
    selected_candidate = min(candidate_validation, key=candidate_validation.get)
    results["selection"] = {
        "development_horizon": development_horizon,
        "baseline_mean_validation_MSE": baseline_mean,
        "candidate_mean_validation_MSE": candidate_validation,
        "combine_ran": combine_ran,
        "frozen_normalized_features": frozen_normalized_features,
        "selected_candidate": selected_candidate,
        "selected_candidate_beats_baseline": candidate_validation[selected_candidate]
        < baseline_mean,
    }
    results["runs"][development_key] = development_records
    print(f"Frozen candidate after validation: {selected_candidate}", flush=True)

    for horizon in data_config["prediction_lengths"]:
        if horizon == development_horizon:
            continue
        horizon_key = f"horizon_{horizon:03d}"
        horizon_records: dict[str, dict[str, Any]] = {}
        datasets = build_window_datasets(
            data,
            input_length=data_config["input_length"],
            prediction_length=horizon,
        )
        builders = builders_for_horizon(
            input_length=data_config["input_length"],
            prediction_length=horizon,
            channels=data_config["channels"],
            moving_average=model_config["moving_average"],
            daily_period=model_config["daily_period"],
            adapter_rank=model_config["adapter_rank"],
        )
        for seed in seeds:
            loaders = make_loaders(datasets, batch_size=training["batch_size"], seed=seed)
            seed_records: dict[str, Any] = {}
            models: dict[str, nn.Module] = {}
            models["DLinear"], seed_records["DLinear"] = train_variant(
                "DLinear", builders["DLinear"], loaders, seed=seed, config=config, device=device
            )
            if selected_candidate == "ConsensusRawRevIN":
                models["RevIN-DLinear"], seed_records["RevIN-DLinear"] = train_variant(
                    "RevIN-DLinear",
                    builders["RevIN-DLinear"],
                    loaders,
                    seed=seed,
                    config=config,
                    device=device,
                )
                add_consensus_result(seed_records, frozen_normalized_features)
            else:
                _, seed_records[selected_candidate] = train_variant(
                    selected_candidate,
                    builders[selected_candidate],
                    loaders,
                    seed=seed,
                    config=config,
                    device=device,
                )
            horizon_records[str(seed)] = seed_records
        results["runs"][horizon_key] = horizon_records

    write_outputs(output, results, config)
    print(f"Wrote sequence results to {output}", flush=True)
    return results


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--reuse-completed-runs", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_sequence(
        config_path=args.config,
        data_path=args.data,
        output=args.output,
        device_name=args.device,
        reuse_completed_runs=args.reuse_completed_runs,
    )
