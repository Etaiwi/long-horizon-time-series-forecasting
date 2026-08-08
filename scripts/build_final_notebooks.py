"""Build the two concise, sectioned notebooks used by the final project."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "final"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def notebook(cells):
    document = nbf.v4.new_notebook(cells=cells)
    document.metadata["kernelspec"] = {
        "display_name": "Python (time-series-project)",
        "language": "python",
        "name": "python3",
    }
    document.metadata["language_info"] = {"name": "python", "version": "3.12"}
    return document


def build_reconstruction():
    return notebook(
        [
            markdown(
                """
                # 1. DLinear reconstruction on Weather

                This notebook reconstructs the method and evaluation protocol from
                *Are Transformers Effective for Time Series Forecasting?*

                By the end we will have checked the dataset, chronological split,
                train-only scaling, daily seasonal-naive baseline, DLinear architecture,
                training objective, paper metrics, and our reproduced test result.
                """
            ),
            markdown(
                """
                ## Section 1 — Imports and reproducible configuration

                The reusable implementation lives under `src/ts_project/`. Keeping the
                data pipeline, model, and training loop outside the notebook prevents
                accidental differences between the reconstruction and improvement.
                """
            ),
            code(
                """
                from pathlib import Path
                import sys

                import pandas as pd
                import torch
                from torch.utils.data import DataLoader

                PROJECT_ROOT = Path.cwd()
                while not (PROJECT_ROOT / "pyproject.toml").exists():
                    PROJECT_ROOT = PROJECT_ROOT.parent
                sys.path.insert(0, str(PROJECT_ROOT / "src"))

                from ts_project.data import prepare_weather, build_weather_window_datasets
                from ts_project.models import DLinear
                from ts_project.training import seed_everything, train_forecaster

                DATA_PATH = PROJECT_ROOT / "data" / "raw" / "weather.csv"
                INPUT_LENGTH = 336
                HORIZON = 96
                BATCH_SIZE = 16
                SEED = 2021
                DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                DEVICE
                """
            ),
            markdown(
                """
                ## Section 2 — Load and audit Weather

                Weather contains 21 meteorological variables sampled every ten minutes.
                The supplied benchmark has one duplicate timestamp and one gap; we keep
                its row order unchanged to match the paper's benchmark representation.
                """
            ),
            code(
                """
                weather = prepare_weather(DATA_PATH)
                split_summary = pd.DataFrame(
                    [
                        {
                            "split": name,
                            "start": weather.split(name).index.min(),
                            "end": weather.split(name).index.max(),
                            "rows": len(weather.split(name)),
                        }
                        for name in ("train", "validation", "test")
                    ]
                )
                print(f"Rows: {len(weather.raw):,}")
                print(f"Variables: {len(weather.channel_names)}")
                display(split_summary)
                """
            ),
            markdown(
                """
                ## Section 3 — Leakage-safe forecasting windows

                Each example uses 336 past observations (56 hours) to predict all 96
                future observations (16 hours) directly. Validation and test may use
                earlier values as historical context, but every target stays inside its
                own chronological partition. The scaler was fitted only on training rows.
                """
            ),
            code(
                """
                datasets = build_weather_window_datasets(
                    weather,
                    input_length=INPUT_LENGTH,
                    prediction_length=HORIZON,
                )
                loaders = {
                    "train": DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True, drop_last=True),
                    "validation": DataLoader(datasets["validation"], batch_size=BATCH_SIZE, shuffle=True, drop_last=True),
                    "test": DataLoader(datasets["test"], batch_size=BATCH_SIZE, shuffle=False, drop_last=False),
                }
                pd.Series({name: len(dataset) for name, dataset in datasets.items()}, name="windows")
                """
            ),
            markdown(
                """
                ## Section 4 — Original DLinear

                DLinear estimates a trend with a centered 25-sample moving average and
                defines the remainder as input minus trend. Two shared linear layers map
                the complete 336-step trend and remainder directly to the 96-step future;
                the forecasts are added. Training minimizes MSE with Adam.
                """
            ),
            code(
                """
                seed_everything(SEED)
                model = DLinear(
                    input_length=INPUT_LENGTH,
                    prediction_length=HORIZON,
                    channels=len(weather.channel_names),
                    moving_average=25,
                    individual=False,
                )
                parameter_count = sum(parameter.numel() for parameter in model.parameters())
                print(f"Trainable parameters: {parameter_count:,}")
                """
            ),
            markdown(
                """
                ## Section 5 — Train and select by validation MSE

                This cell takes roughly one to two minutes on the project GPU. Early
                stopping restores the checkpoint with the lowest validation MSE. The test
                set is not available to the training function.
                """
            ),
            code(
                """
                seed_everything(SEED)
                training = train_forecaster(
                    model,
                    loaders["train"],
                    loaders["validation"],
                    device=DEVICE,
                    learning_rate=1e-4,
                    learning_rate_schedule="type1",
                    max_epochs=10,
                    patience=3,
                    verbose=True,
                )
                pd.DataFrame(training.history)
                """
            ),
            markdown(
                """
                ## Section 6 — Evaluate MSE and MAE

                MSE penalizes large errors quadratically; MAE reports the average absolute
                error. Both are computed over every test window, future step, and variable
                in standardized space, matching the paper.
                """
            ),
            code(
                """
                @torch.inference_mode()
                def evaluate(model, loader):
                    squared_sum = absolute_sum = 0.0
                    elements = 0
                    model.eval()
                    for inputs, targets in loader:
                        errors = model(inputs.to(DEVICE)) - targets.to(DEVICE)
                        squared_sum += errors.square().sum().item()
                        absolute_sum += errors.abs().sum().item()
                        elements += errors.numel()
                    return {"MSE": squared_sum / elements, "MAE": absolute_sum / elements}

                measured = evaluate(model, loaders["test"])
                comparison = pd.DataFrame(
                    [
                        {"result": "Paper DLinear", "MSE": 0.176, "MAE": 0.237},
                        {"result": "Our reconstruction", **measured},
                    ]
                )
                comparison
                """
            ),
            markdown(
                """
                ## Section 7 — Reconstruction conclusion

                The expected seed-2021 result is approximately MSE 0.17421 and MAE
                0.23333, very close to the paper's 0.176 and 0.237. This establishes a
                faithful baseline before changing DLinear's decomposition.
                """
            ),
        ]
    )


def build_improvement():
    return notebook(
        [
            markdown(
                """
                # 2. Dynamic multiscale DLinear improvement

                This notebook explains the final improvement, follows its validation-only
                selection, and analyzes the frozen three-seed test results.
                """
            ),
            markdown(
                """
                ## Section 1 — Limitation and hypothesis

                Original DLinear applies the same 25-sample trend scale to all 21 Weather
                variables and all input windows. Our hypothesis is that different variables
                and current regimes require different smoothing scales.

                We test a clear progression:

                1. DLinear: one fixed global scale.
                2. V1: one learned scale mixture per variable.
                3. V2A: one learned mixture per variable **and input window**.
                """
            ),
            markdown(
                """
                ## Section 2 — Architecture

                V2A computes trends with kernels 25, 73, and 145. For each window and
                variable, a small gate receives the window mean, standard deviation, last
                value, and last-minus-first change. A softmax converts the gate and static
                variable prior into positive weights summing to one.

                The weighted trend, remainder, and two temporal forecasting heads remain
                DLinear-style. See `docs/final_method.md` for the equations.
                """
            ),
            code(
                """
                from pathlib import Path
                import sys

                import matplotlib.pyplot as plt
                import pandas as pd
                import torch

                PROJECT_ROOT = Path.cwd()
                while not (PROJECT_ROOT / "pyproject.toml").exists():
                    PROJECT_ROOT = PROJECT_ROOT.parent
                sys.path.insert(0, str(PROJECT_ROOT / "src"))

                from ts_project.models import (
                    DLinear,
                    PerVariableMultiScaleDLinear,
                    DynamicPerVariableMultiScaleDLinear,
                )

                models = {
                    "DLinear": DLinear(input_length=336, prediction_length=96, channels=21),
                    "V1": PerVariableMultiScaleDLinear(input_length=336, prediction_length=96, channels=21),
                    "V2A": DynamicPerVariableMultiScaleDLinear(input_length=336, prediction_length=96, channels=21),
                }
                pd.Series(
                    {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()},
                    name="parameters",
                )
                """
            ),
            markdown(
                """
                ## Section 3 — Validation-only architecture selection

                V1 and V2A were trained with seeds 2021–2023 under the same optimizer,
                early stopping, split, and metrics. V2A beat V1 on validation MSE and MAE
                for every seed, so it was frozen before V2A test evaluation.
                """
            ),
            code(
                """
                RESULT_ROOT = PROJECT_ROOT / "results" / "dlinear" / "weather" / "horizon_096"
                validation = pd.read_csv(RESULT_ROOT / "validation_by_seed.csv")
                display(validation)

                validation.pivot(index="seed", columns="model", values="val_mse").plot(
                    marker="o", ylabel="Validation MSE", title="Validation selection across seeds"
                )
                plt.show()
                """
            ),
            markdown(
                """
                ## Section 4 — Frozen test results

                After freezing V2A, all three saved checkpoints were evaluated. Repeated
                seeds measure sensitivity to initialization; they do not remove possible
                validation-selection bias.
                """
            ),
            code(
                """
                test_by_seed = pd.read_csv(RESULT_ROOT / "test_by_seed.csv")
                test_summary = pd.read_csv(RESULT_ROOT / "test_summary.csv")
                display(test_by_seed)
                display(test_summary)

                figure, axes = plt.subplots(1, 2, figsize=(11, 4))
                for metric, axis in zip(("test_mse", "test_mae"), axes):
                    test_by_seed.pivot(index="seed", columns="model", values=metric).plot(
                        kind="bar", ax=axis, title=metric.upper(), rot=0
                    )
                    axis.set_ylabel(metric.upper())
                figure.tight_layout()
                plt.show()
                """
            ),
            markdown(
                """
                ## Section 5 — Interpretation

                V2A improves over paired DLinear for all three seeds. Mean test reductions
                are about 5.29% MSE and 4.23% MAE, with only 130 additional parameters.
                V1's intermediate gain is an important ablation: variable-specific scales
                help, and adapting them to the current window helps further.
                """
            ),
            markdown(
                """
                ## Section 6 — Reproduce training

                The source modules and one command per model are the authoritative training
                path. Running all three seeds requires six paired DLinear/V2A runs.

                ```powershell
                .\\.venv\\Scripts\\python.exe scripts\\run_weather_dlinear.py --model dlinear --seed 2021 --evaluate-test
                .\\.venv\\Scripts\\python.exe scripts\\run_weather_dlinear.py --model v2a --seed 2021 --evaluate-test
                ```

                Repeat with seeds 2022 and 2023. Generated run JSON files are written under
                `outputs/weather_dlinear/`; the frozen report tables are under `results/`.
                """
            ),
            markdown(
                """
                ## Section 7 — Limitations

                The strongest gain is Weather-specific. Multiple ideas were inspected on
                the same validation period, and V1 test behavior was known before V2A was
                designed. We therefore claim a lightweight improvement on this Weather
                task, not universal superiority across all datasets.
                """
            ),
        ]
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    nbf.write(build_reconstruction(), OUTPUT / "01_dlinear_reconstruction.ipynb")
    nbf.write(build_improvement(), OUTPUT / "02_dynamic_multiscale_improvement.ipynb")
    print(f"Built final notebooks in {OUTPUT}")


if __name__ == "__main__":
    main()
