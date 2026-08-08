"""Generate the three notebooks used in the final project."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def make_notebook(cells):
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python (time-series-project)",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
    return notebook


def reconstruction_notebook():
    return make_notebook(
        [
            md(
                """
                # 1. DLinear reconstruction

                In this notebook we reconstruct DLinear on the Weather dataset and
                compare our result with the paper.

                - **Task:** supervised multivariate forecasting
                - **Sampling frequency:** 10 minutes
                - **Input:** 336 observations (56 hours)
                - **Output:** 96 observations (16 hours)
                - **Variables:** 21 weather measurements

                The notebook follows the full first stage of the project: data checks,
                preprocessing, temporal split, a simple baseline, training, and evaluation.
                """
            ),
            md("## 1. Setup"),
            code(
                """
                from pathlib import Path
                import sys

                import matplotlib.pyplot as plt
                import numpy as np
                import pandas as pd
                import torch
                from torch.utils.data import DataLoader

                PROJECT_ROOT = Path.cwd()
                while not (PROJECT_ROOT / "pyproject.toml").exists():
                    PROJECT_ROOT = PROJECT_ROOT.parent
                sys.path.insert(0, str(PROJECT_ROOT / "src"))

                from ts_project.baselines import seasonal_naive_forecast
                from ts_project.data import prepare_weather, build_weather_window_datasets
                from ts_project.models import DLinear
                from ts_project.training import seed_everything, train_forecaster

                DATA_PATH = PROJECT_ROOT / "data" / "raw" / "weather.csv"
                INPUT_LENGTH = 336
                HORIZON = 96
                BATCH_SIZE = 16
                SEED = 2021
                DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                print("Device:", DEVICE)
                """
            ),
            md(
                """
                ## 2. Load and check the data

                The loader parses the date column, converts every forecasting variable to
                a numeric value, and checks for missing or infinite values. We also inspect
                the time index before deciding whether any cleaning is needed.
                """
            ),
            code(
                """
                weather = prepare_weather(DATA_PATH)
                time_differences = weather.raw.index.to_series().diff().dropna()

                audit = pd.Series(
                    {
                        "rows": len(weather.raw),
                        "variables": len(weather.channel_names),
                        "start": weather.raw.index.min(),
                        "end": weather.raw.index.max(),
                        "missing values": int(weather.raw.isna().sum().sum()),
                        "infinite values": int(np.isinf(weather.raw.to_numpy()).sum()),
                        "duplicate timestamps": int(weather.raw.index.duplicated().sum()),
                        "most common interval": time_differences.mode().iloc[0],
                        "non-10-minute intervals": int((time_differences.dt.total_seconds() != 600).sum()),
                    },
                    name="Weather audit",
                )
                audit
                """
            ),
            md(
                """
                The supplied benchmark contains one duplicate timestamp and one gap. We do
                not resample or fill them because the paper's implementation treats the file
                as an equally spaced row sequence. Keeping it unchanged gives the fairest
                comparison with the paper. No extra calendar features are used by DLinear.
                """
            ),
            md("## 3. Chronological split and scaling"),
            code(
                """
                split_table = pd.DataFrame(
                    [
                        {
                            "split": name,
                            "rows": len(weather.split(name)),
                            "start": weather.split(name).index.min(),
                            "end": weather.split(name).index.max(),
                        }
                        for name in ("train", "validation", "test")
                    ]
                )
                display(split_table)

                print("Mean of scaled training variables:",
                      abs(weather.split("train", scaled=True).mean()).max())
                """
            ),
            md(
                """
                We use the official chronological 70%/10%/20% split. `StandardScaler` is
                fitted only on the training observations and then applied to validation and
                test. This prevents future information from entering preprocessing.
                """
            ),
            md("## 4. Forecasting windows"),
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
            code(
                """
                example_input, example_target = datasets["validation"][0]
                channel = weather.channel_names.index("OT")

                plt.figure(figsize=(11, 3.5))
                plt.plot(range(INPUT_LENGTH), example_input[:, channel], label="input")
                plt.plot(
                    range(INPUT_LENGTH, INPUT_LENGTH + HORIZON),
                    example_target[:, channel],
                    label="future target",
                )
                plt.axvline(INPUT_LENGTH, color="black", linestyle="--", linewidth=1)
                plt.title("One standardized OT forecasting window")
                plt.xlabel("10-minute step")
                plt.ylabel("standardized OT")
                plt.legend()
                plt.show()
                """
            ),
            md(
                """
                Each window uses the previous 336 rows to predict the following 96 rows.
                Validation and test inputs may use earlier observations as context, but all
                forecast targets remain inside their own split.
                """
            ),
            md(
                """
                ## 5. Seasonal-naive baseline

                Weather is sampled six times per hour, so one day contains 144 observations.
                The seasonal-naive forecast repeats the most recent observed day. It has no
                learned parameters and gives us a simple reference point.
                """
            ),
            code(
                """
                @torch.inference_mode()
                def evaluate_predictions(predict_batch, loader):
                    squared_sum = absolute_sum = 0.0
                    count = 0
                    for inputs, targets in loader:
                        predictions = predict_batch(inputs, targets)
                        errors = predictions - targets
                        squared_sum += errors.square().sum().item()
                        absolute_sum += errors.abs().sum().item()
                        count += errors.numel()
                    mse = squared_sum / count
                    return {"MSE": mse, "MAE": absolute_sum / count, "RMSE": mse**0.5}

                baseline_metrics = evaluate_predictions(
                    lambda inputs, targets: seasonal_naive_forecast(
                        inputs, HORIZON, season_length=144
                    ),
                    loaders["test"],
                )
                baseline_metrics
                """
            ),
            md(
                """
                ## 6. DLinear

                DLinear first separates every input channel into a moving-average trend and
                a remainder:

                \\[
                T=MA_{25}(X), \\qquad R=X-T.
                \\]

                Two shared linear layers map the full 336-step history directly to all 96
                future steps, and their forecasts are added:

                \\[
                \\hat{Y}=W_RR+W_TT.
                \\]

                The training objective is mean squared error. We use Adam, learning rate
                0.0001, at most 10 epochs, and early stopping with patience 3. The best
                checkpoint is selected by validation MSE.
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
                print("Trainable parameters:", sum(p.numel() for p in model.parameters()))
                """
            ),
            md("## 7. Train the reconstruction"),
            code(
                """
                seed_everything(SEED)
                training = train_forecaster(
                    model,
                    loaders["train"],
                    loaders["validation"],
                    device=DEVICE,
                    learning_rate=1e-4,
                    max_epochs=10,
                    patience=3,
                    verbose=True,
                )
                history = pd.DataFrame(training.history)
                history
                """
            ),
            code(
                """
                history.plot(
                    x="epoch",
                    y=["train_mse", "validation_mse"],
                    marker="o",
                    title="DLinear training history",
                    ylabel="MSE",
                )
                plt.show()
                """
            ),
            md(
                """
                ## 8. Reconstruction results

                MSE gives more weight to large errors. MAE is the mean absolute error, and
                RMSE is the square root of MSE. All three are calculated over every test
                window, forecast step, and variable in standardized space.
                """
            ),
            code(
                """
                model_metrics = evaluate_predictions(
                    lambda inputs, targets: model(inputs.to(DEVICE)).cpu(),
                    loaders["test"],
                )

                comparison = pd.DataFrame(
                    [
                        {"Model": "Paper DLinear", "MSE": 0.176, "MAE": 0.237, "RMSE": 0.176**0.5},
                        {"Model": "Seasonal naive", **baseline_metrics},
                        {"Model": "Our DLinear reconstruction", **model_metrics},
                    ]
                )
                comparison
                """
            ),
            md(
                """
                ## 9. Conclusion

                Our expected seed-2021 result is MSE 0.17421 and MAE 0.23333, compared
                with 0.176 and 0.237 in the paper. The small difference can come from the
                random seed, software version, and low-level GPU behavior. The reconstruction
                is close enough to use as the baseline for our improvement.
                """
            ),
        ]
    )


def improvement_notebook():
    return make_notebook(
        [
            md(
                """
                # 2. Dynamic multiscale DLinear improvement

                The reconstruction showed that our DLinear implementation matches the paper.
                We now change only the trend decomposition and keep the dataset, split,
                training procedure, forecast horizon, and metrics unchanged.
                """
            ),
            md(
                """
                ## 1. Motivation

                Original DLinear uses the same moving-average kernel, 25, for every Weather
                variable and every input window. Weather contains variables with different
                behavior, so one smoothing scale may not be suitable for all of them.

                Our question is: **can DLinear improve if the trend scale adapts to both the
                variable and the current input window?**
                """
            ),
            md(
                """
                ## 2. From DLinear to V1 and V2A

                We use three moving averages:

                - kernel 25: about 4.2 hours;
                - kernel 73: about 12.2 hours;
                - kernel 145: about 24.2 hours.

                **V1** learns one fixed softmax mixture for each variable.

                **V2A**, our final model, also changes the mixture for each input window. A
                small gate receives four values for each variable: the window mean, standard
                deviation, last value, and last minus first value. The gate output and a
                variable-specific prior are passed through softmax, so the three weights are
                positive and sum to one.

                The weighted trend and remainder are still forecast by DLinear's original
                two shared linear layers.
                """
            ),
            code(
                """
                from pathlib import Path
                import sys

                import matplotlib.pyplot as plt
                import numpy as np
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

                RESULT_ROOT = PROJECT_ROOT / "results" / "dlinear" / "weather" / "horizon_096"
                """
            ),
            md("## 3. Model size and a quick implementation check"),
            code(
                """
                models = {
                    "DLinear": DLinear(input_length=336, prediction_length=96, channels=21),
                    "V1": PerVariableMultiScaleDLinear(input_length=336, prediction_length=96, channels=21),
                    "V2A": DynamicPerVariableMultiScaleDLinear(input_length=336, prediction_length=96, channels=21),
                }
                parameters = pd.Series(
                    {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()},
                    name="parameters",
                )
                parameters
                """
            ),
            code(
                """
                example = torch.randn(2, 336, 21)
                weights = models["V2A"].scale_weights(example)

                print("Forecast shape:", tuple(models["V2A"](example).shape))
                print("Weight shape:", tuple(weights.shape))
                print("Smallest weight:", weights.min().item())
                print("Largest error in weight sums:", (weights.sum(dim=-1) - 1).abs().max().item())
                """
            ),
            md(
                """
                V2A has 64,834 parameters, only 130 more than DLinear. The check above also
                confirms that every sample and variable receives three valid scale weights.
                """
            ),
            md(
                """
                ## 4. Validation-based selection

                We first compared V1 and V2A on validation data using seeds 2021, 2022, and
                2023. Both models used the same split, scaler, optimizer, loss, early stopping,
                and metrics as the reconstruction. Test data was not used to choose between
                V1 and V2A.
                """
            ),
            code(
                """
                validation = pd.read_csv(RESULT_ROOT / "validation_by_seed.csv")
                validation
                """
            ),
            code(
                """
                validation.pivot(index="seed", columns="model", values="val_mse").plot(
                    marker="o",
                    ylabel="Validation MSE",
                    title="Validation MSE for V1 and V2A",
                )
                plt.show()
                """
            ),
            md(
                """
                V2A improves validation MSE and MAE for all three seeds, so we selected it
                as the final architecture before evaluating its test checkpoints.
                """
            ),
            md("## 5. Final test results"),
            code(
                """
                test_by_seed = pd.read_csv(RESULT_ROOT / "test_by_seed.csv")
                test_summary = pd.read_csv(RESULT_ROOT / "test_summary.csv")
                display(test_by_seed)
                display(test_summary)
                """
            ),
            code(
                """
                final_comparison = pd.DataFrame(
                    [
                        {"Model": "Paper DLinear", "MSE": 0.176, "MAE": 0.237, "Parameters": np.nan},
                        {"Model": "Seasonal naive", "MSE": 0.3167066, "MAE": 0.2879560, "Parameters": 0},
                        {"Model": "Our DLinear", "MSE": 0.1746375, "MAE": 0.2348639, "Parameters": 64704},
                        {"Model": "V1 static multiscale", "MSE": 0.1673884, "MAE": 0.2290998, "Parameters": 64767},
                        {"Model": "V2A dynamic multiscale", "MSE": 0.1654027, "MAE": 0.2249218, "Parameters": 64834},
                    ]
                )
                final_comparison["RMSE"] = np.sqrt(final_comparison["MSE"])
                final_comparison
                """
            ),
            md(
                """
                Across the three paired seeds, V2A reduces test MSE by about 5.29% and MAE
                by about 4.23% compared with reconstructed DLinear. It wins on both metrics
                for all three seeds.
                """
            ),
            md("## 6. Reproducing a run"),
            md(
                """
                The same tested runner is used for the reconstruction and improvement:

                ```powershell
                .\\.venv\\Scripts\\python.exe scripts\\run_weather_dlinear.py --model dlinear --seed 2021 --evaluate-test
                .\\.venv\\Scripts\\python.exe scripts\\run_weather_dlinear.py --model v2a --seed 2021 --evaluate-test
                ```

                Repeating both commands with seeds 2022 and 2023 reproduces the repeated-seed
                comparison. Generated run files are written to `outputs/weather_dlinear/`.
                """
            ),
            md(
                """
                ## 7. Discussion and limitations

                The main result supports our hypothesis on Weather: variable-specific scales
                help, and adapting them to the current window helps further. The improvement
                is also small in model size and easy to interpret.

                The result should not be described as a universal improvement. Earlier tests
                showed smaller gains on Electricity and ETTh1. We also tried several ideas on
                the same Weather validation split, so some validation-selection bias is
                possible. V1's test result was already known before V2A was designed, although
                V2A itself was frozen using validation results before its test evaluation.
                """
            ),
        ]
    )


def figures_notebook():
    return make_notebook(
        [
            md(
                """
                # 3. Final tables and figures

                This optional notebook only creates material for the report. It does not
                train models or make any model-selection decisions.
                """
            ),
            code(
                """
                from pathlib import Path

                import matplotlib.pyplot as plt
                import numpy as np
                import pandas as pd

                PROJECT_ROOT = Path.cwd()
                while not (PROJECT_ROOT / "pyproject.toml").exists():
                    PROJECT_ROOT = PROJECT_ROOT.parent

                RESULT_ROOT = PROJECT_ROOT / "results" / "dlinear" / "weather" / "horizon_096"
                FIGURE_ROOT = PROJECT_ROOT / "outputs" / "report_figures"
                FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

                validation = pd.read_csv(RESULT_ROOT / "validation_by_seed.csv")
                test_by_seed = pd.read_csv(RESULT_ROOT / "test_by_seed.csv")
                test_summary = pd.read_csv(RESULT_ROOT / "test_summary.csv")
                """
            ),
            md("## 1. Final comparison table"),
            code(
                """
                report_table = pd.DataFrame(
                    [
                        ["Paper DLinear", 0.176000, 0.237000, np.nan],
                        ["Seasonal naive", 0.316707, 0.287956, 0],
                        ["Our DLinear", 0.174638, 0.234864, 64704],
                        ["V1 static multiscale", 0.167388, 0.229100, 64767],
                        ["V2A dynamic multiscale", 0.165403, 0.224922, 64834],
                    ],
                    columns=["Model", "MSE", "MAE", "Parameters"],
                )
                report_table["RMSE"] = np.sqrt(report_table["MSE"])
                report_table
                """
            ),
            md("## 2. Validation selection figure"),
            code(
                """
                axis = validation.pivot(index="seed", columns="model", values="val_mse").plot(
                    marker="o", figsize=(7, 4), ylabel="Validation MSE"
                )
                axis.set_title("Validation MSE across three seeds")
                axis.figure.tight_layout()
                axis.figure.savefig(FIGURE_ROOT / "validation_mse_by_seed.png", dpi=200)
                plt.show()
                """
            ),
            md("## 3. Test comparison figure"),
            code(
                """
                plot_data = test_summary.set_index("model")
                figure, axes = plt.subplots(1, 2, figsize=(10, 4))
                plot_data["test_mse_mean"].plot(
                    kind="bar", yerr=plot_data["test_mse_std"], capsize=4, ax=axes[0]
                )
                plot_data["test_mae_mean"].plot(
                    kind="bar", yerr=plot_data["test_mae_std"], capsize=4, ax=axes[1]
                )
                axes[0].set_title("Test MSE")
                axes[1].set_title("Test MAE")
                for axis in axes:
                    axis.set_xlabel("")
                    axis.tick_params(axis="x", rotation=20)
                figure.tight_layout()
                figure.savefig(FIGURE_ROOT / "test_metrics.png", dpi=200)
                plt.show()
                """
            ),
            md(
                """
                The saved figures are optional report assets. The numerical CSV files under
                `results/` remain the source of truth.
                """
            ),
        ]
    )


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    outputs = {
        "01_dlinear_reconstruction.ipynb": reconstruction_notebook(),
        "02_dynamic_multiscale_improvement.ipynb": improvement_notebook(),
        "03_results_and_figures.ipynb": figures_notebook(),
    }
    for name, notebook in outputs.items():
        nbf.write(notebook, NOTEBOOKS / name)
    print("Built:", ", ".join(outputs))


if __name__ == "__main__":
    main()
