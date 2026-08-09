# Dynamic Multiscale DLinear for Weather Forecasting

Final project for the Time-Series Analysis course. We reconstruct DLinear from
*Are Transformers Effective for Time Series Forecasting?* and improve its trend
decomposition on the public Weather dataset.

## Main files

| File | Purpose |
|---|---|
| `notebooks/01_dlinear_reconstruction.ipynb` | Stage 1: data, baseline, DLinear training, and paper comparison |
| `notebooks/02_dynamic_multiscale_improvement.ipynb` | Stage 2: improved architecture, selection, and final results |
| `notebooks/03_multihorizon_evaluation.ipynb` | Generalization across horizons 96, 192, 336, and 720 |
| `notebooks/04_results_and_figures.ipynb` | Optional report tables and figures |
| `src/ts_project/` | Tested data pipeline, models, metrics, baselines, and training code |
| `results/dlinear/weather/horizon_096/` | Final validation and test tables |
| `docs/final_method.md` | Mathematical description and experimental protocol |
| `report/` | Final report workspace |

Earlier ETTh1, PatchTST, and rejected improvement experiments are preserved in
`archive/etth1_patchtst/`. They are not part of the final execution path.

## Experiment

- supervised multivariate forecasting;
- Weather: 52,696 rows, 21 variables, 10-minute sampling;
- input length 336 and forecast horizon 96;
- chronological 70%/10%/20% train/validation/test split;
- `StandardScaler` fitted on training observations only;
- MSE, MAE, and RMSE;
- daily seasonal-naive baseline with period 144;
- seeds 2021, 2022, and 2023.

## Main result

| Model | Parameters | Test MSE (mean ± SD) | Test MAE (mean ± SD) |
|---|---:|---:|---:|
| DLinear reconstruction | 64,704 | 0.17464 ± 0.00052 | 0.23486 ± 0.00174 |
| Static per-variable multiscale DLinear | 64,767 | 0.16739 ± 0.00044 | 0.22910 ± 0.00114 |
| **Dynamic per-variable multiscale DLinear** | **64,834** | **0.16540 ± 0.00045** | **0.22492 ± 0.00137** |

The dynamic model reduces paired test MSE by approximately 5.29% and MAE by 4.23%, with
only 130 additional parameters.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Tested environment:

```text
Python 3.12.13
PyTorch 2.12.1+cu126
NumPy 2.5.1
pandas 2.3.3
scikit-learn 1.9.0
matplotlib 3.11.1
```

## Dataset

Download the Weather benchmark and place it at:

```text
data/raw/weather.csv
```

The dataset link, expected dimensions, and checksum are in `data/README.md`.

## Run

Open the notebooks in numerical order and select the project `.venv` kernel.
Notebook 1 trains DLinear with three seeds, notebook 2 trains the static and dynamic multiscale models with
the same paired seeds, and notebook 3 runs the optional repeated-seed
multi-horizon experiment. On the project GPU, notebooks 1 and 2 take roughly
5 and 13 minutes, respectively. All project experiments run directly from the
notebooks.

Run tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
