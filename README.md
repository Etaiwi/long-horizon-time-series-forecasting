# Dynamic Multiscale DLinear for Weather Forecasting

Time-Series Analysis final project by Etai Wigman and Guy Inbar. We reconstruct
DLinear from *Are Transformers Effective for Time Series Forecasting?* and
replace its fixed trend scale with a lightweight, input-dependent mixture of
three moving-average scales.

[Read the public final report](final_report.pdf).

## Project files

| Path | Purpose |
|---|---|
| `notebooks/01_dlinear_reconstruction.ipynb` | Data pipeline, seasonal-naive baseline, DLinear training, and paper reconstruction |
| `notebooks/02_dynamic_multiscale_improvement.ipynb` | Static and dynamic multiscale models, validation selection, and main test results |
| `notebooks/03_multihorizon_evaluation.ipynb` | Three-seed evaluation at horizons 96, 192, 336, and 720 |
| `notebooks/04_results_and_figures.ipynb` | Final result tables and report figures |
| `src/ts_project/` | Data preparation, forecasting models, metrics, baselines, and training utilities |
| `results/dlinear/weather/` | Saved numerical results used by the report |
| `data/README.md` | Dataset download instructions and checksum |

## Experiment

- supervised multivariate-to-multivariate forecasting;
- Weather dataset: 52,696 rows, 21 variables, 10-minute sampling;
- input length 336 and main forecast horizon 96;
- chronological 70%/10%/20% train/validation/test split;
- standardization fitted on training observations only;
- MSE and MAE evaluation;
- daily seasonal-naive baseline with period 144;
- paired seeds 2021, 2022, and 2023.

## Main result

| Model | Parameters | Test MSE (mean ± SD) | Test MAE (mean ± SD) |
|---|---:|---:|---:|
| DLinear reconstruction | 64,704 | 0.17464 ± 0.00052 | 0.23486 ± 0.00174 |
| Static per-variable multiscale DLinear | 64,767 | 0.16739 ± 0.00044 | 0.22910 ± 0.00114 |
| **Dynamic per-variable multiscale DLinear** | **64,834** | **0.16540 ± 0.00045** | **0.22492 ± 0.00137** |

The dynamic model reduces paired test MSE by 5.29% and MAE by 4.23% while
adding only 130 trainable parameters.

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

Download `weather.csv` using the links in `data/README.md`, then place it at:

```text
data/raw/weather.csv
```

## Reproduce the results

Open the four notebooks in numerical order and select the project `.venv`
kernel. Notebooks 1–3 train and evaluate the models; notebook 4 reads the saved
CSV results and recreates the final tables and figures. Completed numerical
outputs are written under `results/dlinear/weather/`, while generated report
figures are written under the ignored `outputs/report_figures/` directory.

On the project GPU, notebooks 1 and 2 take approximately 5 and 13 minutes,
respectively. The multi-horizon experiment in notebook 3 takes longer because
it trains both models at four forecast horizons.
