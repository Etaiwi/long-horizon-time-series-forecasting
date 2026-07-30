# Long-Horizon Time-Series Forecasting: DLinear vs. PatchTST

This repository contains a reproducible reconstruction and improvement study for
long-horizon multivariate time-series forecasting.

## Current status

The project is in Part 1: paper reconstruction. Two candidate papers will be
evaluated on the same data pipeline before the primary project is selected:

1. [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504)
   - Candidate model: DLinear
   - [Official implementation](https://github.com/cure-lab/LTSF-Linear)
2. [A Time Series Is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730)
   - Candidate model: PatchTST
   - [Official implementation](https://github.com/yuqinie98/PatchTST)

The reconstruction will begin with DLinear. PatchTST will reuse the verified
dataset, splitting, scaling, windowing, metrics, and baseline code.

## Shared benchmark

Both candidates will use the public
[ETTh1 dataset](https://github.com/zhouhaoyi/ETDataset), an hourly multivariate
electricity-transformer time series with seven numerical variables.

The initial evaluation protocol will follow the papers:

- chronological train/validation/test partitions;
- scaling fitted on training data only;
- input window of 336 hourly observations;
- forecast horizons of 96, 192, 336, and 720 hours;
- MSE and MAE for comparison with published results;
- RMSE as an additional forecasting metric;
- last-value and seasonal-naive baselines;
- fixed random seeds and validation-based model selection.

## Reconstruction workflow

For each candidate model, the project will distinguish between:

1. the result reported in the paper;
2. a local run of the authors' official code;
3. a concise reconstruction maintained in this repository.

Results do not need to match the paper exactly. Differences will be measured and
explained using the dataset version, preprocessing, software environment,
hardware, random seed, and training configuration.

## Planned repository layout

```text
configs/       Experiment configurations
data/          Downloaded data (not committed)
docs/          Project notes and references
external/      Authors' repositories (not committed)
notebooks/     Dataset exploration and result analysis
results/       Reproducible tables and figures
scripts/       Simple download and experiment commands
src/           Data, model, training, and evaluation code
tests/         Automated correctness checks
```

Python modules and scripts will be the reproducible source of truth for model
training and evaluation. Notebooks will keep exploratory operations visible for
explanation and visualization without duplicating model-training logic.

## First dataset exploration

From the repository root, download and validate ETTh1:

```powershell
.\.venv\Scripts\python.exe scripts\download_etth1.py
```

Then start JupyterLab:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

Open `notebooks/01_etth1_eda.ipynb`, select the `.venv` Python kernel, and run
all cells. The notebook introduces the variables, data quality, benchmark
partitions, distributions, correlations, seasonality, and forecasting windows.

## Project instructions

The current assignment brief is available at
[`docs/project_instructions.pdf`](docs/project_instructions.pdf).
