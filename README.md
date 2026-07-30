# Long-Horizon Time-Series Forecasting: DLinear vs. PatchTST

This repository is a notebook-centered reconstruction and improvement study for
long-horizon multivariate time-series forecasting.

Two candidate papers are being evaluated on the same ETTh1 benchmark:

1. [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504)
   — DLinear ([official code](https://github.com/cure-lab/LTSF-Linear))
2. [A Time Series Is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730)
   — PatchTST ([official code](https://github.com/yuqinie98/PatchTST))

We will reconstruct both candidates before selecting one paper and one
meaningful improvement for the final submission.

## Project approach

The notebooks are the main learning and execution path. Each notebook is divided
into numbered sections that explain:

- what the step does;
- why the step is necessary;
- what the code calculates;
- how to interpret the output;
- how the result affects the next project decision.

Small Python modules contain reusable logic that should not be copied between
notebooks. For example, the notebooks will call the same tested data loader,
forecasting metrics, baselines, model definitions, and training loop.

## Repository structure

```text
configs/                  Small, readable experiment settings
data/                     Downloaded datasets (not committed)
docs/                     Assignment instructions and final documentation
notebooks/                EDA, reconstructions, improvement, and comparison
results/                  Compact generated metrics, tables, and figures
scripts/                  Dataset download and other essential commands
src/ts_project/
├── data/                 ETTh1 validation, scaling, and forecasting windows
├── models/               DLinear, PatchTST, and the selected improvement
├── baselines.py          Simple forecasting baselines
├── metrics.py            MSE, MAE, and RMSE
└── training.py           Shared neural-network training loop
tests/                    Focused correctness and leakage tests
```

Files listed above are added only when their corresponding project step is
implemented. This keeps the repository compact and prevents placeholder code
from becoming confusing.

## Notebook plan

| Notebook | Purpose |
|---|---|
| `01_etth1_eda.ipynb` | Understand and validate the dataset |
| `02_dlinear_reconstruction.ipynb` | Explain, train, and evaluate DLinear |
| `03_patchtst_reconstruction.ipynb` | Explain, train, and evaluate PatchTST |
| `04_improved_method.ipynb` | Implement and test the selected improvement |
| `05_final_comparison.ipynb` | Produce final tables and report figures |

The final submission may retain all notebooks for transparency while identifying
the selected reconstruction and improvement clearly in the README and report.

## Shared benchmark

Both models use the public
[ETTh1 dataset](https://github.com/zhouhaoyi/ETDataset), an hourly,
seven-variable electricity-transformer time series.

The planned evaluation protocol is:

- chronological training, validation, and test partitions;
- scaling fitted on training data only;
- input window of 336 hourly observations;
- forecast horizons of 96, 192, 336, and 720 hours;
- MSE and MAE for comparison with the papers;
- RMSE as an additional forecasting metric;
- last-value and daily seasonal-naive baselines;
- validation-based model selection;
- fixed seeds, followed by repeated-seed evaluation for the final comparison.

## Setup

Create and activate a virtual environment, then install the project:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements install CUDA-enabled PyTorch and the `ipykernel` support needed
to run notebooks directly in VS Code. JupyterLab is not required.

Download and validate ETTh1:

```powershell
.\.venv\Scripts\python.exe scripts\download_etth1.py
```

Then open `notebooks/01_etth1_eda.ipynb` in VS Code, select the project `.venv`
kernel, and use **Run All** or run one section at a time.

## Current status

- ETTh1 download and validation: complete
- leakage-safe chronological partitions and scaling: complete
- forecasting-window construction: complete
- explanatory ETTh1 EDA: complete
- naive baselines and shared metrics: complete
- DLinear reconstruction: next
- PatchTST reconstruction: planned
- selected improvement: planned

## Assignment instructions

The assignment brief is available at
[`docs/project_instructions.pdf`](docs/project_instructions.pdf).
