# Adaptive Multiscale DLinear for Weather Forecasting

This repository reconstructs **DLinear** from *Are Transformers Effective for
Time Series Forecasting?* and evaluates one lightweight improvement on the
public Weather benchmark.

The selected method is **Dynamic Per-Variable Multiscale DLinear**. Original
DLinear uses one fixed moving-average kernel for every variable and forecasting
window. The improved model combines three trend scales and lets their weights
depend on both the variable and the current input window.

## Final forecasting task

- dataset: Weather, 52,696 ten-minute observations and 21 variables;
- task: multivariate-to-multivariate forecasting;
- input length: 336 observations (56 hours);
- prediction length: 96 observations (16 hours);
- split: chronological 70% train, 10% validation, 20% test;
- preprocessing: `StandardScaler` fitted only on training observations;
- metrics: standardized-space MSE and MAE;
- baseline: daily seasonal naive with period 144;
- seeds: 2021, 2022, and 2023.

## Main result

The architecture was selected using validation data. After it was frozen, the
saved checkpoints were evaluated on the test partition.

| Model | Parameters | Test MSE (mean ± SD) | Test MAE (mean ± SD) |
|---|---:|---:|---:|
| DLinear reconstruction | 64,704 | 0.17464 ± 0.00052 | 0.23486 ± 0.00174 |
| Static per-variable multiscale (V1) | 64,767 | 0.16739 ± 0.00044 | 0.22910 ± 0.00114 |
| **Dynamic per-variable multiscale (V2A)** | **64,834** | **0.16540 ± 0.00045** | **0.22492 ± 0.00137** |

V2A improves over the paired DLinear reconstruction for all three seeds. Its
mean relative reduction is approximately **5.29% MSE** and **4.23% MAE**, while
adding only 130 parameters (about 0.20%). The paper reports DLinear test results
of MSE 0.176 and MAE 0.237 for the same Weather 336-to-96 task.

## Final project path

The concise final workflow is:

```text
configs/final_weather_dlinear.yaml
docs/final_method.md
notebooks/final/01_dlinear_reconstruction.ipynb
notebooks/final/02_dynamic_multiscale_improvement.ipynb
results/dlinear/weather/horizon_096/
scripts/run_weather_dlinear.py
src/ts_project/data/weather.py
src/ts_project/models/adaptive_dlinear.py
```

The original ETTh1 DLinear/PatchTST reconstructions and exploratory improvement
notebooks remain in the repository as development evidence. They are no longer
the final-project path.

## Environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements use CUDA-enabled PyTorch and include `ipykernel`, so the
notebooks can be opened directly in VS Code without installing JupyterLab.

Place the Weather file at:

```text
data/raw/weather.csv
```

Run the focused tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_weather_data.py tests/test_adaptive_dlinear.py
```

Run a reconstruction or improvement experiment:

```powershell
.\.venv\Scripts\python.exe scripts/run_weather_dlinear.py --model dlinear --seed 2021 --evaluate-test
.\.venv\Scripts\python.exe scripts/run_weather_dlinear.py --model v2a --seed 2021 --evaluate-test
```

## Assignment material

The assignment instructions remain available at
[`docs/project_instructions.pdf`](docs/project_instructions.pdf).
