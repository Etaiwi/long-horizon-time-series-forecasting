# Validation-Selected Feature-Adaptive DLinear

## Motivation

Earlier ETTh1 experiments showed a consistent feature split. RevIN and daily
slice correction substantially helped the target temperature feature (`OT`),
but applying the same normalization to all seven variables worsened aggregate
validation error. The daily/weekly period-aware model also learned weekly
strengths close to zero.

This experiment tests whether DLinear should use different preprocessing for
different features. It also evaluates two small architectural alternatives
before adopting the feature selector:

1. a cross-period daily residual branch;
2. rank-1 channel-specific temporal adapters; and
3. validation-selected raw versus RevIN-DLinear forecasts.

The method is implemented without changing the original DLinear source.

## Candidate Architectures

### Cross-period daily residual

The 336-hour input contains fourteen complete days. For each feature, values are
grouped by hour of day into 24 sequences of length 14. One shared linear layer
maps those fourteen observed days to the required number of future days:

\[
R_{h,c,1:D_{out}}
= W_{daily}X_{h,c,1:14}.
\]

The daily forecast is interleaved back into hourly order and added to DLinear.
Its weights start at zero, so the initial model is exactly the paired DLinear
forecast rather than an uncontrolled perturbation.

### Rank-1 channel adapters

Ordinary shared DLinear uses one temporal projection for every feature. The
adapter candidate retains that shared matrix and adds a small feature-specific
correction:

\[
W_c = W_{shared} + A_cB_c,
\qquad \operatorname{rank}(A_cB_c)=1.
\]

The correction is added independently to the trend and seasonal projections.
Its output factor starts at zero, again making the initial forecast identical to
shared DLinear.

### Consensus raw/RevIN selector

Raw DLinear and RevIN-DLinear are trained independently. For every feature, the
method compares their mean validation MSE across seeds 2021, 2022, and 2023:

\[
m_c =
\mathbb{1}\left[
\overline{MSE}^{val}_{RevIN,c}
<
\overline{MSE}^{val}_{raw,c}
\right].
\]

One mask is then frozen before aggregate test results are inspected:

\[
\hat{Y}_c =
m_c\hat{Y}_{RevIN,c}
+(1-m_c)\hat{Y}_{raw,c}.
\]

The development validation results selected RevIN only for `OT`. The same
`OT`-only rule is used for all seeds and forecast horizons. Test data never
selects a feature, model, or hyperparameter.

## Protocol

| Setting | Value |
|---|---:|
| Dataset | ETTh1 |
| Task | Multivariate-to-multivariate forecasting |
| Frequency | Hourly |
| Input length | 336 |
| Development horizon | 96 |
| Confirmation horizons | 192, 336, 720 |
| Seeds | 2021, 2022, 2023 |
| Moving-average kernel | 25 |
| Daily period | 24 |
| Adapter rank | 1 |
| Batch size | 32 |
| Optimizer / loss | Adam / MSE |
| Initial learning rate | 0.005 |
| Maximum epochs / patience | 10 / 3 |

Every model uses the same chronological partitions and train-fitted scaler.
Each run receives a newly seeded shuffled training loader. The best checkpoint
is selected using validation MSE only.

## Development Decision at Horizon 96

| Model | Mean validation MSE | Mean test MSE | Mean test MAE | Parameters |
|---|---:|---:|---:|---:|
| Paired DLinear | 0.641923 | 0.381133 | 0.405212 | 64,704 |
| Cross-period daily | 0.646853 | 0.373296 | 0.395716 | 64,760 |
| Rank-1 DLinear | 0.667478 | 0.374527 | 0.397293 | 70,752 |
| RevIN-DLinear | 0.673488 | 0.371109 | 0.392090 | 64,718 |
| **Consensus raw/RevIN** | **0.641418** | **0.373201** | **0.394751** | 129,422* |

\*The selector retains two trained forecasts. Its effective output still uses
one forecast per feature, but storing both branches approximately doubles the
parameter count.

Neither daily nor low-rank adaptation beat DLinear on mean validation MSE, so
the predefined rule prevented training their combined model. RevIN-DLinear had
the lowest test error but much worse validation MSE; choosing it from test would
therefore be leakage. The consensus selector was the only candidate that beat
DLinear on the development validation criterion.

## Repeated-Seed Results Across Horizons

| Horizon | DLinear MSE | Selected MSE | MSE improvement | DLinear MAE | Selected MAE | MAE improvement |
|---:|---:|---:|---:|---:|---:|---:|
| 96 | 0.381133 | **0.373201** | **2.08%** | 0.405212 | **0.394751** | **2.58%** |
| 192 | 0.414595 | **0.406044** | **2.06%** | 0.424826 | **0.414709** | **2.38%** |
| 336 | 0.489374 | **0.451056** | **7.83%** | 0.480688 | **0.446559** | **7.10%** |
| 720 | 0.490353 | **0.448748** | **8.48%** | 0.502329 | **0.459416** | **8.54%** |

The selected method improves both paper metrics at every horizon. At 96 hours,
test-MSE standard deviation falls from 0.012862 to 0.005497, indicating that the
`OT` normalization also reduces sensitivity to the training seed.

## Per-Feature Result at Horizon 96

The frozen mask leaves the six load features exactly equal to raw DLinear. Only
`OT` changes:

| Feature | DLinear MSE | Selected MSE | DLinear MAE | Selected MAE |
|---|---:|---:|---:|---:|
| HUFL | 0.754462 | 0.754462 | 0.588070 | 0.588070 |
| HULL | 0.213210 | 0.213210 | 0.344447 | 0.344447 |
| MUFL | 0.779031 | 0.779031 | 0.584323 | 0.584323 |
| MULL | 0.172671 | 0.172671 | 0.299405 | 0.299405 |
| LUFL | 0.516136 | 0.516136 | 0.500107 | 0.500107 |
| LULL | 0.123999 | 0.123999 | 0.272232 | 0.272232 |
| **OT** | 0.108419 | **0.052898** | 0.247898 | **0.174670** |

Across three seeds, selecting RevIN for `OT` reduces its mean test MSE by 51.21%
and MAE by 29.54%. This one stable feature-level effect produces the aggregate
improvement without exposing the other variables to harmful normalization.

## Comparison With Original Metrics

| Result | Seed(s) | MSE | MAE |
|---|---|---:|---:|
| Paper-reported DLinear | - | 0.375000 | 0.399000 |
| Original repository reconstruction | 2021 | 0.370509 | 0.392145 |
| Selected method | 2021 | **0.368971** | **0.390531** |
| Paired DLinear mean | 2021-2023 | 0.381133 | 0.405212 |
| Selected-method mean | 2021-2023 | **0.373201** | **0.394751** |

Under the same seed as the earlier repository result, the selected method
improves MSE by 0.42% and MAE by 0.41%. The repeated-seed mean also beats the
paper-reported values, although the paired repeated-seed comparison is the more
reliable evidence.

## What We Learned

- Explicit daily modeling can improve test accuracy but did not generalize on
  the validation criterion, so it should not be claimed as the improvement.
- Rank-1 adapters also failed validation and were relatively expensive on CPU.
- Uniform RevIN is inappropriate for ETTh1 because its value is concentrated in
  `OT`.
- A validation-frozen, feature-selective transformation is a small but robust
  improvement and becomes more valuable at longer horizons.
- Negative ablations are useful: they prevent test-driven architecture choices
  and narrow the final explanation to the mechanism supported by evidence.

## Reproduction

Run and inspect:

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook notebooks\05_daily_adaptive_dlinear.ipynb
```

The notebook currently reuses the completed run file so tables can be regenerated
quickly. Set `reuse_completed_runs=False` in its experiment cell to retrain all
models from scratch.

Machine-readable outputs are stored in
`results/dlinear/daily_adaptation_sequence/`:

- `sequence_metrics.json`: every seed, horizon, feature, and training history;
- `summary.csv`: aggregate mean and standard deviation;
- `per_feature_h096.csv`: repeated-seed feature results; and
- `config.yaml`: the exact experimental protocol.

## Research Basis

- [SparseTSF](https://arxiv.org/abs/2405.00946) motivates cross-period sparse
  forecasting.
- [CycleNet](https://proceedings.neurips.cc/paper_files/paper/2024/file/bfe7998398779dde03cad7a73b1f81b6-Paper-Conference.pdf)
  supports explicit residual cycle modeling and identifies ETTh1's main cycle as
  24 hours.
- [C-LoRA](https://arxiv.org/abs/2407.17246) motivates efficient channel-specific
  low-rank adaptation.
- [Revisiting Linear Mapping](https://arxiv.org/abs/2305.10721) motivates
  reversible normalization and channel-aware treatment in linear forecasting.
