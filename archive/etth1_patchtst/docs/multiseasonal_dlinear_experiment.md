# Period-Aware Multi-Seasonal DLinear on ETTh1

## Goal

This experiment tests whether explicit hourly daily and weekly structure improves
the reconstructed DLinear model. The original implementation in
`src/ts_project/models/dlinear.py` is unchanged; the candidate is implemented
separately in `src/ts_project/models/multiseasonal.py`.

## Architecture

The candidate first applies DLinear's centered 25-hour moving average to obtain
the trend. It then decomposes the detrended signal into three additive parts:

1. a 24-hour template, calculated by averaging the same hour across the 14 input
   days;
2. a 168-hour template, calculated from the daily-adjusted signal across the two
   input weeks; and
3. the irregular remainder left after removing both templates.

The components exactly reconstruct the input. Trend and irregular remainder use
the same shared 336-to-96 linear projections as ordinary DLinear. Daily and
weekly templates are phase-aligned and repeated into the forecast horizon, with
one learned strength per period and feature. This adds 14 parameters rather than
another forecasting backbone.

## Experimental Protocol

| Setting | Value |
|---|---:|
| Dataset | ETTh1 |
| Task | Multivariate-to-multivariate forecasting |
| Frequency | Hourly |
| Input length | 336 hours |
| Prediction length | 96 hours |
| Daily / weekly periods | 24 / 168 hours |
| Seed | 2021 |
| Batch size | 32 |
| Optimizer / objective | Adam / MSE |
| Initial learning rate | 0.005 |
| Schedule | DLinear `type1` halving schedule |
| Maximum epochs / patience | 10 / 3 |
| Device | CPU |

Both models used the same train-fitted scaler, chronological train/validation/test
partitions, independently re-seeded shuffled loaders, and validation-selected
checkpoint. The test set was evaluated only after each model's checkpoint had
been selected from validation MSE.

## Overall Results

| Model | Validation MSE | Validation MAE | Test MSE | Test MAE | Parameters |
|---|---:|---:|---:|---:|---:|
| Paper-reported DLinear | - | - | 0.375000 | 0.399000 | - |
| Paired DLinear | 0.648863 | 0.542037 | 0.371554 | **0.393450** | 64,704 |
| Period-aware DLinear | 0.651099 | 0.543226 | **0.371183** | 0.393940 | 64,718 |

The candidate lowers test MSE by 0.10%, but raises test MAE by 0.12% and has
0.34% worse validation MSE. The MSE change is too small and inconsistent with
validation and MAE to claim an improvement from one seed.

## Per-Feature Test Results

Positive change means the period-aware model has lower error than paired
DLinear.

| Feature | DLinear MSE | Period-aware MSE | MSE change | DLinear MAE | Period-aware MAE | MAE change |
|---|---:|---:|---:|---:|---:|---:|
| HUFL | 0.749956 | 0.749201 | +0.101% | 0.581284 | 0.581995 | -0.122% |
| HULL | 0.208278 | 0.207990 | +0.138% | 0.340493 | 0.339567 | +0.272% |
| MUFL | 0.774866 | 0.774727 | +0.018% | 0.577100 | 0.577941 | -0.146% |
| MULL | 0.169171 | 0.168731 | +0.260% | 0.296307 | 0.295487 | +0.277% |
| LUFL | 0.507130 | 0.506916 | +0.042% | 0.496620 | 0.500060 | -0.693% |
| LULL | 0.120639 | 0.119579 | +0.879% | 0.267577 | 0.267288 | +0.108% |
| OT | 0.070840 | 0.071137 | -0.419% | 0.194767 | 0.195240 | -0.243% |

Only `HULL`, `MULL`, and `LULL` improve on both metrics. The largest MSE gain is
for `LULL`, while `OT` deteriorates on both metrics. All changes remain below
1%, so feature-level conclusions also require replication across seeds.

## Learned Seasonal Strengths

| Feature | Daily | Weekly |
|---|---:|---:|
| HUFL | 0.957 | -0.035 |
| HULL | 0.901 | -0.035 |
| MUFL | 0.963 | -0.019 |
| MULL | 0.890 | -0.042 |
| LUFL | 0.835 | 0.051 |
| LULL | 0.777 | 0.066 |
| OT | 0.867 | 0.120 |

The daily strengths remain substantial. Weekly strengths converge close to zero,
suggesting that two observed weeks do not provide a stable extra seasonal signal
for this 96-hour task under the current decomposition.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\run_multiseasonal_dlinear.py
```

Exact metrics, per-feature values, learned strengths, configuration, runtimes,
and training histories are stored under
`results/dlinear/multiseasonal/etth1/horizon_096/seed_2021/`.

## Conclusion

This simple period-aware design is parameter-efficient and competitive with
DLinear, but it is not a validated improvement. A justified follow-up would
repeat several seeds and ablate daily-only versus daily-plus-weekly components.
That would test whether the near-zero weekly strengths are stable before adding
more complex seasonal machinery.
