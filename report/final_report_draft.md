# Dynamic Multiscale Decomposition for DLinear Forecasting

**Course:** Time Series Analysis<br>
**Students:** [add names and student numbers]<br>
**Selected paper:** *Are Transformers Effective for Time Series Forecasting?* (Zeng et al., 2023)

This document is the working source for the final PDF.

## 1. Original architecture

### 1.1 Forecasting task

We reconstruct DLinear on the Weather dataset. This is a **supervised, multivariate-to-multivariate forecasting** task: each input contains 336 observations of all 21 variables, and the model predicts the next 96 observations of all 21 variables. The data are sampled every ten minutes, so the input covers 56 hours and the main forecast horizon covers 16 hours.

DLinear predicts the complete output window in one forward pass. It does not predict one future observation and then feed that prediction back into the model.

### 1.2 Decomposition

For an input window \(X\), the original model estimates a trend with a centered moving average of kernel size 25:

\[
T=MA_{25}(X), \qquad R=X-T.
\]

The official implementation pads both ends by repeating the endpoint values before applying the moving average. This keeps the trend sequence the same length as the input and avoids introducing zeros at the boundaries.

### 1.3 Forecasting layers

Two linear layers separately map the 336-point trend and remainder sequences to the complete 96-point forecast:

\[
\hat{Y}=W_TT+W_RR.
\]

We use the paper's shared-channel configuration (`individual=False`). The same temporal projection weights are applied to every variable, while each variable is still forecast from its own trend and remainder sequences. The model is trained by minimizing mean squared error (MSE).

### 1.4 Main hyperparameters

| Setting | Value |
|---|---:|
| Sampling frequency | 10 minutes |
| Number of variables | 21 |
| Input length | 336 observations (56 hours) |
| Main output length | 96 observations (16 hours) |
| Moving-average kernel | 25 |
| Batch size | 16 |
| Optimizer | Adam |
| Initial learning rate | 0.0001 |
| Learning-rate schedule | DLinear `type1` |
| Training objective | MSE |
| Maximum epochs | 10 |
| Early-stopping patience | 3 |
| Reconstruction seed | 2021 |
| Robustness seeds | 2021, 2022, 2023 |

## 2. Paper results

For the Weather dataset with input length 336 and forecast horizon 96, the DLinear paper reports:

| Model | MSE | MAE |
|---|---:|---:|
| DLinear (paper) | 0.176 | 0.237 |

MSE squares the errors and therefore gives more weight to large forecast errors. MAE is the average absolute error and is easier to interpret as the typical error magnitude. Both metrics are calculated in standardized space, and lower values are better.

## 3. Reconstruction results

### 3.1 Reproducible data pipeline

The Weather file contains 52,696 rows and 21 numeric variables. We parse and validate the datetime column, retain the supplied benchmark rows, and use the paper's chronological 70%/10%/20% train/validation/test split. The scaler is fitted only on the training partition and then applied to validation and test data. Input windows may use earlier observations as context, but every target remains completely inside its own partition.

The public file includes one duplicate timestamp and one 100-minute gap. We retain them to reproduce the row-based benchmark used by the original implementation. We do not interpolate or resample the values.

### 3.2 Simple baseline

Our seasonal-naive baseline repeats the latest 144 observations because 144 ten-minute samples represent one day.

| Model | MSE | MAE | RMSE |
|---|---:|---:|---:|
| Seasonal naive | 0.316707 | 0.287956 | 0.562767 |

### 3.3 Direct comparison with the paper

The paper's public experiment script uses one run (`itr=1`), and the official runner sets seed 2021. We therefore use our seed-2021 run for the direct reconstruction comparison.

| Metric | Paper | Our reconstruction | Relative difference |
|---|---:|---:|---:|
| MSE | 0.176000 | 0.174205 | -1.02% |
| MAE | 0.237000 | 0.233328 | -1.55% |

The reconstruction is very close to the reported values. Its slightly lower errors can result from software versions, random initialization, or low-level GPU behavior. This result is close enough to use as the experimental baseline for the improvement.

We also repeat the reconstruction with three explicitly stated seeds to measure variability:

| Model | Test MSE (mean ± SD) | Test MAE (mean ± SD) |
|---|---:|---:|
| DLinear | 0.17464 ± 0.00052 | 0.23486 ± 0.00174 |

The three-seed average is a robustness analysis for our project; it is not presented as the value reported by the paper.

## 4. Improved architecture

### 4.1 Motivation

Original DLinear uses the same kernel size for every variable and every input window. Weather variables can change at different speeds, and the most useful trend scale may also change over time. We therefore test whether the model can improve by combining several trend scales while preserving DLinear's simple forecasting layers.

### 4.2 V1: static per-variable multiscale decomposition

V1 computes trends with kernels 25, 73, and 145. At ten-minute sampling, these cover approximately 4.2, 12.2, and 24.2 hours. Each variable learns one fixed mixture of these trends:

\[
\alpha_{c,:}=\operatorname{softmax}(a_c),
\qquad
T_{b,t,c}=\sum_{m=1}^{3}\alpha_{c,m}MA_{k_m}(X)_{b,t,c}.
\]

The softmax makes the three weights non-negative and forces them to sum to one. V1 is an ablation that tests whether variables benefit from different fixed decomposition scales. It adds only \(21\times3=63\) learned parameters.

### 4.3 V2A: dynamic per-window and per-variable decomposition

Our final improvement also adapts the mixture to the current input window. For each window \(b\) and variable \(c\), it calculates four features using only the observed input:

\[
f_{b,c}=[\operatorname{mean}(X),\operatorname{std}(X),x_{last},x_{last}-x_{first}].
\]

A small gate shared by all variables converts these features into three dynamic corrections:

\[
d_{b,c}=W_2\operatorname{GELU}(W_1f_{b,c}+b_1)+b_2.
\]

These corrections are added to a learned variable-specific prior \(a_c\), followed by softmax:

\[
\alpha_{b,c,:}=\operatorname{softmax}(a_c+d_{b,c}).
\]

The resulting weights combine the three moving-average trends. The remainder is still \(R=X-T\), and the original DLinear forecasting layers predict the trend and remainder separately. The gate has hidden size 8. It adds 67 parameters beyond V1 and does not use future target values.

| Model | Parameters | Increase from DLinear |
|---|---:|---:|
| DLinear | 64,704 | — |
| V1 | 64,767 | +63 (+0.097%) |
| V2A | 64,834 | +130 (+0.201%) |

## 5. Improved results

All models use the same data split, scaling, input length, forecast horizon, optimizer, training budget, early stopping rule, and three seeds. Model selection is based on validation MSE before evaluating the final checkpoints on the test set.

### 5.1 Main 96-step experiment

| Model | Test MSE (mean ± SD) | Test MAE (mean ± SD) | MSE wins vs DLinear | MAE wins vs DLinear |
|---|---:|---:|---:|---:|
| DLinear | 0.17464 ± 0.00052 | 0.23486 ± 0.00174 | — | — |
| V1 static multiscale | 0.16739 ± 0.00044 | 0.22910 ± 0.00114 | 3/3 | 3/3 |
| **V2A dynamic multiscale** | **0.16540 ± 0.00045** | **0.22492 ± 0.00137** | **3/3** | **3/3** |

Relative to paired DLinear runs, V1 reduces mean MSE by approximately 4.15% and MAE by 2.45%. V2A reduces mean MSE by approximately 5.29% and MAE by 4.23%. V2A also improves over V1 by approximately 1.19% MSE and 1.82% MAE. The improvement therefore comes with only a 0.201% increase in parameter count.

### 5.2 Multi-horizon evaluation

To test whether the improvement is limited to the main 96-step experiment, we repeat the paired three-seed comparison at the four forecast horizons used in the paper. The input length remains 336 observations. At ten-minute sampling, horizons 96, 192, 336, and 720 correspond to 16, 32, 56, and 120 hours.

| Horizon | DLinear MSE | V2A MSE | MSE reduction | DLinear MAE | V2A MAE | MAE reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 96 | 0.17464 ± 0.00052 | **0.16540 ± 0.00045** | **5.29%** | 0.23486 ± 0.00174 | **0.22492 ± 0.00137** | **4.23%** |
| 192 | 0.21570 ± 0.00029 | **0.20875 ± 0.00072** | **3.22%** | 0.27270 ± 0.00082 | **0.26537 ± 0.00175** | **2.69%** |
| 336 | 0.26297 ± 0.00142 | **0.25730 ± 0.00214** | **2.16%** | 0.31445 ± 0.00313 | **0.30870 ± 0.00422** | **1.83%** |
| 720 | 0.32444 ± 0.00069 | **0.31983 ± 0.00109** | **1.42%** | 0.36299 ± 0.00125 | **0.35737 ± 0.00151** | **1.55%** |

V2A obtains lower MSE and MAE than its paired DLinear run for all three seeds at all four horizons. Its largest relative gain occurs at horizon 96. The gain becomes smaller as the forecast horizon increases but remains consistent at horizon 720. This suggests that the adaptive decomposition is most useful for the shorter forecast while still generalizing across all tested horizons.

## 6. Discussion

The ablation gives a clear progression. V1 shows that one decomposition scale is not equally suitable for all Weather variables. V2A improves further, suggesting that the preferred scale also changes with the current window. Because V2A retains DLinear's forecasting path and changes only the decomposition, the source of the improvement is easy to isolate.

The result is parameter-efficient. V2A adds 130 parameters to a 64,704-parameter baseline, while producing lower MSE and MAE for all three paired seeds at horizon 96. However, computing three moving averages increases training time, so the small parameter overhead should not be confused with zero computational cost. Total runtime also depends on the epoch selected by early stopping and is therefore not a controlled speed benchmark.

The multi-horizon experiment strengthens this result: V2A wins on both metrics in all 12 paired comparisons. However, the decreasing relative gain suggests that decomposition-scale adaptation has less influence as forecast uncertainty grows over longer horizons.

The main limitations are that the current evidence uses one dataset and three seeds, and the final architecture was chosen using the validation set. The multi-horizon evaluation provides an additional test of robustness, but it does not replace evaluation on an independent dataset. We therefore claim an improvement for this forecasting task rather than a universal replacement for DLinear.

## 7. Conclusion

We reconstructed DLinear closely on the Weather benchmark and introduced a dynamic multiscale decomposition. The proposed V2A method lets each variable and input window choose a convex mixture of short-, medium-, and daily-scale trends. At the 96-step horizon, it improves both MSE and MAE for all three paired seeds with only 0.201% more parameters. It also outperforms DLinear for every seed at horizons 192, 336, and 720, although the relative improvement decreases for longer forecasts.

## 8. References

1. A. Zeng, M. Chen, L. Zhang, and Q. Xu, “Are Transformers Effective for Time Series Forecasting?” *Proceedings of the AAAI Conference on Artificial Intelligence*, 2023.
2. Official DLinear implementation: <https://github.com/cure-lab/LTSF-Linear>
3. Public time-series datasets used by the paper: <https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy>
