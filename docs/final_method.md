# Final Method: Dynamic Per-Variable Multiscale DLinear

## Research question

Can DLinear forecast heterogeneous Weather variables more accurately when its
trend scale adapts to both the variable and the current input window?

## Reconstruction

Original DLinear applies a centered moving average with kernel 25:

\[
T=MA_{25}(X), \qquad R=X-T.
\]

Two shared linear maps project the 336 historical values of each variable to
the complete 96-step forecast:

\[
\hat{Y}=W_RR+W_TT.
\]

The implementation preserves the official repeated-endpoint padding,
`individual=False` shared temporal projections, MSE training objective, Adam
optimizer, chronological Weather split, and train-only scaling.

## V1 ablation: static per-variable scales

V1 computes three moving-average trends with kernels 25, 73, and 145. For
Weather's ten-minute observations these span approximately 4.2, 12.2, and 24.2
hours. Every variable learns one fixed convex scale mixture:

\[
\alpha_{c,:}=\operatorname{softmax}(a_c),
\qquad
T_{b,t,c}=\sum_m\alpha_{c,m}MA_{k_m}(X)_{b,t,c}.
\]

This isolates whether variables prefer different decomposition scales. It adds
only \(21\times3=63\) parameters.

## V2A: dynamic per-variable scales

The final model makes the scale weights depend on the current window. For every
window \(b\) and variable \(c\), it calculates four standardized summary
features:

\[
f_{b,c}=[\mu,\sigma,x_{last},x_{last}-x_{first}].
\]

A shared two-layer gate produces a dynamic correction, which is added to a
learned variable-specific prior:

\[
d_{b,c}=W_2\operatorname{GELU}(W_1f_{b,c}+b_1)+b_2,
\]

\[
\alpha_{b,c,:}=\operatorname{softmax}(a_c+d_{b,c}).
\]

The adaptive trend and remainder are then forecast through the original two
shared DLinear projections. The gate uses hidden dimension 8 and adds only 67
parameters beyond V1, for 64,834 total parameters.

## Experimental protocol

- Weather: 52,696 rows, 21 variables, ten-minute sampling.
- Input 336 observations; forecast 96 observations.
- Chronological 70%/10%/20% train/validation/test split.
- Validation and test receive historical context, but their targets remain
  inside their respective partitions.
- `StandardScaler` is fitted on training observations only.
- Adam, MSE loss, learning rate 0.0001, DLinear `type1` schedule.
- Maximum 10 epochs and patience 3.
- Seeds 2021, 2022, and 2023.
- Architecture selection uses validation MSE; MSE and MAE are reported.
- The daily seasonal-naive baseline repeats the last 144 observations.

The public Weather file contains one duplicate timestamp and one 100-minute
gap. They are retained for benchmark fidelity, matching the supplied row-based
benchmark representation.

## Results

| Model | Parameters | Test MSE (mean ± SD) | Test MAE (mean ± SD) |
|---|---:|---:|---:|
| DLinear | 64,704 | 0.17464 ± 0.00052 | 0.23486 ± 0.00174 |
| V1 static multiscale | 64,767 | 0.16739 ± 0.00044 | 0.22910 ± 0.00114 |
| **V2A dynamic multiscale** | **64,834** | **0.16540 ± 0.00045** | **0.22492 ± 0.00137** |

V2A beats paired DLinear on both metrics for all three seeds. Mean paired
relative improvements are 5.29% MSE and 4.23% MAE. For seed 2021, V2A obtains
MSE 0.16495 and MAE 0.22353; the paper reports DLinear MSE 0.176 and MAE 0.237.

Across horizons 96, 192, 336, and 720, V2A beats its paired DLinear run on both
metrics for every seed. Its mean paired MSE reductions are 5.29%, 3.22%, 2.16%,
and 1.42%; its corresponding MAE reductions are 4.23%, 2.69%, 1.83%, and 1.55%.
The improvement therefore remains consistent while becoming smaller at longer
forecast horizons.

## Interpretation and limitations

The progression from one global scale, to variable-specific scales, to
window-and-variable-specific scales is supported by the ablation results. The
current evidence uses one dataset and three seeds, so the project claims an
improvement for this forecasting task rather than a universal improvement on
every time-series dataset.

The final architecture was selected using the Weather validation set. V2A was
frozen using validation results before its checkpoints were evaluated on test;
this distinction is stated in the final report.
