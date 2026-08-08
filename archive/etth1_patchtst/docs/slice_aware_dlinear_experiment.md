# Slice-Aware DLinear Experiment

## Summary

This experiment tested whether DLinear could handle input-to-output distribution shift more effectively by predicting the mean of each future daily slice instead of reconstructing the entire forecast with a single input-window mean. It also tested whether a learned per-channel gate could choose between ordinary DLinear and the slice-aware forecast.

The experiment produced an informative negative overall result. Neither slice-aware variant beat the paired DLinear baseline on aggregate validation or test MSE. However, the slice-aware mean model reduced the `OT` (oil temperature) test MSE by **21.71%**, from `0.070816` to `0.055442`. The improvement was outweighed by errors on several electrical-load channels.

The implementation and complete executed analysis are in [`notebooks/04_slice_aware_dlinear.ipynb`](../notebooks/04_slice_aware_dlinear.ipynb).

## Motivation

Ordinary RevIN normalizes each input channel using statistics calculated over the lookback window and restores the forecast using the same statistics. This assumes that the future window has approximately the same mean and scale as the input window. For long-horizon forecasting, that assumption can fail when a channel's baseline continues to rise or fall.

An earlier paired experiment showed that RevIN-DLinear improved `OT` substantially but made all six load-channel forecasts worse. This motivated two questions:

1. Can a small statistics head predict the evolving future baseline instead of copying the input mean?
2. Can a learned channel gate preserve the benefit on `OT` while retaining raw DLinear on channels harmed by normalization?

The objective was to answer these questions without replacing DLinear. Its moving-average decomposition and temporal linear projection layers remain the forecasting backbone.

## Experimental Setup

The experiment used multivariate ETTh1 forecasting with the following configuration:

| Setting | Value |
|---|---:|
| Input length | 336 hours |
| Forecast horizon | 96 hours |
| Input daily slices | 14 × 24 hours |
| Future daily slices | 4 × 24 hours |
| Number of channels | 7 |
| Random seed | 2021 |
| Auxiliary statistics-loss weight | 0.1 |

All variants used the same train, validation, and test partitions, optimizer settings, and seeded data-loading procedure. Each model was initialized and trained independently from the same seed.

## Models Evaluated

### 1. Paired DLinear baseline

The baseline applies the standard DLinear decomposition and independently projects the seasonal and trend components into the forecast horizon.

### 2. RevIN-DLinear

For each sample and channel, RevIN calculates the input mean \(\mu_X\) and standard deviation \(\sigma_X\), normalizes the input, passes it through DLinear, and restores the original scale:

\[
\hat{Y}_{\text{RevIN}}
=
\sigma_X DLinear\!\left(\frac{X-\mu_X}{\sigma_X}\right)+\mu_X.
\]

The notebook uses learnable channel-wise affine parameters in the RevIN layer.

### 3. Slice-aware future-mean DLinear

The 336-hour input is divided into fourteen 24-hour slices. For each channel \(c\), the model calculates a sequence of daily means:

\[
m_{k,c}=\frac{1}{24}\sum_{t=1}^{24}X_{k,t,c},
\qquad k=1,\ldots,14.
\]

A small channel-wise linear head maps the 14 observed means to four predicted future daily means. These are expanded across the corresponding four 24-hour forecast blocks. The normalized DLinear output models temporal fluctuations, while the statistics head supplies the evolving baseline:

\[
\hat{Y}_{\text{slice}}
=
\sigma_X DLinear\!\left(\frac{X-\mu_X}{\sigma_X}\right)
+\hat{\mu}_{\text{future}}.
\]

During training, the true future-slice means supervise the statistics head. They are never used as inputs and are unavailable during validation and test inference, so the method does not leak future data.

### 4. Gated slice-aware DLinear

The gated model computes both raw and slice-aware forecasts and blends them independently for each channel:

\[
\hat{Y}_c
=
(1-g_c)\hat{Y}_{\text{raw},c}
+g_c\hat{Y}_{\text{slice},c},
\qquad
g_c=\operatorname{sigmoid}(a_c).
\]

A gate near zero favors ordinary DLinear; a gate near one favors the slice-aware forecast.

## Training Objective

The slice-aware variants use the forecast MSE plus an auxiliary future-statistics loss:

\[
\mathcal{L}
=
\operatorname{MSE}(\hat{Y},Y)
+0.1\operatorname{MSE}
\left(\hat{\mu}_{\text{future slices}},
\mu_{\text{true future slices}}\right).
\]

Future-scale prediction was implemented as an optional extension, with positive scale enforced through an exponential transform. It was not trained in this experiment because the predefined decision rule required a mean-aware model to improve validation MSE first.

## Aggregate Results

| Model | Validation MSE | Test MSE | Test MAE | Parameters | Test MSE vs DLinear |
|---|---:|---:|---:|---:|---:|
| DLinear | 0.648873 | 0.371544 | 0.393440 | 64,704 | — |
| RevIN-DLinear | 0.678503 | 0.384905 | 0.403304 | 64,718 | 3.60% worse |
| Slice-aware mean | 0.700887 | 0.372975 | 0.394345 | 65,124 | 0.39% worse |
| Gated slice-aware mean | 0.678392 | 0.375075 | 0.396327 | 129,835 | 0.95% worse |

The slice-aware mean model recovered most of the aggregate loss caused by ordinary RevIN, but it still did not outperform raw DLinear. The learned gate also failed to improve the aggregate result and approximately doubled the number of backbone parameters.

## Per-Feature Results

Positive improvement values mean lower MSE than DLinear.

| Feature | DLinear MSE | RevIN improvement | Slice-aware improvement | Gated improvement |
|---|---:|---:|---:|---:|
| HUFL | 0.749956 | -5.51% | -0.70% | -0.25% |
| HULL | 0.208271 | -3.14% | +0.39% | +0.37% |
| MUFL | 0.774865 | -5.84% | -0.95% | -0.55% |
| MULL | 0.169163 | -3.26% | +0.19% | -0.23% |
| LUFL | 0.507095 | -2.31% | -1.57% | -3.91% |
| LULL | 0.120640 | -0.48% | -4.95% | -3.65% |
| **OT** | **0.070816** | **+24.48%** | **+21.71%** | **+7.34%** |

The main finding is strongly channel-dependent:

- `OT` improved under all three normalization-based variants.
- Slice-aware prediction also produced very small improvements on `HULL` and `MULL`, but a single seed is insufficient to treat gains below 0.5% as established.
- Deterioration on `HUFL`, `MUFL`, `LUFL`, and `LULL` outweighed the substantial `OT` improvement in the overall average.

## Interpretation

The results support the hypothesis that future-mean correction is useful for `OT`, whose baseline appears to shift in a way that the daily-slice statistics head can exploit. They do not support applying the same normalization mechanism uniformly to all ETTh1 channels.

The learned sigmoid gate did not discover a reliable channel selection. Its final values remained above 0.5 for most load variables even though raw DLinear was preferable overall. Because both branches and the gates were optimized jointly under a single aggregate loss, the gate values are not guaranteed to represent an independently validated model choice. The additional raw-DLinear branch also caused the parameter count to increase from 64,704 to 129,835.

## Decision on Future-Scale Prediction

Future-scale prediction was intentionally skipped. Before examining test performance, the experiment specified that this step would run only if either mean-aware model beat DLinear on validation MSE. The best mean-aware validation MSE was `0.678392`, compared with `0.648873` for DLinear, so the condition was not met.

This decision prevents the test set from being used to select the next model variant and avoids adding a noisier scale-prediction mechanism when mean prediction had not yet generalized overall.

## Limitations

- Results are from one seed and one forecast configuration, so the `OT` improvement must be replicated.
- Aggregate MSE weights the seven channels equally but can obscure meaningful channel-specific behavior.
- The learned gate was trained jointly with both forecasting branches and had no sparsity or selection constraint.
- The comparison does not yet include multiple horizons or formal uncertainty estimates.
- The optional future-scale path was tested structurally but not trained and evaluated.

## Recommended Next Experiment

The next justified experiment is validation-selected channel masking:

1. Train raw DLinear and the slice-aware candidate without selecting channels from test results.
2. Compare their MSE separately for every channel on the validation set.
3. Select one method per channel using validation performance only.
4. Freeze the resulting binary mask.
5. Evaluate the combined forecast once on the test set.
6. Repeat across several seeds and forecast horizons.

This experiment directly tests whether the `OT`/load-channel split generalizes, while avoiding the weakly identified end-to-end sigmoid gate. If the mask is stable across seeds and horizons, it can motivate a constrained learned selector in a later model.

## Conclusion

Slice-aware future-mean prediction did not improve DLinear's overall ETTh1 score for the 336-to-96 configuration. It did, however, reveal a clear and repeatable-looking feature-level signal: `OT` MSE improved by 21.71%, while several load channels deteriorated. The attempt therefore narrows the research direction from universal normalization to validation-driven, channel-selective normalization while keeping DLinear as the core forecasting model.
