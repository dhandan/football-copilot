# H10 two-stage match-statistic screen

Status: NO-GO: +2pp development accuracy hurdle not met

Inspired by Wheatcroft (2021), but not an exact implementation of GAP ratings.
The test builds strictly lagged shots/corners features and trains six stage-1 forecasts out of season. Stage 2 is trained only on out-of-season stage-1 predictions, never actual match statistics for those fixtures.
Baseline fixed 40% Model 2 / 60% market: 594/1100 (54.00%). Target >=616/1100 with no Log Loss or Brier deterioration.
Historical 1100-fixture cohort previously used in many tests, so this is a development screen, not pristine out-of-sample validation.

| Method | Standalone | Fixed 40/60 blend | Accuracy uplift | 95% paired bootstrap CI | Log Loss | Brier |
|---|---:|---:|---:|---|---:|---:|
| ordinal | 51.273% | 54.364% (598/1100) | +0.364pp | [-0.545, 1.364]pp | 0.97647 | 0.5811 |
| multinomial | 51.091% | 54.273% (597/1100) | +0.273pp | [-0.727, 1.273]pp | 0.97747 | 0.58166 |
| direct | 49.818% | 53.455% (588/1100) | -0.545pp | [-1.727, 0.636]pp | 0.98102 | 0.58465 |

## Caveats
- Off-target proxy is total shot attempts minus on-target attempts; blocked attempts are included.
- Stage 1 uses original 25 pre-match football features plus 12 prior-five-match shot/corner features.
- Predictions within a test season may use team match statistics from earlier fixtures in that season, not future matches.
- No hyperparameter search on test outcomes. Paired bootstrap intervals do not correct for prior historical research or season clustering.
- Source data and reference predictions unchanged. No 2026/27 results were used.
- The referenced study primarily demonstrates useful probabilistic and betting signals, not +2pp 1X2 accuracy.
- Full technical reproduction of GAP ratings is outside this bounded screen; the core two-stage causal question is tested.

## Files
- experiments/h10_two_stage/bounded_screen.py
- reports/h10_two_stage/results.json
- reports/h10_two_stage/predictions.csv
- https://journals.sagepub.com/doi/full/10.3233/JSA-200462
