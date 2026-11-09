# H11: High-confidence prediction policy – verified retrospective coverage/accuracy trade-off

**Date:** 9 November 2026.  **Status:** demonstrated historical selective-pick accuracy, **NOT** an all-fixture model accuracy uplift, **NOT** an untested guarantee of future results. No production forecasts modified.

## Objective and important distinction

Original target: improve fixed 40% Model 2 / 60% opening-market **1X2 accuracy across every one of 1,100 historical fixtures** from 54.00% (594/1,100) to at least 56.00% (616/1,100). This remains **unmet**.

One pragmatic alternative is a **selective prediction policy**: only label a pick as *high confidence* when `max(P_home, P_draw, P_away) >= 0.50`. Continue publishing the unchanged H/D/A probability vector for all 10 fixtures, but make *high-conviction pick accuracy* a distinct, clearly labelled outcome with reduced coverage. Pre-match probability is the sole input to the rule. Never re-label this as a +2pp improvement across all matches.

## Historical 1,100 benchmark

Source: `reports/model6/model6_walk_forward_predictions.csv`. Reconstructed probabilities exactly as `0.4*Model2 + 0.6*MarketOpen`. Recomputed baseline 594/1,100 = **54.00%**.

For threshold at least **0.50**, **576/1,100 fixtures selected (52.36% coverage)**, **366/576 right (63.54% selected-pick accuracy)**.

- 2023/24: 201 picks, 67.16% correct.
- 2024/25: 196 picks, 61.73% correct.
- 2025/26: 179 picks, 61.45% correct.

Threshold **0.45**: 762/1,100 selected (69.27%), 456/762 correct (59.84%). Threshold **0.40**: 967/1,100 (87.91%), 542/967 correct (56.05%). These alternatives were inspected retrospectively and **must not** be selected post-hoc by cherry-picking the highest result; 0.50 represents a simple, easily explicable policy.

## 2026/27 matched frozen predictions (GW6–GW10 only)

Matched all 50 fixture identities from:
`data/live/challenger_predictions/2026_27_gw{06..10}_market_shadow_predictions.csv`
and `data/live/results/2026_27_gw{06..10}_results.csv`, with correct **quoted CSV parsing** (market source fields contain commas).

- Existing official fixed 40/60: **25/50 correct (50.00%)**.
- Same pre-specified confidence >=0.50 rule: **21/50 selected (42% coverage), 13/21 correct (61.90%)**.
- Snapshot timing of existing live predictions is preserved; GW8 and GW10 have documented T-24 exceptions.
- This 21-match sample is very small and selected retrospectively after outcomes are available. The precise threshold was considered in the historical screen, so it must not be presented as fully independent preregistered verification.

**Crucial:** these selected-fixture picks are exactly the *same forecasts* as the baseline on those fixtures. Therefore selected-pick accuracy of 63.54% is **not** a +9.54pp improvement of model predictive ability on an identical fixture set; it is a **coverage/precision trade-off**. No evidence of enhanced full-fixture predictive accuracy.

## Time-of-information empirical check

Also tested existing frozen Model 2 (40%) plus closing average bookmaker odds (60%) from Football-Data source `AvgCH/AvgCD/AvgCA` on exactly the 1,100 historical fixtures:
- **596/1,100 (54.18%)** versus fixed original opening blend 594/1,100 (54.00%).
- Closing-average market **standalone**: **604/1,100 (54.91%)** and Log Loss 0.96299 versus opening-only **595/1,100 (54.09%)** and LL 0.96815.
- Closing model using the original 40/60 weighting does not meet +2pp and closing odds are at a different later forecast timestamp, so not comparable as if available T-24h.

## Decision

**Practical optional feature**: surface *high confidence* as a separate label for forecasts where existing 40/60 max probability >=50%, and track both all-fixture and selected-fixture accuracy with mandatory coverage reporting. No model retraining and no T-24 source changes. This is a **credible path to >56% accuracy on selected published picks**, demonstrated historically and in a small 2026/27 retrospective series, but does **not** satisfy the user's unchanged +2pp **all-fixture model accuracy** objective.

**Full-coverage objective**: still **NO PROVEN +2pp MODEL UPLIFT**. Avoid asserting a guarantee, using thresholds to redefine the denominator, or promoting another unverified model. Require genuinely independent new source information, a preregistered full-coverage challenger, and future prospective confirmation.
