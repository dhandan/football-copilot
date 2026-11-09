# H12 cross-market Asian handicap/total goals screen

**Decision: NO-GO: +2pp all-fixture target unmet**

This is a fixed, bounded evaluation, not a claim of independent performance.
Models were fitted chronologically with previous seasons only. No closing odds, results or same-match event features used.
Historical 1100 fixture cohort has been inspected repeatedly, so positive findings would require new prospective confirmation.
The 40%/60% blend holds the existing 60% archived market probabilities fixed and replaces only the 40% football component.
Markets used: pre-match AvgH/AvgD/AvgA, AvgAHH/AvgAHA, AHh, Avg over/under 2.5.
All wagers and implied probabilities are used only for forecasting research, not betting advice.

| Candidate | Standalone | 40/60 substituted | Uplift | LL | Brier | paired 95% CI |
|---|---|---|---|---|---|---|
| market_only_recalibration | 599/1100 (54.455%) | 596/1100 (54.182%) | +0.182pp | 0.96904 | 0.57693 | [-1.0, 1.364] |
| market_plus_ah_ou | 591/1100 (53.727%) | 596/1100 (54.182%) | +0.182pp | 0.96927 | 0.5772 | [-1.091, 1.455] |
| ah_ou_without_1x2 | 596/1100 (54.182%) | 599/1100 (54.455%) | +0.455pp | 0.96894 | 0.57705 | [-0.727, 1.636] |

## Reproduce
- Script: experiments/h12_cross_market/screen.py
- Output: reports/h12_cross_market/results.json
- Fixture probabilities: reports/h12_cross_market/predictions.csv

## Deployment constraint
The project's live bookmaker feed supplies 1X2 odds only; sourcing equivalent Asian handicap and goal-total prices at the matching T-24 freeze remains unverified. Even a historical PASS would not justify production.

