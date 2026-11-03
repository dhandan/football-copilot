# H08: market-residual learning (research only)

**Status:** implementation committed, **not yet run or validated**. No GW10 forecast or production model changed.

## Reuse and non-repetition audit
- Reuse existing `prediction/build_market_data.py` and `prediction/calculate_market_probabilities.py` (average pre-match market odds, no-vig H/D/A).
- Reuse existing `prediction/build_features_v3.py` (strictly earlier-date match form and strength attributes). Do **not** rebuild the odds pipeline, Model 4 direct 1X2 GBM, Model 3A/3B goal/draw calibration, Model 5 xG, Model 6D overrides or parked H07 player features.
- H08 differs from Model 4: bookmaker H/D/A **log-probabilities are fixed offsets**; a jointly regularised three-class linear correction uses selected lagged football features to learn *market residuals*.
- Exclude `EloDifference` until same-day ordering in the legacy v3 feature builder has been separately audited. Avoid outcome, goals from the match being predicted, closing odds and post-match xG.

## Reproducible inputs
- `data/processed/market_probabilities.csv` from the existing market builder.
- `data/processed/prediction_features_v3.csv` from the existing feature builder.
- Both are **generated** outputs and may not be committed. The builder stages already belong to `scripts/build_project.py`.
- Historical market source contains 380 valid AvgH/AvgD/AvgA triples in each completed season 2021/22–2025/26 (1,900 raw fixtures). Feature-builder eligibility can reduce the matched sample and must be reported by the script.

## Evaluation and limits
- Train on 2021/22–2022/23, select L2 penalty on 2023/24 only.
- Chronologically test 2024/25, then refit on 2021/22–2024/25 before testing 2025/26 (the selected penalty is not changed).
- Compare paired market-only and residual 1X2 accuracy, multiclass Brier and Log Loss on exactly the same matches.
- Historical 2025/26 was **already examined during earlier Model 5/H07 research**. These are chronological, not independently untouched, tests; no confirmatory or champion-promotion claims.
- The exact historical 40/60 baseline requires leakage-safe out-of-time **Model 2 probabilities on identical matches**, not the GW6–GW9 live series. It is deliberately **not** manufactured by this first script.
- Working hurdle: target at least 5% relative Log Loss reduction against matched 40/60 (once available), with no material Brier/accuracy deterioration, robust across years. Hurdle is aspirational and does not certify statistical significance.
- For future live shadows, data/source timing and bookmaker composition must be aligned; football-data.co.uk historical AvgH/D/A may differ materially from the UK Odds API Friday capture.

## Commands (from repository root)
```powershell
python experiments/h08_market_residual/run_h08.py --audit-only
python experiments/h08_market_residual/run_h08.py
python experiments/h08_market_residual/run_h08.py --write
```

The script makes no writes unless `--write` is supplied. Experimental outputs are never official pre-match forecasts. Do not claim performance until the script is executed and its output examined.
