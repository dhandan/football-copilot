# H03: Draw discrimination — reproducible screen (24 November 2026)

**Status: empirical NO-GO for the tested draw-aware adjustment; diagnostic pipeline implemented. No model promotion.**

This milestone turns draw-discrimination research into repeatable Python and GitHub Actions tests, while retaining the pre-registered champion and all frozen snapshots unchanged. It is not a claim that draw recognition is a solved problem.

## Original historical reference, 1,100 fixtures (2023/24–2025/26)

Source: `reports/model6/model6_walk_forward_predictions.csv`, a **previously used development cohort** (not fresh hold-out). `scripts/h03_historical_diagnostics.py` verifies exact fixture identity, original vectors, draw count and required **594/1,100** fixed 40/60 matches, then produces fixed probability-band diagnostics.

| Method | Correct | Accuracy | Draw picks | Actual draws | Mean draw probability | Draw-only Brier | Log Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Model 2 | 574/1,100 | 52.18% | 0 | 268 | 22.81% | 0.18417 | 1.00375 |
| Opening market | 595/1,100 | 54.09% | 0 | 268 | 23.33% | 0.18293 | 0.96815 |
| **Fixed 40/60** | **594/1,100** | **54.00%** | **0** | **268** | **23.12%** | **0.18317** | **0.97607** |

**Interpretation:** No modal draw picks does **not** mean the models assign draws zero probability. A draw can have about 23% probability while the home or away win is more likely individually. The precise bottleneck is weak discrimination of drawn fixtures *relative to the win classes*. Generic threshold increases must not be promoted merely to force draws into the top class.

### Draw calibration by fixed baseline-probability bands (historical)

| Fixed blend draw probability | Fixtures | Actual draws | Observed draw rate |
|---|---:|---:|---:|
| ≤15% | 49 | 3 | 6.12% |
| 15–20% | 147 | 34 | 23.13% |
| 20–25% | 487 | 118 | 24.23% |
| 25–30% | 417 | 113 | 27.10% |
| >30% | 0 | 0 | — |

Binned statistics are descriptive, not an optimised draw-selection threshold; sample sizes and empirical reuse limit generalisation.

## One fixed, simple challenger (v1) and no-leak test

`scripts/h03_draw_discrimination.py` raises/lowers the fixed blend draw probability on a **fixed rule** using similarity of Model 2's expected goals and their combined total; home/away remaining probability retains its original ratio. Parameters are constants defined before these CI tests, not fitted on GW6–GW11 results. No changes to production model code.

- GW1–GW5: **50 historical, retrospective opening-odds comparisons**, explicitly developmental (source market odds are retrospective, not identical provenance to the prospective shadow).
- GW6–GW11: **60 prospectively frozen, previously unseen-by-this-rule match forecasts** (held out from fitting; by 24 November their results were publicly known, so this is a *post hoc research test*, not truly prospective validation of the new H03 challenger).
- Exact per-fixture keys and source provenance checked; outcomes enter scoring only; independent scoreline checks and official snapshot flags required.

| Split | Method | Correct | Accuracy | Log Loss ↓ | Brier ↓ | Actual draws | Draw picks |
|---|---|---:|---:|---:|---:|---:|---:|
| GW1–GW5 retrospective (50) | Fixed 40/60 | 23 | 46% | 1.04783 | 0.62943 | 16 | 0 |
| GW1–GW5 retrospective (50) | H03 v1 | 23 | 46% | 1.04154 | 0.62572 | 16 | 0 |
| GW6–GW11 frozen forecasts (60) | **Fixed 40/60** | **30** | **50%** | **1.03314** | **0.62161** | 14 | 0 |
| GW6–GW11 frozen forecasts (60) | H03 v1 | 30 | 50% | 1.03478 | 0.62471 | 14 | 0 |

**Decision: reject H03 v1**. It neither raises 1X2 accuracy nor improves either overall probability-quality metric on the 60-match held-out period. Do not adjust its parameters in response to the observed held-out results and call that validation.

## Continuity with previous project decisions

- H08 previously rejected simple market-implied goal / over-under / Asian-handicap models and documented prior Dixon–Coles and generic draw-multiplier failures.
- H09 previously rejected retrospective offset tuning (592/1,100 versus original 594/1,100; **zero draw picks**), and determined that better independent data may be necessary.
- The original **+2pp hurdle remains 616/1,100** on exactly the same reference cohort, plus separate genuinely out-of-sample prospective validation. No H03 evidence meets that hurdle.

## Recommended next research gate — independent data, not more threshold tuning

Assess the legal, free historical `openplay` event data identified in H09 with fixture reconciliation and lagged match features, and establish whether a compatible T−24h live source exists. Focus only on evidence which may explain draw outcomes **beyond** the market and existing expected-goals estimates: parity of chance quality, low-quality-shot profiles, shot quality dispersion, opponent-adjusted chance suppression and plausible lineup availability with time-valid sourcing. If the new evidence cannot be sourced legally and point-in-time, stop before model training.

## Execution and verification

- `python -m unittest tests.test_h03_draw_discrimination -v`
- `python -m scripts.h03_draw_discrimination --output-dir artifacts/h03`
- `python -m scripts.h03_historical_diagnostics --output-dir artifacts/h03`
- [GitHub Actions H03 research workflow](../../.github/workflows/h03-draw-discrimination.yml) runs these commands with saved outputs as downloadable research artifacts. Run without Codespaces.
- Production Model 2, the fixed 40/60 champion, official GW1–GW11 predictions and verified results remain unmodified.

**Product status:** H03 measurement and reproducible research harness implemented; experimental draw-aware formulation rejected; no new predictive model released.
