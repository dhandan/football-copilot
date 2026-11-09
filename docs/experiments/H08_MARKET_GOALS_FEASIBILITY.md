# H08: Market-implied goal distribution — rapid feasibility screen

**Date:** 9 November 2026
**Disposition:** **NO-GO for full-scale implementation of these candidate formulations.** No evidence of meaningful uplift over opening 1X2 market. Not a blanket rejection of every possible handicap/totals formulation.

## Context and existing decisions
Football Copilot has already tested and rejected generic draw multipliers (Model 3B), alternative goal architectures including Dixon–Coles, xG-enriched Model 5 OOT accuracy, Models 6B–6E, and player-availability H07. The existing official Model 2, market-only, fixed 40/60 and adaptive GW6–GW10 snapshots remain immutable. The current fixed 40/60 prospective leader was **not** modified or trained in this exploratory screen.

## Existing data and coverage
Source files committed to `main`: `data/raw/E0_2122.csv`, `E0_2223.csv`, `E0_2324.csv`, `E0_2425.csv`, `E0_2526.csv`. All 1,900/1,900 matches have usable opening average 1X2 decimal odds (`AvgH`, `AvgD`, `AvgA`), opening average over/under 2.5 odds (`Avg>2.5`, `Avg<2.5`), an opening Asian handicap line (`AHh`), and final H/D/A result (`FTR`). Both sets of odds were converted from inverse-decimal prices to no-vig probabilities. **Historical snapshots' exact pre-match capture lead times were not independently verified, so this screen cannot establish live T−24h feasibility.**

## Screen A: chronological draw-probability incremental signal
A low-complexity logistic adjustment to the opening 1X2 draw probability was fitted on **earlier seasons only**, retaining the market's conditional home/away ratio. Three rolling training/test splits: 2021/22–2022/23 -> 2023/24; 2021/22–2023/24 -> 2024/25; 2021/22–2024/25 -> 2025/26. Features: (i) opening over/under 2.5 no-vig probability; (ii) optionally absolute opening AH line. An intercept-only offset adjustment was an internal control (not a revived draw-multiplier promotion candidate). Ridge regularisation and deterministic Newton updates were used; no feature hyperparameter search.

On 1,140 chronological test matches:

| Method | Correct | Accuracy | Log Loss | Multiclass Brier |
|---|---:|---:|---:|---:|
| 1X2 opening market | 618 | 54.21% | **0.965018** | **0.574014** |
| Intercept-only control | 618 | 54.21% | 0.965562 | 0.574242 |
| + Over/under signal | 618 | 54.21% | 0.965539 | 0.574287 |
| + Over/under and AH line | 618 | 54.21% | 0.966412 | 0.574797 |

No selected-result gains, and no overall probability-quality gains.

## Screen B: independent-Poisson, market-implied score distribution
A **zero-fitting** diagnostic translated the no-vig O/U 2.5 probability into Poisson total-goal mean `mu` by solving `P(Poisson(mu)>2)=P_market(over2.5)`. Two candidate splits into home/away goal means:
1. Choose home goal share by bisection so independent-Poisson `P(H)/P(A)` equals the opening 1X2 market's `P(H)/P(A)`. This keeps the market's directional home/away information and uses the over/under market to set expected goals.
2. Set approximate goal-mean difference to the negative opening Asian handicap line `-AHh`, bounded to positive home/away goal rates. This deliberately simple screen uses **the handicap line, not the full handicap price/settlement structure** and is not a fitted Asian-handicap pricing model.

H/D/A probabilities were obtained by summing an independent Poisson score matrix truncated at 18 goals per side (and renormalised).

| Method | Correct / 1140 | Accuracy | Log Loss | Multiclass Brier | Draw selections |
|---|---:|---:|---:|---:|---:|
| 1X2 opening market | **618** | **54.21%** | **0.965018** | **0.574014** | 0 |
| O/U with 1X2 home/away ratio | 617 | 54.12% | 0.966465 | 0.574767 | 0 |
| O/U with opening AH line | 611 | 53.60% | 0.967860 | 0.575339 | 0 |

Year-by-year:
- **2023/24**: market 224/380 (LL 0.909250), O/U + 1X2 223/380 (0.909849), O/U + AH 218/380 (0.914116)
- **2024/25**: market 206/380 (0.970552), O/U + 1X2 206/380 (0.971304), O/U + AH 209/380 (0.972084)
- **2025/26**: market 188/380 (1.015252), O/U + 1X2 188/380 (1.018242), O/U + AH 184/380 (1.017379)

None of the simple goal-distribution formulations beat the existing market's Log Loss in any of the three test seasons. None selected draws as the modal 1X2 outcome.

## Interpretation and limits
- **No claimed significant improvement.** Two different bounded formulations failed to improve on the existing 1X2 market benchmark. This does **not** demonstrate that full two-sided Asian-handicap prices, alternative goal distributions or point-in-time market microstructure could never work.
- Data coverage is good, but pre-match timestamps of these historical opening fields are not available to establish comparable T−24h information. The 2025/26 season was previously used for other project research and is **not a fresh untouched holdout** for general model promotion.
- The screen is against opening 1X2 market, **not the live fixed 40/60 blend** on exactly equivalent historical cohorts. The no-lift signal is sufficient to decline a larger speculative project, not to claim that every possible candidate would fail the fixed blend.
- No production Python source, live forecast, evaluation, configuration or historical data were changed. These calculations were performed transiently from original GitHub CSV content through the connected repository tool. This document contains enough method/column definitions and outcome totals for a later independent reproduction, but no reusable evaluation script was created for this rejected candidate.

## Decision
**NO-GO for a dedicated two-week build of the proposed H08 market-implied goal model.** The opportunity cost is not justified by these preliminary negative screens. Do not reopen as another small parameter sweep; require a materially different *independent signal* and demonstrably better feasibility/evidence before repeating.

Keep prospective official four-way GW11 testing unchanged and prioritise the broader Football Copilot V2 work. Future new-signal candidates should first pass an equally cheap data/provenance feasibility and incremental-information screen.
