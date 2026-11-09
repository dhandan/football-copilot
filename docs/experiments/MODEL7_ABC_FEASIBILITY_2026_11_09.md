# Model 7 A/B/C: evidence-led feasibility gate (9 November 2026)

**Decision: CONDITIONAL GO for a bounded A/B/C historical feasibility test, NO-GO for a full production build or any claim that +2pp has been established.**  
**Type:** repository/source audit and matched-fixture feasibility assessment. This document is **not a trained CatBoost or event-enriched Poisson accuracy backtest**.

## Fixed question

Can changing (A) modelling technique alone, (B) data alone, or (C) both improve 1X2 accuracy by at least +2.00 percentage points beyond the original fixed 40% Model 2 / 60% opening-market benchmark, without unacceptable Log Loss and multiclass Brier degradation?

- Fixed historical benchmark: **594/1,100 = 54.00%**. Required threshold on exactly the same cohort: **616/1,100 = 56.00%**, net **22 extra correct selections**.
- This 1,100-match development cohort is **repeatedly examined**, including H07/H08/H09 and earlier models. Do **not** describe it as an independent untouched test.
- Candidate probability forecasts should be compared **standalone** and by substituting the football 40% component, with identical 60% saved opening-market probabilities, pre-registered weights and same fixtures. Do not optimise blend weights on the evaluation outcomes.
- GW1–GW10 2026/27 results are known to us, so a locked retrospective chronological replay can be informative, **not unseen/independent proof**.

## Newly verified source: Premier-League-Stats event aggregates

Repository: https://github.com/imadeddine-belkat/Premier-League-Stats

GitHub source files inspected directly:
- `pl_stats/_merged/events/2023-24_events_stats.csv`: 760 team-match records, 380 distinct fixtures
- `pl_stats/_merged/events/2024-25_events_stats.csv`: 760, 380
- `pl_stats/_merged/events/2025-26_events_stats.csv`: 760, 380
- `pl_stats/_merged/events/2026-27_events_stats.csv`: **200, 100**, exactly GW1–GW10 (10 matches each)
- Historical 2019/20 and 2021/22, 2022/23 source files also inspected; all 760 team-match rows per year.

**Canonical reconciliation (date + home + away after known club-name aliases):**
- 2023/24–2025/26 Football-Data.co.uk source CSVs: **1,140/1,140** match identities exactly aligned to the event source.
- 2026/27: **100/100** saved `data/live/predictions/2026_27_gw01...gw10_predictions.csv` fixture identities matched the event source after `Coventry -> Coventry City`, `Hull -> Hull City`, `Man Utd -> Man United`, `Spurs -> Tottenham`. This is a key/coverage reconciliation, **not** an audit of numerical validity or point-in-time publication.

**Core candidate features with high measured population**:
- `totalScoringAtt`, `totalFinalThirdPasses`, `finalThirdEntries`, `possessionPercentage`: **760/760** nonblank team rows for 2023/24, 2024/25, 2025/26; **200/200** for 2026/27.
- `attemptsIbox`, `attemptsConcededIbox`: **759/760** in 2023/24, **758/760** in 2024/25, **759/760** in 2025/26, **200/200** in 2026/27.
- `touchesInOppBox`: **760/760** in 2023/24, 2024/25, 2025/26 and **200/200** in 2026/27.
- `possWonAtt3rd`: 738/760, 729/760, 731/760, 191/200, respectively.
- The archive contains ~190 columns per recent season, **but many sparsely filled fields must not be silently converted to zero**; test missingness/definition and time lag.

**Crucial xG mismatch**:
- `expectedGoals` nonblank: **4/760** rows in 2023/24, **340/760** in 2024/25, **760/760** in 2025/26, **200/200** through GW10 of 2026/27. In 2021/22: 6/760 and 2022/23: 4/760. Therefore **DO NOT** build a long-history model assuming consistently observed xG from this archive. Predefine a common-feature core **without** xG; a newer-year xG add-on would be a distinct, lower-power sensitivity analysis, not the main B/C feature set.

**Live source caveats:** data source provides published *post-match* event statistics, not archived exact pre-match snapshots. Only prior completed fixtures can contribute to the next forecast; prevent same-day and after-freeze leakage, and test revisions/actual availability by time. Data README says MIT, but underlying content is sourced from Premier League / PulseLive services; independently verify licence/redistribution rights and practical continuity before production use. A public repository or stated repository licence does not automatically authorise all underlying feed data.

**Alternative:** openplay (https://github.com/nlbair/openplay) has 4,556/4,560 Premier League matches 2014/15–2025/26 and many event fields, but **no 2026/27 feed**, so is a less directly deployable primary source unless a same-definition live bridge is proven. Its release is a preview and underlying rights also deserve review.

## Experiment design and readiness

| Arm | Training | Features | Data feasibility | Next test |
| --- | --- | --- | --- | --- |
| Control | Existing Poisson Model 2 | Existing 25 | Original `prediction/build_features_v2.py` and `data/processed/matches_clean.csv` available; generated `prediction_features_v2.csv` is gitignored | Reproduce controlled 1,100 reference; preserve frozen Model 2 |
| A | CatBoost direct multiclass H/D/A | Exact original 25 | **GO** for bounded prototype, additional free package required (CatBoost not in project requirements) | Chronological CatBoost vs baseline, no market features internally |
| B | Two-Poisson revised retrained challenger | 25 + small fixed lagged event feature set | **CONDITIONAL GO**, 1,140/1,140 historical match joins and 100/100 2026/27 joins established | Common-feature completeness, gameweek-lag, and known-available-date audit |
| C | CatBoost direct multiclass | Exact same 25 + same event feature set as B | **CONDITIONAL GO** subject to B feature gate | Compare to A and B under identical folds |

Use chronological 2023/24, 2024/25, 2025/26 test seasons; fit/tune on earlier seasons only with nested temporal hyperparameter selection. Do not tune on 2026/27. Freeze the *same candidates* and replay 2026/27 GW1–GW10 with lagged prior-match stats, independent of current fixture outcomes.

The project's existing fixed 40/60 market prediction snapshots are **actually frozen only for GW6–GW10 (50 matches)**. Model 2 predictions are frozen for GW1–GW10 (100). Historical `data/raw/E0_2627.csv` has only **50 early-season fixtures**; using historical opening-odds fields for GW1–GW5 would reconstruct a **retrospective** blend and the precise pre-match capture time is unverified. Thus **DO NOT label a 100-match 40/60 reference as prospectively frozen**. Use the 50 protected GW6–GW10 forecasts as the strong clean real-season comparator; report a separate retrospective 100-fixture cohort with metadata/limitations if feasible.

## Precision / attainable confidence

If a new model genuinely improves accuracy by **+2pp**, a 100-match 2026/27 comparison corresponds to **two additional correct matches**. The uncertainty of a paired accuracy difference depends on forecast disagreements. For illustrative **10–20% discordant fixtures**, a 2pp paired difference on 100 matches has an approximate 95% interval half-width of **6.2–8.8pp** (normal approximation; ignoring clustering/time series). An approximately 80%-powered two-sided 5% test for a true +2pp advantage would require roughly **1,950–3,900 independent matched matches**, under those assumed discordance rates. Those are *assumption-based precision calculations*, not measured model uncertainty. Weekly/season clustering can worsen precision.

The 1,100-match benchmark's previous screens achieved at best **597/1,100 = 54.27% (+0.27pp)** in a number of exploratory, repeatedly tested feature/model variants. That is **NOT** a calibrated point forecast for CatBoost + genuinely new event features. There is **no defensible confident estimate** of A, B or C's eventual uplift until those models are actually fitted and scored. For planning, assume **zero verified uplift** rather than claim likely +2pp; the 56% hurdle is a success criterion, not an estimate.

## Go / no-go conclusion

- **GO**: low-cost source reconciliation (done), bounded A/B/C modelling prototype if licensing, availability, feature-parity and archived 25-feature pipeline reproduction checks pass.
- **NO-GO**: full production build/promotion now, or claims of >=2pp; no fitted A/B/C results exist yet, 2026 100 is not an unseen holdout, and pre-match source availability and rights are unverified.
- **Immediate next logical test:** build deterministic, well-tested 10–15-feature **non-xG** rolling feature table using the consistent PL event stats fields. Run independent seasonal walk-forward A/B/C against original frozen 40/60 historical benchmark. Report each arm's raw football-only accuracy, fixed 40/60 substituted blend accuracy, Log Loss/Brier, by-season performance, fixture-level switches, and paired interval. Stop if none approaches the 616/1,100 development hurdle; do not brute-force many feature/hyperparameter sweeps.
- Only a version frozen **before GW11 or later fixtures** can provide genuinely new prospective confirmation. Keep V2 conversational AI roadmap primary; forecast research is bounded.

**Implementation status:** inspected GitHub sources, counted source coverage and missingness, reconciled 1,140 historical + 100 current-season fixture identities, and calculated statistical precision. **No new CatBoost or B/C model trained, no accuracy-uplift claim, no production code changed**.
