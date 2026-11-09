# H09 / 56%+ accuracy challenge: broad, read-only feasibility screen

**Date:** 9 November 2026  
**Status:** NO-GO on the tested existing-data candidates. CONDITIONAL GO for feasibility of genuinely new event-level data only.  
**Scope:** exploratory research; no production implementation or prospective forecasts modified.

## Goal and fixed comparator

- Frozen historical comparator: 40% Model 2 + 60% opening market, **594/1,100 (54.00%)** on the exact original Model 6 historical cohort (2023/24–2025/26).
- 56.00% hurdle: **616/1,100**, or **22 net additional correct 1X2 predictions**.
- Recomputed benchmark probability scores from saved unrounded historical reference vectors: Log Loss **0.97607**, multiclass Brier **0.58104**.
- The 1,100-match historical cohort has been used repeatedly during prior model development. It is *development evidence*, not untouched independent validation.

## Source files inspected

- `data/raw/E0_2122.csv`, `E0_2223.csv`, `E0_2324.csv`, `E0_2425.csv`, `E0_2526.csv`: 1,900 matches in total. Data include prior-match goals, shots, shots on target, corners, cards, dates, and market odds.
- `reports/model6/model6_walk_forward_predictions.csv`: exactly 1,100 historical matched predictions with Model 2 and opening market H/D/A vectors. Matching on date and both teams reproduced **594 correct fixed-blend picks**, exact.
- Existing documentation including `reports/post_gw5/model_deep_dive.md`, Models 3A/4/5, H07/H08 research and GW10 journal was checked to avoid interpreting earlier failures as new ideas.

## Additional screens performed in a temporary analysis environment

**Do not interpret these as fully reproducible repository experiments.** Models were evaluated via short-lived in-tool JavaScript calculations reading GitHub contents. The complete exploratory scripts and a reusable runner were *not* added to the repository. No hyperparameter improvement is claimed as fresh validation.

### A: Non-market dynamic Poisson strength

Previously completed H09 screen: team attack and defence ratings updated from opponents and goal surprises, with season reversion and three predefined update speeds. On the same 1,100 fixtures:
- Best independent dynamic model: **585/1,100 (53.18%)**.
- Best dynamic substitute in existing fixed 40/60 blend: **589/1,100 (53.55%)**.
- Best 20% dynamic + 20% Model 2 + 60% market: **594/1,100 (54.00%)**.
**Decision:** no-go for this formulation, not a disproof of all possible dynamic models.

### B: Chronological richer feature models

Built strictly lagged (previous-date) short- and long-memory team form, home/away splits, goal strength, shots, shots on target, corners, estimated finishing and defensive performance, rest and Elo. All match outcomes were processed only after that day's prediction features were created.

Used 4 feature groups (goals/form, shots, compact combined, wide combined), 2 shrinkage values, and 2 training architectures (football-only multinomial model, and football adjustments around market probability): **16 model settings**. For each:
- 2023/24 model trained only on 2021/22–2022/23.
- 2024/25 trained only on earlier seasons.
- 2025/26 trained only on earlier seasons.
- Scored full 1,100 original fixed-blend fixtures. Evaluated standalone, 40% replacement + 60% market, and 20% new + 20% Model 2 + 60% market.

Highest 1X2 performance found:
- Football-only shots: **597/1,100 (54.27%)**, Log Loss 0.98662, Brier 0.58890.
- Combined compact football correction plus 40/60 market weighting: **597/1,100 (54.27%)**, Log Loss **0.96736**, Brier **0.57580**.
- Augment existing blend with a goals/form model: **597/1,100 (54.27%)**, Log Loss 0.97415, Brier 0.58000.
These **+3** correct predictions equal **+0.27pp**, not +2pp, and remain exploratory post-selection scores.

### C: Direct outcome-selection correction

Calibrated H/A/D selection offsets on prior-season *opening-market* outcomes only, applied selected offsets to the saved frozen 40/60 probabilities in each next test season.
- **592/1,100 (53.82%)**, Log Loss 0.97858, Brier 0.58232 versus 594/1,100 baseline.
- No draw selections; 268/1,100 actual draws. 
**Decision:** reject; do not revive generic draw/market threshold sweeps.

## Existing-source disagreement upper bound

On 1,100 matched fixtures:
- **450** were incorrectly selected by *all three*: fixed 40/60, independent Model 2 and opening market.
- In **20** additional cases where the blend was wrong, market-only was correct.
- In **36** additional cases where the blend was wrong, Model 2 alone was correct.
- This means a hypothetical perfect oracle selecting only among those existing three forecasts could at most correct another **56** errors before any adverse switches. It is not an achievable backtest; it reinforces the need for independent information.

The fixed blend also shows historical season deterioration: **211/360 (58.61%)** in 2023/24, **203/370 (54.86%)** in 2024/25, **180/370 (48.65%)** in 2025/26. The season-to-season variation must be diagnosed before assuming a reproducible +2pp.

## Genuinely new data candidate: openplay

Primary source: https://github.com/nlbair/openplay (v0.1.0 preview released 3 Nov 2026). The source reports **4,556/4,560 Premier League fixtures (99.91%) with event data from 2014/15 to 2025/26**. GitHub release includes `shots.parquet`, `player_match_stats.parquet`, `team_match_stats.parquet`, `lineups.parquet`, event files, source issues/gaps and hashes. Licence: **CC-BY 4.0**, attribution required. This is materially richer in shot quality, possession/progression, defensive events and player minutes than Football-Data.co.uk's match aggregates.

**Current limitations:** data preview unvalidated in Football Copilot; repo source data not downloaded or joined; historical fixture reconciliation, event integrity, out-of-sample benefit and pre-match source availability still untested. The snapshot stops at 2025/26, so it does *not* provide the current 2026/27 live feed. A same-definition live source and publication-lag integrity audit are required before any model could qualify for real live forecasting.

Cost alternative rejected: StrikerAPI publishes 500 free credits/month, but states a single 2025/26 Premier League shot-map season requires 3,800 credits; full historical research is therefore not a realistic immediate £0 solution. Source: https://strikerapi.com/pricing

## Decision and forward gate

**NO-GO:** Do not build or deploy another model from the same existing feature base in pursuit of a guaranteed 56%. None of the screened variants has passed the +22 correct hurdle. Do not repeat H07, Model 5 xG, Model 6, H08, or H09 without a materially new source or hypothesis.

**CONDITIONAL GO, DATA FEASIBILITY ONLY:** Assess openplay for:
1. 98%+ source integrity and exact fixture match to canonical historic PL; document known data issues and CC-BY attribution. All prediction features must be calculated using *only prior matches*, and same-day events may not leak.
2. A viable, legal, repeatable **£0 current-season source** for matching live event and player features, with capture times compatible with the target T−24h freeze. If absent, class as historical learning only, not live replacement.
3. Predefine a small, auditable independent feature shortlist (non-penalty open-play chance quality vs set-pieces; shot-quality distribution; chance creation/progression; opponent-adjusted defensive suppression; lagged player availability/rotation if point-in-time known). Benchmark individually, avoid a 584-feature overfit.
4. Only if source and signal gates pass, test independently trained, season-walk-forward model/ensemble against the exact unchanged 1,100-match 40/60 reference. Aim for **>=616/1,100**, with no material Log Loss/Brier deterioration and no single-season-only apparent advantage. Treat all previously observed historical cohorts as development, not untouched holdout.
5. Freeze a versioned prospective 2026/27 shadow, with immutable pre-match forecasts and pre-registered sample/stopping protocol, before claiming verified lift.

**No observed +2pp uplift currently exists.** The target remains an aspiration, not a promised or estimated improvement. Do not let this research displace the Football Copilot V2 AI learning/conversational roadmap without meaningful evidence.
