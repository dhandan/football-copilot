# Accelerated champion/challenger experimentation policy

**Adopted 13 October 2026.** Supplements [programme issue #1](https://github.com/dhandan/football-copilot/issues/1) and the [Gameweek operating checklist](gameweeks/OPERATING_CHECKLIST.md).

## Decision

Decouple **rapid model development** from **controlled promotion**. We will not require each challenger to wait five successive Gameweeks before development of another begins.

1. **Weekly diagnosis and development:** Prioritise 1–3 evidence-led tests at a time (within the existing maximum of two active Kanban issues, including at most one new model-development experiment unless we explicitly revise capacity). Run historical walk-forward/temporal out-of-time comparisons rapidly; document attempted variants and reject weak ones. Previously inspected data is development evidence, never falsely described as untouched validation.
2. **Parallel prospective shadows:** Historically promising, independently versioned challengers may forecast the **same upcoming fixtures simultaneously**, alongside frozen Model 2 and the existing frozen 40/60 challenger. Register design, features, training cutoff, acceptance and guardrails **before** the relevant kickoff/results. Store predictions immutably with separate paths and provenance; retain model/market source comparability.
3. **Reviews every 2–3 Gameweeks:** Inspect paired match-level Accuracy, Log Loss, Brier, draw/cold-start segments, confidence intervals, stability and data quality. Reviews may **reject, retain or continue collecting evidence**. A review is not an automatic promotion.
4. **Controlled promotion only with defensible evidence:** Pre-register for each version the minimum independent validation evidence/horizon, effect size, guardrails, uncertainty procedure, sample size and stopping rule **before** using prospective results to make the promotion decision. Repeatedly looking at live outcomes and picking a winner without accounting for sequential monitoring or multiplicity is prohibited. Evaluate all candidates on the same fixtures, and distinguish exploratory comparisons from confirmatory ones. If evidence is inconclusive, continue the shadow trial without holding back other development.
5. **Production isolation:** `Model2_v1.0` stays the champion; `Model2_OpeningMarket_40_60_v1.0` stays a frozen shadow at weights 0.40/0.60. GW1–GW5 40/60 results were selection/development data; GW6 onward are prospective. **Never retune the same frozen challenger using GW6+ outcomes.** Any revision is a **new version**, with its own registration and future validation.

## Sprint responsibilities and evidence gates

- At each completed GW: freeze results, evaluate all existing registered shadows against identical actual fixtures, record what improved/worsened and update the Gameweek journal + Kanban.
- Within the same sprint: select next hypothesis based on historical and observed diagnostics, do bounded reproducible experiments and record unsuccessful variants.
- Before the next kickoff: freeze eligible new shadow prediction files and verify IDs, code version, timestamps and commit references.
- **No untested live promotion**; no accidental use of retrospective results to claim pre-match forecasts.
- Track prospective sample counts **per challenger**, since later entries have shorter live histories. Report multiple-comparison and sequential-selection uncertainty honestly.

## First application after GW6

The frozen 40/60 shadow achieved **4/10 accuracy, 1.0745 Log Loss, 0.6552 Brier** versus Model 2 **3/10, 1.1747, 0.7308** on GW6. This is favourable *first prospective evidence*, not a promotion. Prioritise the registered draw, scoreline/xG and cold-start diagnostics without recycling rejected calibrations. Preserve GW6 prediction artefacts unchanged, complete GitHub evidence and review, and start a separate versioned historical experiment only if it passes the registered development gate.
