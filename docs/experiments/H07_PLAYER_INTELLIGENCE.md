# H07 Player Intelligence Research

Status: Closed - no deployable improvement established (23 October 2026)
Priority: P1
Last updated: 23 October 2026

## Objective
Evaluate whether player availability, recent form and replacement
strength can improve football match predictions beyond frozen Model 2.

## Historical research

Historical Premier League seasons analysed: 2021/22 to 2025/26.

Candidate signals included:
- Player attacking contributions and recent form.
- Player availability and absences.
- Positional depth and replacement options.
- Team-level player-impact features.

## Validation findings

Chronological 2025/26 evaluation:

| Metric | Model 2 | H07 xG+xA |
|---|---:|---:|
| Accuracy | 51.85% | 51.28% |
| Log Loss | 1.0263 | 1.0279 |
| Multiclass Brier | 0.6152 | 0.6168 |

The tested H07 variant did not outperform Model 2.

Decision: Do not promote the historical H07 challenger.

## GW7 prospective development

- Official FPL data snapshot captured 13 October 2026.
- Player availability and attacking contribution diagnostics prepared.
- Replacement-quality methodology remains unvalidated.
- No calibrated H07 probability adjustments approved.
- H07 is observational for GW7.

## GW7 result and GW8 scope

GW7 frozen FPL snapshot captured 16 October 2026 (667 players across 20 clubs). Recorded statuses: 476 available, 105 FPL-unavailable, 63 injured, 20 doubtful, 3 suspended. These are pre-match observational flags, not evidence of causal match impacts. There was no separately frozen H07 forecast, so no valid GW7 H07 scorecard.

**GW8 decision:** Use existing H07 issue #10 as the prioritised **weekly optimisation** opportunity, starting with source, licensing, point-in-time knowability and leakage feasibility. If feasible, predefine historical validation metrics/guardrails, evaluate player availability, importance and replacement strength against the provisional 40/60 baseline, then consider a separately versioned release candidate **for the next available Gameweek** if it passes. If not feasible or it fails validation, reject it quickly and retain the existing verified baseline. Do not revive rejected historical player variants. The 40/60 blend remains the agreed provisional primary-release candidate pending H01 governance sign-off and a validated operational switch.

## Frozen model governance

Neither Model2_v1.0 nor the frozen 40/60 market challenger
has been modified by H07 research.

## Local research artefacts

Source scripts:
experiments/h07_player_impact/

Historical research outputs:
experiments/h07_player_impact/*.csv

Current player snapshots:
data/live/players/h07_snapshots/

Updated GW7 match history:
data/live/model2/matches_asof_gw07.csv

These datasets remain local pending review of size,
licensing, provenance and reproducibility requirements.

## Final decision: park H07 (23 October 2026)

Existing historical research assessed player form, injury/availability impact, squad depth, combined player features and xG+xA challengers. Some variants improved accuracy, but none showed a credible, independently validated improvement in overall Log Loss and multiclass Brier suitable for release. The independent 2025/26 xG+xA result was worse than Model 2 on all three reported measures. The lagged absence dataset has 3,800 team-fixture rows with substantial nonzero coverage (Last1_ImportanceKnown 89.7%, Last5_WeightedInjury 83.8%), but **coverage alone is not predictive lift**; a distinct validated gain against the current 40/60 primary was not demonstrated. Some retrospective availability/lineup sources lack verified pre-match publication timestamps, creating a leakage risk. No H07 model was frozen prospectively for GW7 or GW8.

**Disposition:** Close the current H07 feasibility/research issue as **not planned for further work now**, rather than claim that all conceivable player signals are disproven. Do not deploy or rerun rejected H07 variants. Reopen only with a specific, new point-in-time source or hypothesis and a demonstrable path to beating the current 40/60 release baseline on prespecified temporal validation. Keep local research artefacts untouched; do not add a replacement issue.

**Next improvement focus:** Work existing H05 disagreement analysis into a bounded diagnostic identifying potential 40/60 failure segments, without repeating the rejected Model 6D override. H03/H04 and H06 remain background scored diagnostics; H01/H02 remain weekly evaluation/data-quality controls.

## GitHub tracking

H07: https://github.com/dhandan/football-copilot/issues/10
H01: https://github.com/dhandan/football-copilot/issues/2
H02: https://github.com/dhandan/football-copilot/issues/3
