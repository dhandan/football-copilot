# H07 Player Intelligence Research

Status: In Progress
Priority: P1
Last updated: 14 October 2026

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

## Next steps

1. Complete the Friday GW7 single-run official freeze.
2. Capture updated player availability.
3. Continue evaluating player availability and replacement quality.
4. Register any new challenger before prospective assessment.
5. Retain frozen Model 2 and 40/60 benchmarks unchanged.

## GitHub tracking

H07: https://github.com/dhandan/football-copilot/issues/10
H01: https://github.com/dhandan/football-copilot/issues/2
H02: https://github.com/dhandan/football-copilot/issues/3
