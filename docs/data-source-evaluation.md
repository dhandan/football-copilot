# Football Copilot Data Source Evaluation

## Objective

Identify a reproducible source of genuine historical expected-goals data that can be used to test whether richer football information materially improves Football Copilot's predictive performance.

Required historical coverage:

- 2021/22
- 2022/23
- 2023/24
- 2024/25
- 2025/26

Future live coverage for 2026/27 is desirable but is not required for the initial historical experiment.

The historical experiment will be completed before investing in a production live-ingestion pipeline.

---

## Source 1: Understat via soccerdata

### Result

Rejected.

The `soccerdata` Understat connector failed on macOS ARM64 because its TLS dependency attempted to download a required native library and could not initialise.

This failure occurred before any Understat data was retrieved.

### Decision

Do not introduce this dependency into Football Copilot.

---

## Source 2: Direct Understat extraction

### Result

Rejected.

Football Copilot tested direct extraction from historical Understat Premier League pages for seasons beginning:

- 2021
- 2022
- 2023
- 2024
- 2025

The returned pages no longer exposed the historical `datesData`, `teamsData` or `playersData` structures expected by existing extraction approaches.

The HTML contained only generic page variables rather than the required football data.

### Decision

Do not continue reverse-engineering Understat.

A modelling project should not depend on an unnecessarily fragile scraping implementation when alternative data sources exist.

---

## Source 3: StatsBomb Open Data

### Result

Not suitable for the primary experiment.

StatsBomb Open Data provides high-quality event data and expected-goals information, but its free Premier League coverage does not span the continuous 2021/22–2025/26 period required by Football Copilot.

### Decision

Retain as a useful football analytics learning source, but not as the primary Model 5 historical dataset.

---

## Source 4: FBref-derived historical datasets

### Status

Preferred candidate for the initial historical enrichment experiment.

Available public datasets provide Premier League match-level information including:

- expected goals
- expected goals against
- shots
- shots on target
- possession
- match results
- team and opponent
- season information

The initial objective is to find a versioned dataset covering the Football Copilot historical modelling window and validate it against the existing match dataset.

### Proposed use

Use the external dataset only as an enrichment layer.

The existing Football Copilot historical results dataset remains the canonical fixture and result source.

External xG information will be joined to those fixtures using:

- season
- match date
- home team
- away team

with explicit team-name reconciliation and validation.

---

## Experimental principle

The xG recorded for the match being predicted must never be used to predict that match.

Observed historical xG will be converted into pre-match rolling features such as:

- rolling xG for
- rolling xG against
- xG difference
- 5-match xG form
- 10-match xG form
- home xG strength
- away xG strength
- xG trend
- goals minus xG

Only information from matches completed before the prediction date may be used.

---

## Current decision

Stop further algorithm-only experimentation on the existing feature set.

Proceed with historical data enrichment before building another challenger.

The next task is to acquire and validate an FBref-derived historical Premier League xG dataset.