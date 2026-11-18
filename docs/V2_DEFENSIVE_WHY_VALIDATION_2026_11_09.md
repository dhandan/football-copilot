# V2 defensive "why" insight — validated increment (9 November 2026)

**Status: IMPLEMENTED, TESTED, EXPERIMENTAL BRANCH ONLY.** No production/default Streamlit promotion or prediction-model changes.

## Problem solved

Existing V2 could answer who conceded fewer goals but did not explain the *observed record* behind the difference. It also lacked an explicit, validated short "Why is that?" follow-up with a carried-over comparison and time window.

## Implementation

- `agent/football_tools.py`: read the existing verified **2026/27 completed gameweek results**, optionally fill a longer match window from the existing **2025/26 DuckDB** source, calculate for each team:
  - `GoalsAgainst` and per-match rate derived deterministically in the answer
  - `CleanSheets` (zero goals conceded)
  - `Conceded2Plus` (at least two goals conceded)
  - `ConcededRecent5`, `ConcededPrevious5` (only when 10+ completed matches are available; verified chronological order)
- `agent/defensive_explanation.py`: explanation function uses only valid evidence, tells the user whether the match-by-match breakdown is unavailable, and explicitly refuses unsupported causal inferences about injuries, tactics, shot quality, goalkeeping or opposition strength.
- `agent/langgraph_v2.py`: recognised defensive "why" questions and short follow-ups reuse the two existing read-only `form_summary` calls and prior team/match window. For unrelated "why" questions, do not invent a defensive explanation. Distinguish recent-form tables from actual head-to-head meetings.
- `tests/test_v2_defensive_why.py`: six tests, including real verified result snapshots, accurate explanation of mocked distributions, context preservation, missing fields and unequal match windows.
- `scripts/evaluate_v2_conversations.py`: **the original 13 conversation turns remain intact**; add four direct/contextual "why" turns with evidence and provenance assertions.
- Existing CI workflow now runs both full unit regression and the 17-turn offline agent suite before DuckDB and immutable forecast smoke.

## Measured acceptance

- [Ubuntu push CI 42647044883](https://github.com/dhandan/football-copilot/actions/runs/42647044883): **77/77 tests**, **17/17 conversation turns**, source/read-only assets and model validation successful.
- [Ubuntu PR CI 42647059559](https://github.com/dhandan/football-copilot/actions/runs/42647059559): successful independently.
- All four new scenario turns use deterministic tool routing and actual data; no local Qwen or Groq call required for those test questions.
- The reported offline mean of 0.05 seconds and max of 0.09 seconds is **not a cloud-LMM benchmark**, since fixed test plans and deterministic routes dominate.
- Earlier intermediate commits had assertion failures and were corrected; both final workflow paths are green.

## Interpret correctly

This is an **evidence-backed descriptive explanation** of the outcomes. E.g. goals conceded, clean sheets and 2+ conceded are calculated from results, **not an identified causal reason** that a team's defence changed.

Known constraints: current supported two-team alias set is limited; match data do not provide shots, chances, line-ups, pressing, opponents' strength or player events. A future "why" question outside this coverage must not produce a tactical story from aggregate results.

## Next worthwhile V2 research increment

Test a broader evidence-grounded "why" capability using *legally obtained*, completed-match-level opponent-adjusted and/or home-away measurements with clear provenance. Separate descriptive associations from causal explanation; test on scenarios unseen by development before extending the agent planner. Do not change models, forecasts, tool permissions, default chatbot or Groq NO-GO status until independent acceptance warrants it.

This increment is on the existing experimental branch and **draft PR #14 remains open/unmerged**, in need of scope review before any production promotion.
