# V2 H07 topic-switch user bug — fixed and verified 9 November 2026

## Reproduction from the user's browser

The user opened the private Codespaces Streamlit preview and asked in order:
1. Compare Liverpool and Arsenal over their last 10 matches.
2. Who has been defensively better over those 10 matches?
3. Why is that?
4. Why was H07 rejected?

**Observed defect:** turn 4 repeated the Liverpool–Arsenal defensive explanation. The execution trace had `deterministic_form_route`, two `form_summary` calls, and **zero Groq requests**. This was not an H07 answer.

## Root cause and targeted remedy

Earlier `asks_why and prior_q` logic treated **every** question containing "why" as a reference to the previous form comparison, even when the question explicitly named a new subject (H07). The router silently reattached the old ten-match window, recovered both team names and selected `form_summary` twice before model planning.

Now only **genuinely referential short forms** such as `Why?`, `Why is that?` or `Explain that` may inherit the previous two-team context. An explicit new subject such as `Why was H07 rejected?` bypasses that carryover and calls the existing experimental planner to request `experiment_history(H07)`. No new tools, external data, retries or privileges were introduced. A straightforward sourced fallback now clearly explains H07's nonpromotion using its recorded 2025/26 metrics: H07 accuracy 51.28% versus Model 2 51.85%; Log Loss 1.0279 versus 1.0263; Brier 0.6168 versus 0.6152. Higher error scores are worse; this did not establish an improvement.

## Validation

- [Ubuntu CI run 42654315526](https://github.com/dhandan/football-copilot/actions/runs/42654315526) (initial repair): **79/79 tests**, **18/18 offline conversational turns**.
- [Live Groq run 42654530463](https://github.com/dhandan/football-copilot/actions/runs/42654530463): **18/18 live conversation acceptance**. In the user-sequence's H07 turn: `planned 1 tool(s)`, `executed experiment_history`; Groq planning called, Groq rewrite rejected by grounding, deterministic H07 evidence preserved. The H07 question no longer inherits Liverpool–Arsenal context.
- [Ubuntu CI run 42654888336](https://github.com/dhandan/football-copilot/actions/runs/42654888336) after clearer factual H07 fallback: **79/79 tests**, **18/18 offline conversation turns**.
- Preview now has an explicit sidebar version caption: `Build: H07 topic-switch fix · 9 Nov 2026` so a user can distinguish the updated code from a running old Codespace.

## Limitations and current decisions

Groq GPT-OSS 20B **answer rewriting remains NO-GO**. It still failed factual grounding for the H07 rewrite, so an answer constructed from read-only experiment-history evidence was shown. The accepted 18/18 results **do not imply Groq-generated responses are reliable**.

The first three user answers report statistics derived from completed gameweek snapshots and 2025/26 historical matches (a mixed-season last-ten window), not pure current-season form, and the defensive `Why is that?` explanation remains descriptive rather than causal. Current model data, default app, predictions and frozen forecasts were not altered. Draft PR #14 remains unmerged.

**Manual smoke for the user:** update the existing feature-branch Codespace, confirm build caption, click **Clear conversation**, and repeat the four turns. Last answer must include **H07**, the accuracy/probability-score comparison, and `experiment_history` in its tool trace, not form_summary. The port must remain PRIVATE. No further Groq key action required.
