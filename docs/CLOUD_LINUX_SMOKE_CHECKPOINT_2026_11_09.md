# Issue #9 / PR #14: Cloud Linux smoke evidence (9 November 2026)

## Decision

**GO: clean Linux deterministic analytics development baseline.**

**NO-GO YET: full cloud chatbot, private deployment, production switch, or provider onboarding.** No LLM cloud runtime, authenticated private application, provider billing controls or end-to-end Streamlit browser flow has been tested. The existing PR stays **draft and unmerged** pending remaining gates.

## Executed runs

- [Feature branch workflow run 42630244422](https://github.com/dhandan/football-copilot/actions/runs/42630244422): **completed successfully**
- [Pull request workflow run 42630255263](https://github.com/dhandan/football-copilot/actions/runs/42630255263): **completed successfully**
- On **Ubuntu 24.04** with **Python 3.13** on public GitHub Actions runners, no local PC/Ollama needed.
- Clean dependency installation: pass
- `pip check`: pass
- `python -m unittest discover -s tests -v`: **56/56 pass**
- `python scripts/validate_project.py`: pass for committed assets, database, 1900 matches/3800 team-match rows, reconciliations and deterministic analytics checks.
- New real-tool smoke: `get_team_record("Liverpool","2025/26")` returns 38 played, **17W, 9D, 12L**; `frozen_gameweek_predictions(8)` returns 10 frozen fixtures and three recorded variants. No Ollama invoked.
- No prediction files, production model weights, or default chatbot changed.

## Minimal cause-and-fix

The existing workflow condition required `github.head_ref` to equal the feature branch, which is absent on a direct `push` event. Replaced it with a guard permitting either the exact feature branch on push or matching PR head. Added a real source-backed analytics/forecast read to the existing workflow. No redundant workflow or new infrastructure was introduced.

## Remaining non-skippable gates

1. Full tracked files and Git history security review is not certified by the smoke test. The earlier local audit found one intentionally synthetic key-header test input; actual deployed Streamlit privacy and repository's native secret scanning remain unverified. No secrets should be printed or committed.
2. This was a temporary CI job, **not** a persistent cloud runtime; no browser-based app was started.
3. No private-authentication, deny-by-default user allowlist, cross-session quotas or hard £0 provider billing controls have been established.
4. No stronger free cloud LLM was evaluated on the 13-turn live conversational suite; local Qwen/Facts-first acceptance remains the latest reference.

## Next scoped action

Proceed to the previously agreed **£0 private cloud development environment / provider feasibility gate**: compare exact free-quota and privacy terms; preserve a usable local Ollama fallback; test a provider-neutral read-only V2 response behind explicit opt-in. Do not start a billable service or advertise cloud deployment before hard guardrails. Keep PR #14 draft until scope and security review are approved.
