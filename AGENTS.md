# Football Copilot — Codex development instructions

This is an experimental, personal Premier League conversational-analytics copilot.
Treat this repository as a **read-only football facts** project, not an opportunity
to redesign the prediction model.

## Ground rules
- Work incrementally on `feature/issue9-cloud-readiness-audit`; do not merge to
  `main`, modify production preview or promote experimental Groq functionality.
- Historical and 2026/27 match data is the authority. Python tools calculate
  football facts; models only plan or explain within verified evidence.
- Do not fabricate results, team statistics, confidence, odds, benchmarks,
  experiments or GitHub test outcomes.
- Keep the 1X2 fixed 40/60 benchmark intact: 54.00% (594/1100); the +2pp
  hurdle is at least 56.00% (616/1100) on the same historical evaluation.
- No paid providers, new credentials, secret exposure or network calls in the
  default test loop. Never commit private keys or raw provider responses.
- Groq is opt-in only (user-confirmed Free plan, Zero Data Retention, configured
  secret). Never turn on Groq automatically in CI.

## Run before handing changes back
From the repository root in the existing environment:

```bash
python -m unittest discover -s tests -q
python -m scripts.audit_v2_investigator_answers
python -m scripts.compare_v2_investigator
```

Also run targeted unit tests for changed modules. If dependencies/data are
missing, disclose the blocker instead of reporting tests as passed. Avoid
needlessly reinstalling dependencies on every iteration.

## Existing CI
`.github/workflows/cloud-linux-smoke.yml` runs on pushes to this branch and
on relevant pull requests; it also includes the offline audit and deterministic
compiler comparison. Inspect actual GitHub Actions status and logs to verify
remote execution rather than assuming successful completion.

## Architecture focus
- Production and the private Streamlit preview must remain separate from the
  experimental investigator lab on private port 8502.
- `agent/investigator_v2.py` contains scope parsing, bounded tool proposals,
  deterministic answer rendering and conversation context.
- `agent/football_tools.py` calculates recent form using verified gameweek
  snapshots and historical team-match data. Preserve source attribution and
  distinguish descriptive comparison from causal evidence.
- `scripts/compare_v2_investigator.py` is a **limited tool-planning ablation**.
  Its deterministic compiler is not the whole existing copilot and its
  test cases do not establish novel LLM tool discovery.
- The default compiler/answer audits are offline controls, not tests of
  autonomous Groq reasoning. The current limited ablation measured 10/10
  in both arms, with Groq slower and no demonstrated advantage.
- Prefer adding substantively new investigative capability and independently
  scored questions to polishing repeated scripted examples.
