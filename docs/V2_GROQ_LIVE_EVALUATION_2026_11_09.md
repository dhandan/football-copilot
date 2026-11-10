# Groq Free V2 live evaluation: first real run (9 November 2026)

## Decision: GO for experimental connectivity, NO-GO for Groq V2 promotion

**User-attested Free plan and Groq Zero Data Retention.** `GROQ_API_KEY` was supplied privately through GitHub Actions and the credential preflight passed. This does not independently verify Groq's account billing or ZDR configuration.

**Run:** https://github.com/dhandan/football-copilot/actions/runs/42639582016 , rerun attempt 2, job 129239074484; isolated draft PR #14 research branch. The initial attempt failed before sending a Groq request because the secret was unavailable; the new attempt passed and sent live Groq inference requests.

| Metric | Observed result |
|---|---|
| Existing 13-turn V2 end-to-end acceptance | **13/13 PASS** |
| Mean wall-clock latency | **0.86 seconds** (one run) |
| Maximum wall-clock latency | **3.14 seconds** |
| Actual Groq planner calls | 2 turns, 2 tool plans returned successfully |
| Groq synthesized answers attempted | **4** |
| Generated answers accepted by grounding | **0 / 4** |
| Syntheses rejected by factual grounding | **1 / 4** |
| Syntheses failing provider/response contract | **3 / 4**, surfaced only as `GroqUnavailable`, exact root cause not yet diagnosed |
| Other deterministic-only turns | 7 (no remote generation required) |

### Cases and honest interpretation

- All expected tools were selected, with retained team and gameweek context and correct safeguarded responses.
- Crucially, the 13/13 suite **does not demonstrate better Groq-generated answer quality**: fallback satisfied the assertions on all four synthesis attempts. Three failures surfaced only as the current broad `GroqUnavailable` category, not enough evidence to attribute to model, rate limit, API HTTP status or output schema.
- 0.86s mean is not a fair all-LLM-vs-all-LLM speed comparison. Seven fast deterministic-only turns dominate; four syntheses were rejected/failed; compare separately by routed stage and accepted-synthesis quality.
- Existing reference: Qwen3 0.6B 13/13, mean 6.85s/max 25.40s, only 1/4 synth accepted; opt-in facts-first 13/13, 1.63s/max 11.66s. Different single runs and environments.
- No files, odds, forecasts, model weights or default Streamlit mode were changed. No secret values logged by evaluation code.

### GO/NO-GO and next bounded action

**GO:** Groq Free authentication, live provider access, strict read-only tool/grounding safeguards and zero Ollama dependency for that run. Provider is technically feasible as an opt-in V2 experiment.

**NO-GO:** Claiming a smarter assistant, declaring Groq superior to Qwen or facts-first, merging to production or enabling public/private cloud hosting. Four syntheses yielded **zero accepted explanations**, requiring investigation.

**Next:** Add *safe categorical* adapter diagnostics (HTTP status category, JSON-response-contract failure, rate limit vs transport), without logging auth headers, request/response bodies, user prompts or verified facts. Run one bounded controlled test to classify the three failures and fix only an evidence-proven cause; keep fallbacks. Then repeat the same 13-turn live suite, review accepted synthesis content, grounding, stage latency and free-tier budget. No automatic model-hopping, unfounded quality claims or paid upgrade.
