# V2 Groq answer synthesis diagnostic closure — 9 November 2026

## GO/NO-GO

**GO** to retain Groq as an **opt-in experimental planner** and keep the read-only football tool execution and deterministic evidence-preserving fallback.

**NO-GO** to promote Groq GPT-OSS 20B as the V2 answer synthesizer, claim AI explanation quality gains or replace the existing Qwen/facts-first paths. **No production or cloud deployment approved.**

User explicitly confirmed Groq **Free** and **Zero Data Retention** account settings and securely configured a GitHub Actions secret. The code cannot independently verify the provider's account settings. No key was disclosed in GitHub files or logs.

## Same 13-turn live acceptance on fixed, existing evaluation

| Stage | Workflow | 13-turn acceptance | Mean | Max | Groq synthesized explanations accepted |
|---|---|---:|---:|---:|---:|
| Initial live Groq | [run 42639582016, attempt 2](https://github.com/dhandan/football-copilot/actions/runs/42639582016) | 13/13 | 0.86s | 3.14s | 0/4 |
| Safe error-class diagnostics (strict schema) | [run 42642758953](https://github.com/dhandan/football-copilot/actions/runs/42642758953) | 13/13 | 0.46s | 1.79s | 0/4 |
| JSON-object-format comparison (local schema and grounding retained) | [run 42643279316](https://github.com/dhandan/football-copilot/actions/runs/42643279316) | 13/13 | 0.39s | 1.46s | 0/4 |
| Final allowlisted provider error-code diagnostic | [run 42643851756](https://github.com/dhandan/football-copilot/actions/runs/42643851756) | 13/13 | 0.33s | 1.21s | 0/4 |

These single-run end-to-end latencies are **not directly comparable** with local Qwen runs. Most V2 scenarios use the deterministic fast path; the four generated explanations were all refused or rejected.

**Observed categories in final run:** three `GroqBadRequestJsonValidation` provider errors for substantive performance/evaluation summaries and one `synthesis_rejected_grounding` on H07 reasoning. The provider returns HTTP 400 with a recognised JSON-validation error code, not a missing key, timeout, or rate limit. Its exact raw error message and any potentially sensitive failed-generation payload were **never logged**. The broader JSON-object mode did not remedy the failures.

Groq's own documentation describes JSON/schema-validation-related HTTP 400 as a possible output-validation problem: https://console.groq.com/docs/structured-outputs and https://console.groq.com/docs/errors . That supports the error classification but does not establish the precise internal generation cause; avoid claiming unsupported certainty.

The application preserved the source-backed response on every failed synthesis. **13/13 acceptance is a fallback/reliability result, not evidence of stronger Groq explanatory quality.**

## Implementation and protection

- Added specific exception subclasses for status, transport, schema and allowlisted error-code categories; exceptions and execution traces contain **only static, non-sensitive categories**, not raw API error text.
- Added regressions for redaction, error classification, non-retry behaviour and fallback. **71/71 CI unit tests passed** before the final live code-check, including Linux/DuckDB smoke.
- The JSON-object experimental change was restored to the original stricter JSON Schema response format on the draft branch because there was no demonstrated improvement.
- No model weights, gameweek freezes, official forecast files, default Streamlit mode, or production deployment were changed.
- **Draft PR #14 remains open/unmerged.** Groq Free is only a local/CI experimental connection, not a multi-user private service with globally enforced usage budget.

## Next meaningful V2 decision

Stop retesting minor format/prompt variations of GPT-OSS 20B on the same four synthesis cases. Preserve Groq read-only planning, the strong facts-first reference, and user-facing grounded fallback. Evaluate a **genuinely different question/reasoning workflow** or a separately bounded alternative model only if it can show **accepted, evidence-backed explanations**, not just 13/13 tests through fallback. Any future candidate needs an independent qualitative/quantitative rubric, read-only source provenance, live reproducibility, privacy/cost controls, and explicit GO/NO-GO before promotion.
