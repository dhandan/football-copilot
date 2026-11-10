# V2 GroqCloud opt-in experiment — 9 November 2026

**Status: IMPLEMENTED on draft PR #14 feature branch; offline/mocked tests only.  No live Groq account or key has been connected.  No paid service, private deployment or production promotion.**

## Why

The existing V2 LangGraph agent's deterministic football tools and immutable forecasts are the source of truth.  The local Qwen 0.6B planner/synthesizer is the baseline; the opt-in facts-first path is the trusted low-latency comparison. GroqCloud is an additional **optional planner and answer writer** for more capable evidence-grounded conversations.  It cannot edit or recalculate source data.

Groq supports multiple documented model IDs including `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`.  Default experiment model: `openai/gpt-oss-20b`.  Actual model availability, throughput and per-organisation Free-plan limits require confirmation.

## Safety gates — all required before a request

1. Groq **Free plan** confirmed by account owner; **no Developer-tier upgrade, paid billable service, billing details or automatic paid fallback**.
2. Groq Data Controls **Zero Data Retention** enabled and personally confirmed by account owner.  The code cannot independently attest the account-level state.
3. Store `GROQ_API_KEY` in a **private runtime secret**, never in the public repository, logs or PR.  No key is stored in this feature.
4. Set the following *only in a deliberately controlled, non-public runtime*:

   ```text
   FOOTBALL_GROQ_ENABLED=1
   FOOTBALL_GROQ_FREE_PLAN_CONFIRMED=1
   FOOTBALL_GROQ_ZDR_CONFIRMED=1
   FOOTBALL_GROQ_MODEL=openai/gpt-oss-20b
   GROQ_API_KEY=<a private key provided through the runtime secret manager>
   ```

All flags are opt-in.  Any missing switch/key/unknown model refuses remote inference.  The current V2 tool router continues to reject destructive/unapproved tool plans.  Data disclosed to Groq for enabled requests may include questions, model-performance statistics and short prior-conversation context; **do not use with private/personal data**.

The experimental adapter uses one fixed HTTPS endpoint, no redirects or automatic retries, connect/read timeouts 3s/12s, at most 320 generated tokens per request, a capped input, a fixed local cap of 24 requests per process (not cross-process or account-wide), and treats 429/4xx/5xx as failures without retry.  Groq responses are checked against Pydantic contracts and the existing independent evidence-grounding validator; invalid output returns original deterministic sourced facts.  **This is not yet sufficient public-deployment cost/abuse control**: provider account limits and cross-process global quotas still require separate verification.

## Test strategy

- `python -m unittest discover -s tests -v` — includes mocked configuration, missing key, unapproved model, response contract, 429/402/403, network timeout, no retry, request cap, deterministic facts fallback, and provider selection.
- `python -m scripts.evaluate_v2_conversations` — baseline 13-turn offline test, no cloud calls.
- **Only after Free, ZDR and private key confirmed:** `python -m scripts.evaluate_v2_conversations --live --groq` — identical 13-turn scenarios, record tool selection accuracy, factual faithfulness, synthesis acceptance/rejection, end-to-end and stage latency.  This command makes remote Groq requests; it is **not run in GitHub Actions**.
- Same 13-turn local Qwen 0.6B reference: 13/13 PASS, mean 6.85s, max 25.40s (single run); facts-first: 13/13 PASS, mean 1.63s, max 11.66s (single run).  These are reference observations, not controlled repeat measurements.

## Next decision

**GO for bounded opt-in credential-free engineering integration and mocked validation** once CI passes.  **NO-GO for claims of improved model quality or cloud deployment** until live model comparison, account Free/ZDR assurance, private runtime and global quota controls have been independently verified.

Sources:
- https://console.groq.com/docs/models
- https://console.groq.com/docs/structured-outputs
- https://console.groq.com/docs/your-data
- https://console.groq.com/docs/rate-limits
- https://console.groq.com/docs/billing-faqs
