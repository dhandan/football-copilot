# V2 LangGraph milestone checkpoint — 23 October 2026

**Status:** Working experimental integration. Preserve as a reproducible baseline, not a replacement for legacy chat.

**Constraints:** Hobby/learning project; zero incremental recurring cost; local Ollama only; avoid expanding modelling scope. Original Streamlit chatbot stays default.

## Delivered
- Separate read-only LangGraph prototype in `agent/langgraph_v2.py`: bounded planner -> tool execution -> answer, with a maximum of three approved football data tools.
- Strict Pydantic/Ollama JSON schema for `frozen_gameweek_predictions`, `model_performance` and `experiment_history`. No paid APIs or unapproved tool actions.
- Model selection via environment variables: Qwen3 0.6B defaults to V2 planner and answer writer; Qwen3 4B remains installed/tested and selectable. **No automatic light/heavy model routing**.
- Conversation history/checkpointing in RAM with isolated `thread_id` sessions; not persistent across restarts.
- Experimental evidence synthesis with numeric/semantic guardrails and safe deterministic fallback. It is **not** a general factuality guarantee.
- Frozen GW8 display includes all ten actual fixtures and model picks, not merely their count.
- V2 opt-in toggle in Streamlit sidebar; existing chatbot continues as the default and offers broader football tools.
- Developer trace/debug capability, plus focused unit and conversation evaluation scripts.

## Evidence from local Windows runs
- **16/16 unit tests passed**: `python -m unittest tests.test_langgraph_v2 -q` (user-run, 23 Oct 2026).
- **7/7 offline conversational scenarios passed**: `python -m scripts.evaluate_v2_conversations`.
- **7/7 live conversational scenarios passed**: `python -m scripts.evaluate_v2_conversations --live` with local Qwen3 0.6B (after earlier 5/7 failure and corrections).
  - Latest live run latencies: 7.37s forecast, 16.64s GW8 comparison, 11.24s performance, 12.16s H07 follow-up, 19.14s three-source question, 0.01s unsupported, 0.01s destructive.
  - The GW8 comparison, champion-only answer and three-source answer reached the factual fallback (`synthesis_rejected_grounding`). A pass reflects correct/evidence-backed end result, **not** universal success of LLM synthesis.
- Both Qwen3 0.6B and Qwen3 4B achieved **12/12** on the separate schema-constrained tool-selection benchmark (single runs, not evidence of statistical reliability).
- The user visually confirmed Streamlit V2 displayed the actual GW8 match predictions after a display bug was fixed.

## Findings / limitations
1. Strict Pydantic schemas materially improved tool-choice results on the observed small benchmark; no broader reliability claim.
2. Qwen3 0.6B was considerably faster than Qwen3 4B on the user's CPU. Both models remain available; 4B is not automatically invoked by V2.
3. The lightweight planner occasionally omits contextual evidence or selects an unrelated tool. There are narrow guards for champion-performance follow-ups and clearly out-of-scope weather queries; these do **not** constitute general intent detection.
4. Grounded answer synthesis is still inconsistent; factual fallback is often used. Some checks are tailored to GW6–GW7 and H07, and general attribution/factual accuracy require more testing.
5. V2 currently exposes only three specialist tools. Legacy chatbot has broader historical team, form, league and on-demand prediction capability.
6. Local in-memory conversation state disappears on process restart. CPU latency for meaningful live conversations is typically seconds to tens of seconds.
7. No production readiness claim; maintain the existing default Streamlit path and separate experimental opt-in.

## Reproduce
```powershell
git switch experiment/h07-player-impact
python -m unittest tests.test_langgraph_v2 -q
python -m scripts.evaluate_v2_conversations
$env:FOOTBALL_AGENT_MODEL = "qwen3:0.6b"
$env:FOOTBALL_ANSWER_MODEL = "qwen3:0.6b"
python -m scripts.evaluate_v2_conversations --live
streamlit run app/streamlit_app.py
```
Enable **Use LangGraph V2** in the sidebar for experimental conversation testing. Requires the project's existing virtual environment/dependencies, installed Ollama and local Qwen3 0.6B; no new hosted services.

## Prioritised next steps (small side-project increments)
1. **Expand tool coverage thoughtfully:** expose existing trusted history/team/form/standings and on-demand prediction tools through LangGraph using typed, read-only wrappers; do not rebuild models or data pipelines.
2. **Test breadth and consistency:** add realistic multi-turn questions covering newly added tools, unsupported intents, incomplete data and factual claims; repeat live runs, track pass rate and latency by planner/tool/synthesis.
3. **Improve synthesis and presentation:** meaningful evidence-backed comparisons, source provenance, correct fixture tables, no invented claims; reduce unnecessary LLM calls and fallbacks.
4. **Evaluate light/heavy strategy:** compare Qwen3 0.6B planning with opt-in Qwen3 4B answer generation, only if measured quality gain warrants CPU cost.
5. **Consider persistent local checkpointing only when useful**, then review if V2 has sufficient feature coverage to replace legacy chat. No premature default switch.
6. V3 portable runtime and public demo remain separate lower-priority issues (#12, #9).

**Decision:** V2 MVP milestone complete; full legacy parity / promotion to default **not complete**. Keep issue #11 open for next incremental work and preserve this checkpoint.


## Mandatory session handoff and change-control rules — 26 October 2026

**These rules apply to ALL Football Copilot work, not just LangGraph V2. This is a £0 incremental-cost, part-time learning project. Time spent rediscovering prior decisions or debugging avoidable breakage is a project defect.**

1. **Review before changing:** At the start of each session, read this handoff, the relevant gameweek journal, relevant GitHub issue and the exact current scripts, input schemas and previous working examples. Distinguish completed work from outstanding work. Do not change code or request commands until the existing process and data contracts are verified.
2. **No speculative fixes:** Reproduce or establish the cause from code and actual evidence, compare with last working gameweek, make the smallest justified change, and run available tests. State clearly what was tested in GitHub versus what still requires the user's local environment. Never present a commit as a verified fix until validated.
3. **One authoritative record:** Update this document with the last verified results, blockers, decisions, next single action and exact validation evidence whenever meaningful progress is made. Keep the relevant issue linked; avoid creating fragmented restart plans. Gameweek-specific immutable forecasts and retrospective findings remain in their own `docs/gameweeks/GWxx.md` journals.
4. **Protect project time:** One bounded increment at a time, with no unsolicited new modelling experiments, reruns of settled work, changes to frozen files, rebuilds or tool/framework migrations. Prefer a stopping point over a troubleshooting spiral. Only one active AI-development item; gameweek scoring remains lightweight maintenance.

### Handoff snapshot — GW8, 26 October 2026

**Verified by user locally:** `python scripts/fetch_gameweek_results.py --gameweek 8` captured all ten final fixtures and created `data/live/results/2026_27_gw08_results.csv`. `python scripts/evaluate_market_shadow.py --gameweek 8` ran after commit `4f7143c` and created `data/live/challenger_evaluations/2026_27_gw08_market_shadow_evaluation.csv`. The frozen 40/60 blend scored **3/10 correct (30%), Log Loss 1.2021, Brier 0.7446**. These files exist on the user's Windows machine; do NOT assume committed/pushed to GitHub.

**Root cause of evaluation failure:** GW8 frozen market-shadow CSV contains `ActualHomeGoals`, `ActualAwayGoals`, `ActualResult` as empty pre-match placeholders. Pandas merged the actual results with suffixes and subsequent direct access to `ActualResult` raised `KeyError`. Commit `4f7143c` removes only empty placeholders before merging and fails if they unexpectedly hold populated values. User rerun successfully evaluated GW8. Do not repeat this investigation.

**Outstanding:** Evaluate the **frozen Model 2 GW8 forecast** with the *existing* scoring workflow after inspecting its current implementation and GW7 precedent. Check whether an adaptive challenger has a valid frozen GW8 snapshot before evaluating. Then independently validate matches, scores, probabilities and forecast timestamps, calculate GW6–GW8 cumulative performance, author `docs/gameweeks/GW08.md` and commit only appropriately checked results/documentation. Do NOT guess Model 2's performance, infer adaptive performance, assume model promotion happened or change weights because GW8 was poor. GW7 journal explicitly says formal 40/60 operational promotion was pending governance exception and technical fallback.

**Next single action:** Inspect the repo's existing Model 2 evaluation process and its frozen GW8 inputs, without editing or running a replacement scoring method first. Give the user only the next necessary command after verification.

**Primary longer-term objective:** Resume V2 grounded-answer quality work in issue #11 after the bounded weekly evaluation, retaining original Streamlit default. No extra paid services, no new backlogs or repeated master-plan discussions.

## Incremental V2 form-tool extension — 28 October 2026

**Committed:** `86b1fb0` adds the existing read-only `form_summary(team, games)` tool to V2's typed allow-list (1–10 completed matches); `1d41c13` adds isolated two-team orchestration and invalid-input tests. No model weights, frozen forecasts or default Streamlit routing changed. The view explicitly labels form as current DuckDB history, **not** a frozen pre-match feature snapshot.

**Verification status:** Changes committed remotely; the new suite and local Ollama end-to-end test have **not** yet been executed on the Windows runtime. Prior 23/23 tests passed before this extension. The next single action is `git pull origin main` followed by `python -m unittest discover -s tests -v`. Then run one real two-team comparison in opt-in V2, observing actual tool selection, data coverage and CPU latency. Do not claim live capability until verified.
