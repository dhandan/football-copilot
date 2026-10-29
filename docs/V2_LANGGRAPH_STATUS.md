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

### Local verification — 28 October 2026

**User-verified on Windows after commit `a6375a6`:** `python -m unittest discover -s tests -v` passed **25/25**. A live local-Qwen query, “Compare Liverpool and Arsenal recent form over their last 5 matches”, selected and executed exactly two `form_summary` tools. DuckDB response: Liverpool W1 D2 L2, 5 points, GF9 GA10; Arsenal W5 D0 L0, 15 points, GF8 GA1. Trace: `planned 2 tool(s)`, two `executed form_summary`. Single successful live run, not a measured reliability rate. No frozen forecasts changed. Minor presentation backlog: integer-like totals rendered with `.0`; no concise comparative conclusion. **Next bounded choice:** stop after successful proof or separately profile repeatability before considering further synthesis adjustments. 

### Streamlit live test incident — 28 October 2026
User live V2 form comparison Liverpool/Arsenal succeeded but synthesis overclaimed; Manchester City/Chelsea and anaphoric 10-match follow-up returned masked `ValueError`. The completed GW8 page showed metrics but failed before fixture table. Code inspection confirmed `matches_evaluated` undefined in `app/live_predictions.py` and corrected to `matches` (commit `a05750a`); V2 now displays `ValueError` details rather than hiding them (commit `eff3387`), with regression test (commit `d0b6038`). **Not locally verified yet.** Do not claim the latter two query failures are fixed. Need user traceback/error details from one focused retest, then targeted root-cause correction. Do not change frozen forecasts.

### Verified Streamlit recovery — 28 October 2026

**User-confirmed live:** On commit `4857186`, all 28 unittest tests passed. The experimental Streamlit V2 correctly answered 'Compare Manchester City and Chelsea's form over their last 5 matches' for both teams (Man City W2 D2 L1, PPG 1.6; Chelsea W1 D1 L3, PPG 0.8), then resolved 'How does that compare with their last 10 matches?' to both teams (Man City W5 D4 L1, PPG 1.9; Chelsea W2 D1 L7, PPG 0.7). Screenshot supplied. This is one successful two-turn scenario, not a repeated-run reliability claim. Remaining UX polish: whole-number statistics currently show `.0`, and deterministic form summaries lack concise head-to-head comparison prose. The previously reported GW8 page-render fix was committed but its visual verification is not confirmed by this screenshot. Avoid further changes until a bounded next increment is selected.

### GW8 monitoring display correction — 28 October 2026
User screenshot showed outdated GW5 heading, literal f-string code in the 1-1 bullet and obsolete five-gameweek governance wording. Code inspection confirmed static triple-quoted Markdown and undefined names embedded literally. Commit `0ea8dde` changes only `app/live_predictions.py`: dynamic gameweek label, actual gameweek-specific modal 1-1 counts from existing evaluation, ongoing diagnostic caveats and current provisional blend governance. **Unverified locally** pending pull, tests and Streamlit refresh. No model or frozen forecast changes.

### GW8 model comparison dashboard — 28 October 2026
Committed `b4382d4`: completed GW8 Streamlit view now leads with the documented four-method GW8 scorecard and a comparable 30-match GW6–GW8 40/60-versus-Model-2 scorecard. Existing detailed Model 2 evaluation is explicitly labelled a benchmark. Metrics and provisional/non-promoted governance are taken from `docs/gameweeks/GW08.md`, not recalculated or written to frozen data. UI change is GW8/2026-27-scoped; no broad generic tracker implemented. **Outstanding:** user to pull and visually verify; not yet locally run. No model, freeze or evaluation changes.

### V2 grounded comparisons increment — 29 October 2026
Committed a small deterministic two-team recent-form comparison to `agent/langgraph_v2.py`. For exactly two valid same-window form summaries the app now calculates differences in points per game, points, goals scored and goals conceded from read-only DuckDB-derived results. It precedes underlying team facts and the existing form `form_facts_preserved` gate prevents unsupported LLM winner claims. No prediction data, model weights, Streamlit routing or database altered. Two unit tests added covering correct differences and missing-data refusal. **Not yet executed on the user's local Windows environment.** Next: `git pull origin main` and `python -m unittest discover -s tests -q`, followed by the existing Man City/Chelsea Streamlit two-turn scenario. Single live run does not establish repeatability; evaluate further only if this passes.

### V2 live comparison validation — 29 October 2026
User confirmed screenshot of one real Streamlit two-turn comparison (Man City vs Chelsea, last five then last ten matches): correct two-team context, PPG/points/goals differences and source-grounded summaries. Previously 30/30 local tests passed before presentation polish. Additional small follow-up commit formats whole-number match counts as integers and describes lower goals conceded as fewer goals; corresponding regression assertion updated. **Latest polish not yet locally tested or visually confirmed.** No prediction changes. Further reliability testing of diverse questions remains open.

### V2 latency and season correctness incident — 29 October 2026
User reported approximately 20-minute response time for two-team form tests and reported answers based on last season, with poorly separated team facts. Code inspection verified `analytics.football_analytics.get_form_summary` lacked season filtering and the V2 form route invoked local Ollama planner even for unambiguous comparisons. Changes: optional season filter added to summary SQL; V2 default form queries now explicitly request 2026/27, without historical fallbacks; the two-team form fast route bypasses Ollama planning and retains bounded tools; response headings label season/team and missing data returns an honest no-current-data message. Added two relevant regression tests. **Not yet locally tested.** The database's 2026/27 `team_match_stats` coverage is unknown; this may correctly show unavailable current-season form, rather than pretending historical rows are current. Historical season selection as a user-controlled option and detailed timing instrumentation remain future improvements. No prediction files touched.

### Live V2 fast route / missing season test — 29 October 2026
User discovered earlier 10–20-minute sessions had the original Qwen4B agent selected, not V2. With LangGraph V2 actually selected, Liverpool versus Arsenal last-five form returned in under one minute. Both teams showed no available 2026/27 DuckDB matches; a bug also generated a `nan` comparison from the zero-match aggregate row. Commit `cdba33e` rejects zero/nonfinite summary rows for comparisons and `f73a49b` adds regression coverage. **Latest fix not yet locally tested.** Next priority: verify current-season 2026/27 data ingestion into DuckDB (currently no matches for these two teams), then validate live form without synthetic values. Do not claim subsecond response or successful current-season analysis yet.

### Current-season snapshot-backed form — 29 October 2026
Local user query verified DuckDB contains only 2021/22–2025/26 (760 team-match rows each); no 2026/27. For V2 2026/27 form, commit `5660a80` now reads existing completed full-gameweek result CSVs in data/live/results, computes team-specific results/read-only, limits to requested matches and identifies season and source. No DuckDB rebuild, results overwrite, model or frozen forecast changes. Unit test added in `cc0df39` to verify Liverpool's confirmed GW8 result is included. **Not yet locally tested.** Coverage may be below requested last-five/last-ten if older results snapshots are missing; UI must show actual played sample rather than implying full coverage. Future option: unified current-season ingest/maintainable source and provenance, after successful verification.

### Live current-season form — 29 October 2026
User's V2 Streamlit Liverpool/Arsenal last-five response returned Liverpool W3 D1 L1, 10 points, GF5 GA4; Arsenal W4 D0 L1, 12 points, GF7 GA4. Difference arithmetic matched. Response was falsely labelled DuckDB despite current 2026/27 code reading result snapshots. Commit `06a00f7` renders the tool-provided source path and W/D/L concise counts; test expectation updated `2d10a07`. **Pending local pull and test; provenance of individual contributing snapshots was not independently audited.** No forecasts touched.

### Cross-season latest-N interpretation — 29 October 2026
User clarified that 'last 10 matches' means the last ten completed league matches across season boundaries. Implemented read-only fallback from verified 2026/27 result snapshots to 2025/26 DuckDB `get_team_form` when the current-season snapshot window is shorter than requested (`88d5235`). V2 answer headings use source-reported season coverage, potentially 2026/27 and 2025/26 (`036ba51`), and acknowledge partial coverage. Added specific ten-match bridge regression test (`cf4e27a`). **Not yet locally tested**. This changes historical lookback only, no frozen predictions or database writes. Longer-term extend season selection to support all team names/promoted-side histories, and align explicit same-season requests with separate scope semantics.

### V2 comparison presentation improvement — 29 October 2026
User requested a much clearer information hierarchy rather than repetitive paragraph-form output. Latest change replaces paired full prose summaries in V2 responses with an insight-first summary, clearly stated season coverage, compact two-column per-team Markdown table (matches, W/D/L, points, PPG, GF/GA) and one source line. Deterministic calculations and raw evidence unchanged; no added model calls. Regression assertions added for rendered table. **Pending local tests and Streamlit visual confirmation.**
