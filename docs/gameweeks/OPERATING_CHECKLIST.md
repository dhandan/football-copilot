# Football Copilot: mandatory Gameweek release and sprint checklist

**Experiment cadence:** [Weekly rapid optimisation policy](../accelerated-experimentation.md). Diagnose, experiment, validate and decide on the next release **every Gameweek**. No fixed three-Gameweek wait or requirement to change the model every week. Distinguish provisional deployment from confirmed statistical promotion.

**Authority:** [programme issue #1](https://github.com/dhandan/football-copilot/issues/1), [audit issue #8](https://github.com/dhandan/football-copilot/issues/8), [model card](../../MODEL_CARD.md). **Apply every Gameweek, starting with GW7.** This checklist is not permission to alter earlier freezes.

## Roles and sign-off
The assistant coordinates the checklist, checks the repository and records issue evidence proactively. The operator runs local Python and Git commands when needed. **One local command at a time; verify its output before continuing.** Status is only “pre-match frozen” when the official source files, Git commit, Gameweek journal and issue evidence all exist. Status is only “Gameweek complete” after actual results, full evaluation and retrospective.

## A. Before the T-24h target (prepare and review)
- [ ] Reconcile GitHub `main`, existing docs and relevant Kanban issues; recall model decisions and *previously rejected experiments* before making recommendations.
- [ ] Create or pre-populate `docs/gameweeks/GWxx.md` using the previous week's structure; link it in `docs/index.md` and record the week's next sprint/experiment scope.
- [ ] Verify fixture schedule, **first fixture kickoff in UK local time**, and T−24h target in UK local time (including BST/GMT). Note provider delays and bank holidays if relevant.
- [ ] Verify the **actual configured primary release and benchmark roles** for this Gameweek against the written H01 approval; do not assume a documentation decision has changed code. For GW8 the unchanged 40/60 blend is the provisional champion candidate (subject to recorded exception and operational switch), while frozen Model 2 remains the benchmark. No look-ahead data or unrecorded weight changes.
- [ ] Inspect local `git status --short` before staging. Avoid unrelated untracked artefacts, September test odds and duplicate report trees.
- [ ] Check local environment and credentials without displaying tokens. Enable **VPN before Odds API request** (GW6 observed SSL workaround, not proven cause).
- [ ] Perform only non-destructive preflight/test checks as appropriate; official snapshots must never be overwritten.

## B. Official freeze (before first kickoff)
- [ ] Fetch/validate the correct 10 fixtures, dates, times and stable IDs; save source and timestamp.
- [ ] Generate the frozen Model 2 benchmark predictions; require exactly 10 successful, version-checked predictions, valid H/D/A probabilities and documented cold starts. Capture actual generation timestamp. If it remains the configured champion, label it accordingly.
- [ ] Fetch official market once, validate ten fixture matches, complete UK H/D/A bookmaker odds, plausible positive decimal prices, bookmaker counts and normalised no-vig probabilities. Record timestamp, source/panel and explicit delay relative to planned T−24h.
- [ ] Generate the unchanged 40/60 blend from **those exact frozen official Model 2 and market files**; assert 10/10 fixture joins, valid probabilities, versions and 0.40/0.60 weighting. Record which version is actually primary versus benchmark; never relabel without the approved operational role switch. Freeze any newly validated, preregistered version separately before kickoff.
- [ ] If odds fetch fails or coverage is incomplete: do **not** silently use test odds or a later substitute; diagnose without printing API keys; record the failure and observed timestamp. If official snapshot has already been written, do **not** overwrite it. Escalate any exception to the planned capture protocol and assess comparability.
- [ ] Stage **explicit paths only**: official fixture, champion, market and shadow snapshots; inspect `git diff --cached --name-only` and `git diff --cached --check`; commit and push; verify the GitHub commit and four file paths.
- [ ] Finalise `GWxx.md` with pre-match probability table, scorelines, baseline and challenger, cold starts, draw/scoreline diagnostics, capture deviations, operational incidents and links to immutable evidence. Update main docs navigation.
- [ ] Record freeze evidence in issues #2/#3 and Kanban. Do not mark the validation experiment Done prior to the results.

## C. After all ten results are final
- [ ] Retrieve exactly the ten official actual results; check stable ID joins, dates and scores. **Do not evaluate partial or substituted samples.**
- [ ] Run champion and shadow evaluators on the **same ten immutable fixtures**. Verify sample sizes, 1X2 accuracy, correct counts, Log Loss and multiclass Brier, plus cold-start/draw/scoreline diagnostics. Confirm outputs do not overwrite previous evaluations.
- [ ] Analyse fixture-level agreement/disagreement and changes in probability quality; report per-GW and cumulative GW6+ challenger performance distinctly from GW1–GW5 development evidence.
- [ ] Update `GWxx.md` with results, metrics, outliers, limitations, hypothesis evidence and sprint retrospective; commit evaluation files, scripts if separately authorised, and documentation with explicit paths.
- [ ] Conduct sprint review: retain/reject/continue evidence for hypotheses; capture limitations and confidence; update Kanban status and RICE only with measured or sourced inputs; select and prepare next week's work.
- [ ] Make a **weekly** retain/improve/provisionally release/revert decision. Do not retune frozen 40/60 weights using GW6+ or declare confirmatory promotion from a single Gameweek. Predefine future candidate validation/guardrails **before** scoring the new candidate; document any exploratory or exceptional deployment transparently.

## D. Audit and security
- [ ] Avoid logs or tracebacks exposing `apiKey` URLs, environment secrets and credentials. Provider key rotation/update to local `.env` is an operator action; never commit `.env`.
- [ ] Link exact commits, timestamps and filenames; distinguish T−24h target from actual capture time; note historical Football-Data opening odds versus live Odds API T−24h snapshot.
- [ ] Review and explicitly close only issues whose registered acceptance criteria are met. An issue comment or completed freeze **does not** itself close a multi-week hypothesis.

## GW7 operational improvements (16 October 2026)

- **VPN preflight:** Connect the VPN before accessing The Odds API. Check connectivity before the official run.
- **Credential security:** Never display API keys in URLs, logs or tracebacks. Rotate the exposed key and update the local `.env` file.
- **Single-run workflow:** Run `python scripts/run_gameweek_once.py --gameweek N` once per gameweek, following a successful dry run.
- **Timing:** Prefer Friday before the first fixture's T-24h cutoff. Record actual capture timestamps and lead hours for all fixtures.
- **Failure recovery:** After any failed run, check for existing official files before retrying. Never overwrite or backdate a snapshot.
- **H07:** Archive a fresh FPL player snapshot. Keep H07 observational until independently validated.
- **Git governance:** Commit official outputs and the gameweek journal, update relevant issues, and verify actual GitHub Projects board statuses.
- **Source files:** The current one-run workflow produces three official CSV files: Model 2, market and the 40/60 challenger. Preserve any separately archived fixture evidence.
- **Implementation follow-up:** Add credential-safe exception handling to the Python API client before the next gameweek. A checklist entry alone does not implement this safeguard.

## GW6 exception record
GW6 was successfully frozen but ran later than its 9 October 12:30 BST target. Official files are in [commit `4efe874`](https://github.com/dhandan/football-copilot/commit/4efe87484d1f1c1e0d4789f55d58cd26870e3589); [GW06 journal](GW06.md). The missing weekly journal was added after the original freeze, and the lifecycle checklist was introduced retrospectively as a process correction. Keep that deviation visible.

## GW8 fast-release guardrails (23 October 2026)

- [ ] Record and approve the reversible H01 40/60 operational promotion exception before changing model role configuration; specify the fallback if market coverage/integrity fails. No three-Gameweek waiting rule.
- [ ] Identify one existing highest-priority opportunity (H07 first) and run a bounded leakage-safe historical feasibility / validation with success criteria set before scoring.
- [ ] If a candidate passes, publish a new versioned release candidate for **this coming Gameweek** after technical checks; otherwise continue with the last verified deployable baseline.
- [ ] Before kickoff, independently validate actual primary and comparator labels, model versions, source timestamps and all ten immutable forecasts; do not overwrite past snapshots.
- [ ] After final results, assess both on the same fixtures, document weaknesses and make the next weekly decision, with immediate fallback for critical data failures.
