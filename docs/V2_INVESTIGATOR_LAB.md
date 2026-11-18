# Football Copilot V2 — bounded investigator lab (research only)

**Status:** isolated implementation + offline real-data verification. **NO-GO for agentic promotion** until live Groq test and an independent audit of answer completeness. Do not merge or deploy this as the production copilot.

## Architecture

- `agent/investigator_v2.py`: extracts a bounded question scope (teams, recent window, head-to-head, opponent strength, venue); asks a planner to select evidence; validates against exact allowed tool calls; executes read-only Python tools; checks coverage; optionally asks for missing evidence once; formats deterministic sourced facts and limitations.
- **Hard limits:** two planning rounds, three proposals per round and six total tool calls, approved tools only. No prediction recomputation or writes.
- **Critical limitation:** Scope and evidence obligations currently depend on a limited deterministic parser. The LLM is responsible for selecting the tool plan **within that scope**, not discovering entirely new tools or analytical operations. A complete evidence plan does not certify full explanation quality.
- No Groq synthesis: the model cannot rewrite verified output into unsupported prose.

## What the offline tests do (not a claim of AI ability)

```bash
python -m unittest discover -s tests -v
python -m scripts.evaluate_v2_investigator
```

The eight-case benchmark uses a **guided oracle control**, not an LLM. It confirms that real repository football data can be retrieved and represented with full planned evidence and audited scope. The independent expected scope prevents missing clubs or parts of a request from silently counting as success.

## Live isolated Groq benchmark (ONE model, opt-in)

Run only in the existing PRIVATE GitHub Codespace, with Groq **Free**, **Zero Data Retention**, and the `GROQ_API_KEY` Codespaces secret already configured. Never paste keys into GitHub files, chat or logs.

```bash
git switch feature/issue9-cloud-readiness-audit
git fetch origin feature/issue9-cloud-readiness-audit
git merge --ff-only FETCH_HEAD
python -m scripts.evaluate_v2_investigator --live-groq
```

This script uses the existing experimental `openai/gpt-oss-20b` model configured in the Codespace, a strict tool-plan response envelope, no retries and a 24-request process cap. It runs at most 18 Groq planner requests (eight first questions, up to one repair each, plus follow-up). Results are reported as source-backed tool-coverage outcomes, **not final answer accuracy**. It doesn't send generated football answers to Groq and doesn't promote an AI answer.

## Optional interactive lab (independent from the existing preview)

```bash
python -m streamlit run app/v2_investigator_lab.py --server.port 8502 --server.address 0.0.0.0
```

Open port **8502** from Codespaces **PORTS**, setting visibility to **PRIVATE**. The original V2 preview stays on 8501 and is not modified. Stop the Codespace when done.

## Go / no-go

Evaluate the live Groq evidence coverage against the 8/8 guided control. Inspect first-pass tool quality, subsequent repair, errors, source coverage, model request count and latency. Do not claim improved agentic intelligence solely from the same model correcting omissions after an oracle-supplied repair hint. Evaluate answer completeness and unsupported causal claims independently before considering promotion.
