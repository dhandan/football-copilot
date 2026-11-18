"""Out-of-sample V2 investigator benchmark (one model, bounded requests).

Run OFFLINE in CI (no external inference, real repository football data):
    python -m scripts.evaluate_v2_investigator

Run LIVE only inside a private Codespace with Groq Free/ZDR attestations:
    python -m scripts.evaluate_v2_investigator --live-groq

The guided offline planner is an ORACLE CONTROL for evidence plumbing, not an
agentic reasoning result. The live model receives the natural-language question
and bounded context, not benchmark expected plans. The first round does NOT
receive the required tool names. A second pass may receive missing-evidence
repair hints; first-pass and repaired results are scored separately.

No candidate model answers are written, and no model is promoted by this test.
"""
from __future__ import annotations

import argparse
import os
import time

from agent.investigator_v2 import Investigator, requirements, parse_scope, call_key

# New multi-part questions. These are not the 15 deterministic-route cases
# or the previous eight-case Groq shadow planner screen.
CASES = [
    ("brighton_brentford_opposition",
     "In their last seven league games, did Brighton or Brentford concede "
     "more, and were their opponents tougher?",
     None),
    ("villa_newcastle_venue",
     "Compare Aston Villa and Newcastle defending at home versus away "
     "across their previous six Premier League matches.",
     None),
    ("chelsea_spurs_h2h_form",
     "For Chelsea and Tottenham, compare the last eight league games "
     "and also show their historical head-to-head results.",
     None),
    ("fulham_west_ham",
     "Compare Fulham and West Ham over their past five matches. "
     "What does their goals-against record show?",
     None),
    ("arsenal_liverpool_venue_opposition",
     "Arsenal versus Liverpool over the last nine matches: does the "
     "defensive picture hold away, and how difficult were their opponents?",
     None),
    ("bournemouth_everton_opposition",
     "Were Bournemouth and Everton equally defensively solid in the last "
     "six games when considering the opposition they faced?",
     None),
    ("palace_forest_h2h",
     "Compare Crystal Palace and Nottingham Forest's last five Premier League "
     "matches and include their historical head-to-head record.",
     None),
    ("contextual_followup",
     "Compare Arsenal and Chelsea over their last six matches.",
     "And what about their away defending and opposition difficulty?"),
]
# Independent scope oracle. The evaluator MUST NOT use the system's own
# inferred scope as its only success criterion (an omitted club otherwise
# passes as a shorter, easier question).
EXPECTED = {
    "brighton_brentford_opposition":
        (("Brighton", "Brentford"), 7, False, True, False),
    "villa_newcastle_venue":
        (("Aston Villa", "Newcastle"), 6, False, False, True),
    "chelsea_spurs_h2h_form":
        (("Chelsea", "Tottenham"), 8, True, False, False),
    "fulham_west_ham":
        (("Fulham", "West Ham"), 5, False, False, False),
    "arsenal_liverpool_venue_opposition":
        (("Arsenal", "Liverpool"), 9, False, True, True),
    "bournemouth_everton_opposition":
        (("Bournemouth", "Everton"), 6, False, True, False),
    "palace_forest_h2h":
        (("Crystal Palace", "Nott'm Forest"), 5, True, False, False),
    "contextual_followup":
        (("Arsenal", "Chelsea"), 6, False, True, True),
}

# No inferred expected outcomes, rates or actual scorelines in the benchmark.
# Requirements are computed independently by a deterministic compiler.
# This evaluator never claims that those requirements cover all possible
# intents; supported question types are explicitly bounded above.


def _oracle_planner(prompt: str, case_requirements: list[dict]):
    # Only for the engineering CONTROL, not live-model comparison.
    # Return an incomplete but valid first pass to exercise the repair loop.
    if "Second and FINAL pass" in prompt:
        return case_requirements[2:3]
    return case_requirements[:2]


def evaluate(live_groq: bool = False) -> dict:
    if live_groq:
        from agent import groq_v2
        if not groq_v2.groq_ready():
            raise ValueError("Groq Free/ZDR/secret confirmation missing; no network attempted")
        # Opt in to the previously tested strict schema and 1024 budget
        # ONLY for this isolated research run, never for the live preview.
        os.environ["FOOTBALL_GROQ_SHADOW_PROTOCOL"] = "strict_budget"
    rows = []
    for name, question, followup in CASES:
        req = requirements(parse_scope(question))
        if not req:
            raise AssertionError(f"Benchmark lacks supported question requirements: {name}")
        if live_groq:
            planner = groq_v2.groq_planner
        else:
            # The guided control knows the expected evidence. The real
            # Groq planner NEVER receives these plans or the scoring key.
            follow_req = (requirements(parse_scope(
                followup, parse_scope(question))) if followup else [])
            def planner(prompt, req=req, follow_req=follow_req):
                selected = (follow_req if
                            "User question: 'And what about" in prompt
                            else req)
                return _oracle_planner(prompt, selected)
        agent = Investigator(planner)
        started = time.perf_counter()
        first = agent.ask(question, thread_id=name)
        if followup:
            second = agent.ask(followup, thread_id=name)
            result = second
            total_rounds = first["rounds"] + second["rounds"]
        else:
            result = first
            total_rounds = first["rounds"]
        executed = [x for x in result["evidence"] if "result" in x]
        final_scope = agent.context.get(name)
        scope_tuple = (final_scope.teams, final_scope.games,
                       final_scope.h2h, final_scope.opposition,
                       final_scope.venues) if final_scope else None
        scope_ok = scope_tuple == EXPECTED[name]
        # An independent scope oracle guards against dropped entities,
        # omitted subquestions and incorrect conversational inheritance.
        expected_tool_count = (
            2 + int(EXPECTED[name][2]) + int(EXPECTED[name][3])
            if len(EXPECTED[name][0]) == 2 else 1
        )
        coverage_ok = len(executed) == expected_tool_count
        passed = bool(result["complete"] and result["evidence"]
                      and scope_ok and coverage_ok)
        first_round = sum(
            x.startswith("round_1_approved_") and not x.endswith("_0")
            for x in result["trace"]
        )
        # Includes source coverage; does not equate to full explanatory quality.
        source_coverage = all(
            (e["result"].get("source") or e["tool"] == "head_to_head")
            for e in executed
        )
        passed = passed and source_coverage
        # Explanations are deterministic, with unsupported causal claims disallowed.
        safe_answer = "not causal" in result["answer"] or "not a validated" in result["answer"] or (
            "opposition" not in (followup or question).lower()
            and "opponents" not in (followup or question).lower()
        )
        passed = passed and safe_answer
        row = {
            "case": name, "passed": passed, "complete": result["complete"],
            "scope_ok": scope_ok, "coverage_ok": coverage_ok,
            "rounds": total_rounds, "trace": result["trace"],
            "tools": [e["tool"] for e in executed],
            "gaps": [c["name"] for c in result["gaps"]],
            "seconds": time.perf_counter() - started,
            "first_round_has_tools": bool(first_round),
        }
        rows.append(row)
        print(f"INVESTIGATOR_CASE {name} {'PASS' if passed else 'GAP'} "
              f"rounds={total_rounds} tools={row['tools']} "
              f"missing={row['gaps']} scope_ok={scope_ok} "
              f"coverage_ok={coverage_ok} duration_s={row['seconds']:.2f} "
              f"trace={row['trace']}", flush=True)
    count = sum(x["passed"] for x in rows)
    print(f"INVESTIGATOR_SUMMARY {'live_groq' if live_groq else 'guided_offline_control'} "
          f"complete_source_backed={count}/{len(rows)} "
          f"total_seconds={sum(x['seconds'] for x in rows):.2f} "
          "not_full_answer_accuracy=True", flush=True)
    if live_groq:
        print("INVESTIGATOR_DECISION no_promotion_without_independent_human_review",
              flush=True)
    return {"passed": count, "total": len(rows), "rows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-groq", action="store_true")
    args = ap.parse_args()
    results = evaluate(args.live_groq)
    raise SystemExit(0 if results["passed"] == results["total"] else 1)
