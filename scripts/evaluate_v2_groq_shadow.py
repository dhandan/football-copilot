"""Bounded real Groq shadow evaluation: NEVER emits candidate text or secrets.

No model-written answer is promoted to Football Copilot. Deterministic
inference is the benchmark; the LLM is probed independently to detect
whether its planning and response writing add any reliable capability.

Run ONLY in private controlled runtime with explicit Free+ZDR attestations:
    python -m scripts.evaluate_v2_groq_shadow

Uses 8 planner + 3 synthesis requests per model, 22 in one process.
The Groq adapter's 24-request process cap, per-call timeout, read-only tools,
and fail-closed behaviour remain in force. No paid fallback and no retries.
"""
from __future__ import annotations

import os
import re
import time

from agent import groq_v2
from agent.langgraph_v2 import ask, _grounded


# Benchmark is intentionally outside the deterministic router. A real model
# must propose a complete tool plan, including where routing would otherwise
# bypass Groq. This is NOT a measurement of live app answer quality.
P = lambda tool, **args: {"name": tool, "arguments": args}

PLANNER_CASES = [
    ("h07", "Why was H07 rejected?",
     [P("experiment_history", experiment="H07")]),
    ("gw8_plus_performance",
     "Show frozen GW8 predictions and compare them with the champion's recent performance",
     [P("frozen_gameweek_predictions", gameweek=8), P("model_performance")]),
    ("other_clubs",
     "Compare Brighton and Brentford over their last 5 Premier League matches",
     [P("form_summary", team="Brighton", games=5),
      P("form_summary", team="Brentford", games=5)]),
    ("head_to_head",
     "What were the last five head-to-head Premier League results for Liverpool and Arsenal?",
     [P("head_to_head", team1="Liverpool", team2="Arsenal")]),
    ("opponent_confounded",
     "Did Arsenal concede fewer goals than Liverpool over their last ten matches "
     "because they played weaker opponents?",
     [P("form_summary", team="Arsenal", games=10),
      P("form_summary", team="Liverpool", games=10),
      P("compare_opponent_strength", team1="Arsenal", team2="Liverpool", games=10)]),
    ("out_of_scope", "What's the current temperature in London?", []),
    ("champion_quality", "How has the provisional 40/60 champion performed so far?",
     [P("model_performance")]),
    ("three_way",
     "Compare GW8 frozen forecasts, champion performance and H07 findings",
     [P("frozen_gameweek_predictions", gameweek=8),
      P("model_performance"), P("experiment_history", experiment="H07")]),
]

SYNTHESIS_CASES = [
    ("h07", "Why was H07 rejected?",
     [P("experiment_history", experiment="H07")]),
    ("performance", "How has the 40/60 champion performed so far?",
     [P("model_performance")]),
    ("form", "Compare Liverpool and Arsenal over their last 10 matches",
     [P("form_summary", team="Liverpool", games=10),
      P("form_summary", team="Arsenal", games=10)]),
]
MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b")


def plan_key(plan):
    """Arguments, not mere tool names, are contract requirements."""
    return {
        (item["name"], tuple(sorted(item["arguments"].items())))
        for item in plan
    }


def grounded_and_complete(candidate: str, facts: str, case: str, question: str):
    is_performance = case == "performance"
    grounded = _grounded(candidate, facts, False, is_performance, question)
    # The existing grounder is necessary but insufficient: reject answers
    # that omit the main user intent, relevant comparison or uncertainty.
    lower = candidate.lower()
    if case == "h07":
        complete = (
            "h07" in lower and "model 2" in lower
            and any(word in lower for word in ("parked", "rejected", "did not outperform"))
            and any(word in lower for word in ("log loss", "brier", "accuracy"))
        )
    elif case == "performance":
        complete = (
            "40/60" in lower and "model 2" in lower and "gw8" in lower
            and "55%" in candidate and "40%" in candidate
            and any(word in lower for word in ("small sample", "one gameweek", "10 matches", "ten matches"))
            and "provisional" in lower
        )
    elif case == "form":
        complete = (
            "arsenal" in lower and "liverpool" in lower
            and any(word in lower for word in ("points", "goals", "form"))
            and not any(word in lower for word in ("caused by", "because of a new formation"))
        )
    else:
        raise ValueError("Unknown test")
    return grounded, complete


def run():
    if not groq_v2.groq_ready():
        print("GROQ_SHADOW NOT_EXECUTED missing private key and/or user-attested Free/ZDR controls")
        return 2
    # No fake tool values; these are independently calculated from the repo.
    facts_for = {}
    for label, question, plan in SYNTHESIS_CASES:
        result = ask(question, planner=lambda _: plan, synthesizer=None)
        if not result["evidence"] or any("error" in e["result"] for e in result["evidence"]):
            print(f"GROQ_SHADOW BLOCKED missing deterministic evidence for {label}")
            return 3
        facts_for[label] = result["answer"]
    records = []
    for model in MODELS:
        os.environ["FOOTBALL_GROQ_MODEL"] = model
        for case, question, expected in PLANNER_CASES:
            start = time.perf_counter()
            try:
                actual = groq_v2.groq_planner(question)
                success = plan_key(actual) == plan_key(expected) and len(actual) == len(expected)
                status = "PASS" if success else "WRONG_PLAN"
            except (ValueError, TypeError) as exc:
                success = False
                status = type(exc).__name__
            elapsed = time.perf_counter() - start
            records.append((model, "planner", case, success, status, elapsed))
            print(f"GROQ_SHADOW {model} planner {case} {status} latency_s={elapsed:.2f}", flush=True)
        for case, question, _ in SYNTHESIS_CASES:
            start = time.perf_counter()
            try:
                candidate = groq_v2.groq_synthesizer(question, facts_for[case])
                grounded, complete = grounded_and_complete(candidate, facts_for[case], case, question)
                success = grounded and complete
                status = ("PASS" if success else
                          "INCOMPLETE_GROUNDED" if grounded else
                          "REJECTED_GROUNDING")
            except (ValueError, TypeError) as exc:
                success = False
                status = type(exc).__name__
            elapsed = time.perf_counter() - start
            records.append((model, "synthesis", case, success, status, elapsed))
            print(f"GROQ_SHADOW {model} synthesis {case} {status} latency_s={elapsed:.2f}", flush=True)

    for model in MODELS:
        rows = [r for r in records if r[0] == model]
        planner = [r for r in rows if r[1] == "planner"]
        synthesis = [r for r in rows if r[1] == "synthesis"]
        n_plan = sum(r[3] for r in planner)
        n_synth = sum(r[3] for r in synthesis)
        mean_time = sum(r[5] for r in rows) / len(rows)
        print(f"GROQ_SHADOW_SUMMARY model={model} planner_exact={n_plan}/{len(planner)} "
              f"synthesis_grounded_complete={n_synth}/{len(synthesis)} "
              f"mean_request_latency_s={mean_time:.2f}", flush=True)
    # Experimental failures are evidence, not CI infra failures: report all
    # results without hiding them or automatically rewriting the benchmark.
    print(f"GROQ_SHADOW_RESULT calls={groq_v2._calls} "
          "user_facing_answers_promoted=0 decision=NO_GO_PENDING_FULL_VALIDATION",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
