"""Isolated conditional-context ablation.  Offline by default.

The benchmark tests *incremental verified evidence*, NOT football explanations
or causal accuracy.  The deterministic baseline renders only form evidence.
The conditional arm offers opponent context after observing a verified gap.
The guided planner is a control, not evidence of Groq planning ability.

--live-groq is manual/opt-in only; no CI secrets, schedule or background calls.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace

from agent.investigator_v2 import (
    Investigator, _execute, conditional_opposition_evidence,
    parse_scope, render_evidence, requirements,
)
from scripts.audit_v2_investigator_answers import audit_answer


@dataclass(frozen=True)
class Case:
    name: str
    question: str
    teams: tuple[str, str]
    games: int
    focus: str
    expect_context: bool


CASES = (
    Case("arsenal_liverpool_defence",
         "Why did Arsenal concede fewer goals than Liverpool over their last seven matches?",
         ("Arsenal", "Liverpool"), 7, "defence", True),
    Case("brighton_brentford_attack",
         "Why did Brighton score more than Brentford over their last seven matches?",
         ("Brighton", "Brentford"), 7, "attack", True),
    Case("chelsea_tottenham_defence",
         "Why did Chelsea and Tottenham have different defensive records over their last eight games?",
         ("Chelsea", "Tottenham"), 8, "defence", True),
)
# Expected context eligibility is independently cross-checked with the actual
# source values; fixed expectations must not be adjusted to force a pass.


class GuidedPlanner:
    def __init__(self, first: list[dict], conditional: dict | None):
        self.first, self.conditional = first, conditional
        self.requests = 0

    def __call__(self, prompt: str) -> list[dict]:
        self.requests += 1
        if "Second and FINAL pass" in prompt:
            return [self.conditional] if self.conditional else []
        return self.first


class CountedPlanner:
    def __init__(self, planner):
        self.planner, self.requests = planner, 0

    def __call__(self, prompt):
        self.requests += 1
        return self.planner(prompt)


def form_only(question: str):
    """Control: run exactly the form tools, no planner or extra context."""
    scope = parse_scope(question)
    calls = requirements(scope)
    if (len(calls) != 2 or any(c["name"] != "form_summary" for c in calls)):
        raise ValueError("Not a two-team first-pass form question")
    evidence = [_execute(call, Investigator(lambda _: []).tools)
                for call in calls]
    complete = all("result" in item for item in evidence)
    return scope, evidence, render_evidence(scope, evidence, [] if complete else calls)


def score_context(case: Case, scope, baseline, result, expected):
    baseline_evidence = baseline
    baseline_only = {e["tool"] for e in baseline_evidence if "result" in e}
    evidence = result["evidence"]
    opponent_items = [e for e in evidence
                      if e.get("tool") == "compare_opponent_strength"
                      and "result" in e]
    extra = bool(opponent_items)
    actual_expected = expected is not None
    grounded = (result["complete"] and
                not result.get("provider_blocked") and
                all("result" in e for e in evidence))
    if grounded:
        render_scope = replace(scope, opposition=extra)
        problems = audit_answer(result["answer"], render_scope, evidence)
        grounded = not problems
    else:
        problems = ["partial_or_missing_evidence"]
    no_causal_claim = ("cannot establish" in result["answer"]
                       and "prove causation" in result["answer"])
    first_pass_complete = {e["arguments"]["team"] for e in evidence
                           if e["tool"] == "form_summary" and "result" in e}
    return {
        "pass": (baseline_only == {"form_summary"}
                 and first_pass_complete == set(case.teams)
                 and actual_expected == case.expect_context
                 and extra == case.expect_context
                 and grounded and no_causal_claim),
        "extra_context": extra,
        "baseline_tools": len(baseline_evidence),
        "extra_tools": len(evidence) - len(baseline_evidence),
        "audit_issues": problems,
        "causal_guardrail": no_causal_claim,
    }


def run(live_groq=False):
    if live_groq:
        from agent import groq_v2
        if not groq_v2.groq_ready():
            raise ValueError("Groq opt-in/Free/ZDR/secret preflight failed; no remote calls.")
        import os
        os.environ["FOOTBALL_GROQ_SHADOW_PROTOCOL"] = "strict_budget"
    results = []
    for case in CASES:
        start = time.perf_counter()
        scope, baseline, base_answer = form_only(case.question)
        if (scope.teams, scope.games, scope.focus, scope.investigate_why) != (
                case.teams, case.games, case.focus, True):
            raise ValueError("Independent benchmark scope differs: " + case.name)
        expected = conditional_opposition_evidence(scope, baseline)
        if (expected is not None) != case.expect_context:
            raise ValueError("Source-reconciled expectation mismatch: " + case.name)
        initial_calls = requirements(scope)
        if live_groq:
            from agent.groq_v2 import groq_planner
            planner = CountedPlanner(groq_planner)
        else:
            planner = GuidedPlanner(initial_calls, expected)
        investigator = Investigator(planner)
        result = investigator.ask(case.question, thread_id=case.name)
        evaluation = score_context(case, scope, baseline, result, expected)
        results.append(evaluation)
        print(f"CONDITIONAL_VALUE {case.name} "
              f"{'PASS' if evaluation['pass'] else 'FAIL'} "
              f"extra_context={evaluation['extra_context']} "
              f"extra_tools={evaluation['extra_tools']} "
              f"groq_requests={planner.requests if live_groq else 0} "
              f"guided_planner={not live_groq} "
              f"seconds={time.perf_counter()-start:.3f} "
              f"issues={evaluation['audit_issues']}", flush=True)
    passed = sum(int(e["pass"]) for e in results)
    print(f"CONDITIONAL_VALUE_SUMMARY {passed}/{len(results)} "
          f"live_groq={live_groq} "
          "claim=incremental_opponent_context_not_causal_explanation "
          "no_production_promotion=True", flush=True)
    return passed, len(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-groq", action="store_true",
                        help="Manual opt-in only; never used in CI.")
    args = parser.parse_args()
    passed, total = run(args.live_groq)
    raise SystemExit(0 if passed == total else 1)
