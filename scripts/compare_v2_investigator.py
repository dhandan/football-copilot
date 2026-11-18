"""Deterministic compiler versus Groq planner: isolated V2 ablation.

By construction, the investigator's tool allowlist is compiled by parse_scope /
requirements before Groq is asked to plan. Therefore this comparison CANNOT
demonstrate novel autonomous tool discovery; it asks whether the extra planner
adds reliability/value over simply executing the same vetted requirements.

Offline default only runs the deterministic baseline. Live runs both arms on
the identical cases with independent expected scopes and evidence obligations.
No inference or network access without --live-groq and provider opt-in.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from agent.investigator_v2 import Investigator, parse_scope, requirements, call_key
from scripts.audit_v2_investigator_answers import audit_answer


@dataclass(frozen=True)
class Case:
    name: str
    question: str
    teams: tuple[str, ...]
    games: int
    focus: str = "defence"
    venue: bool = False
    opponent: bool = False
    h2h: bool = False
    followup: str | None = None
    follow_focus: str | None = None
    follow_venue: bool = False
    follow_opponent: bool = False
    follow_h2h: bool = False


# Not drawn from the earlier 8-case benchmark; source data stays unchanged.
CASES = [
    Case("brighton_chelsea_attack",
         "Who scored more goals in the last six league matches, Brighton or Chelsea?",
         ("Brighton", "Chelsea"), 6, focus="attack"),
    Case("brentford_arsenal_defence",
         "Across their last seven Premier League games, compare Brentford and Arsenal's defending.",
         ("Brentford", "Arsenal"), 7),
    Case("west_ham_liverpool_attack",
         "Compare West Ham and Liverpool's goals scored and points across the last five matches.",
         ("West Ham", "Liverpool"), 5, focus="attack"),
    Case("tottenham_newcastle_away_attack",
         "In their previous eight matches, did Tottenham or Newcastle score at a higher away rate?",
         ("Tottenham", "Newcastle"), 8, focus="attack", venue=True),
    Case("chelsea_arsenal_opposition",
         "Which defence conceded less over the last nine matches, Chelsea or Arsenal, and who faced stronger opposition?",
         ("Chelsea", "Arsenal"), 9, opponent=True),
    Case("villa_west_ham_h2h",
         "Compare Aston Villa and West Ham's last six league games alongside historical head-to-head.",
         ("Aston Villa", "West Ham"), 6, h2h=True),
    Case("everton_bournemouth_defence",
         "Were Everton or Bournemouth more defensively solid in their last five games?",
         ("Everton", "Bournemouth"), 5),
    Case("forest_palace_attack",
         "Over their past seven matches, who has scored more, Nottingham Forest or Crystal Palace?",
         ("Nott'm Forest", "Crystal Palace"), 7, focus="attack"),
    Case("contextual_attack_followup",
         "In their last seven matches, compare Brentford and Brighton's goals scored.",
         ("Brentford", "Brighton"), 7, focus="attack",
         followup="And has that attacking lead held away from home?",
         follow_focus="attack", follow_venue=True),
    Case("explicit_subject_switch",
         "Compare Brighton and Brentford's last six Premier League matches.",
         ("Brighton", "Brentford"), 6,
         followup="Forget Brighton and Brentford. Compare Chelsea and Tottenham's last eight matches and their historical head-to-head results.",
         follow_focus="defence", follow_h2h=True),
]
SWITCH_EXPECTED = (("Chelsea", "Tottenham"), 8)


def expected_scope(case: Case, followup: bool = False):
    if followup and case.name == "explicit_subject_switch":
        return (SWITCH_EXPECTED[0], SWITCH_EXPECTED[1],
                case.follow_focus, False, False, case.follow_h2h)
    if followup:
        return (case.teams, case.games, case.follow_focus or case.focus,
                case.follow_venue, case.follow_opponent, case.follow_h2h)
    return (case.teams, case.games, case.focus,
            case.venue, case.opponent, case.h2h)


def expected_calls(expected):
    teams, games, _, _, opponent, h2h = expected
    calls = [("form_summary", (("games", games), ("team", team)))
             for team in teams]
    if opponent:
        calls.append(("compare_opponent_strength",
                      (("games", games), ("team1", teams[0]), ("team2", teams[1]))))
    if h2h:
        calls.append(("head_to_head",
                      (("team1", teams[0]), ("team2", teams[1]))))
    return set(calls)


class CompilerPlanner:
    """Compile exact authorised calls; no LLM, no oracle scoring lookup."""
    def __init__(self):
        self.calls = []
        self.requests = 0

    def set_question(self, question, previous=None):
        self.calls = requirements(parse_scope(question, previous))
        self.requests = 0

    def __call__(self, prompt):
        self.requests += 1
        if "Second and FINAL pass" in prompt:
            return self.calls[3:6]
        return self.calls[:3]


class CountedPlanner:
    def __init__(self, planner):
        self.planner = planner
        self.requests = 0

    def __call__(self, prompt):
        self.requests += 1
        return self.planner(prompt)


def score(result, scope, expected):
    actual_scope = (scope.teams, scope.games, scope.focus,
                    scope.venues, scope.opposition, scope.h2h)
    scope_ok = actual_scope == expected
    evidence_calls = {(e["tool"], tuple(sorted(e["arguments"].items())))
                      for e in result["evidence"] if "result" in e}
    expected_evidence = expected_calls(expected)
    evidence_ok = evidence_calls == expected_evidence
    problems = []
    if scope_ok and evidence_ok and result["complete"]:
        try:
            problems = audit_answer(result["answer"], scope, result["evidence"])
        except (KeyError, TypeError, ValueError, IndexError):
            problems = ["audit_incomplete"]
    else:
        problems = ["incomplete_evidence_or_scope"]
    passed = (scope_ok and evidence_ok and result["complete"]
              and not problems and not result.get("provider_blocked", False))
    return {"passed": passed, "scope_ok": scope_ok, "coverage_ok": evidence_ok,
            "answer_audit_ok": not problems, "issues": problems,
            "provider_blocked": bool(result.get("provider_blocked"))}


def evaluate(live_groq=False):
    if live_groq:
        from agent import groq_v2
        if not groq_v2.groq_ready():
            raise ValueError("Groq Free/ZDR/key not enabled; no live calls made.")
        # Matches the isolated research protocol, without modifying preview.
        import os
        os.environ["FOOTBALL_GROQ_SHADOW_PROTOCOL"] = "strict_budget"
    baseline_rows, groq_rows = [], []
    for case in CASES:
        for arm in ("compiler", "groq") if live_groq else ("compiler",):
            if arm == "compiler":
                planner = CompilerPlanner()
            else:
                from agent.groq_v2 import groq_planner
                planner = CountedPlanner(groq_planner)
            investigator = Investigator(planner)
            prior = None
            total_requests = 0
            start = time.perf_counter()
            for index, question in enumerate(
                    (case.question, case.followup) if case.followup else (case.question,)):
                if arm == "compiler":
                    planner.set_question(question, prior)
                before = planner.requests
                result = investigator.ask(question, thread_id=case.name)
                total_requests += planner.requests - before
                prior = investigator.context.get(case.name)
                expected = expected_scope(case, followup=index == 1)
                scored = score(result, prior, expected) if prior else {
                    "passed": False, "scope_ok": False, "coverage_ok": False,
                    "answer_audit_ok": False, "issues": ["scope_missing"],
                    "provider_blocked": False,
                }
                # A whole conversation must pass every step.
                if index == 0:
                    all_passed = scored["passed"]
                else:
                    all_passed = all_passed and scored["passed"]
            record = {"case": case.name, "arm": arm, "passed": all_passed,
                      "final": scored, "requests": total_requests,
                      "seconds": round(time.perf_counter() - start, 3)}
            (baseline_rows if arm == "compiler" else groq_rows).append(record)
            outcome = ("PROVIDER_LIMIT" if scored["provider_blocked"] else
                       "PASS" if all_passed else "FAIL")
            print(f"COMPARISON_CASE {case.name} {arm} {outcome} "
                  f"requests={total_requests} seconds={record['seconds']} "
                  f"scope_ok={scored['scope_ok']} coverage_ok={scored['coverage_ok']} "
                  f"answer_ok={scored['answer_audit_ok']} "
                  f"issues={scored['issues']}", flush=True)
    def report(arm, rows):
        p = sum(int(row["passed"]) for row in rows)
        requests = sum(row["requests"] for row in rows)
        duration = sum(row["seconds"] for row in rows)
        print(f"COMPARISON_SUMMARY {arm} complete={p}/{len(rows)} "
              f"planner_requests={requests} total_s={duration:.3f} "
              "scope=limited_compiler_tool_contract", flush=True)
        return p
    base = report("compiler_control", baseline_rows)
    groq = report("live_groq", groq_rows) if live_groq else None
    print("COMPARISON_DECISION no_production_promotion; "
          "compiler_control_is_not_existing_product; "
          "no_novel_tool_discovery_tested", flush=True)
    return {"baseline": base, "groq": groq, "total": len(CASES),
            "baseline_rows": baseline_rows, "groq_rows": groq_rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-groq", action="store_true")
    args = parser.parse_args()
    results = evaluate(args.live_groq)
    # Offline control regression is gating; exploratory live reliability is a report.
    raise SystemExit(0 if args.live_groq or results["baseline"] == results["total"] else 1)
