"""Non-gating breadth screen for V2 deterministic conversational coverage.

This intentionally uses NO language-model planner. It is NOT a whole-product
accuracy score, Groq acceptance score, or a claim about football performance.
Each case states an observable user-facing obligation before running.
Failures are retained and counted: do not auto-improve the benchmark.
Run: python -m unittest tests.test_v2_breadth_screen -v
"""
from __future__ import annotations

import unittest

from agent.langgraph_v2 import Conversation


# These are natural-language questions from materially distinct capabilities.
# Deliberately retain unsupported intents to reveal where functionality is absent.
CASES = [
    {
        "name": "two_team_form_liverpool_arsenal",
        "questions": ["Compare Liverpool and Arsenal over their last 10 matches"],
        "tools": {"form_summary", "form_summary"},
        "teams": {"Liverpool", "Arsenal"},
        "contains": ["Last 10 matches", "Points per game"],
    },
    {
        "name": "two_team_form_city_chelsea",
        "questions": ["Compare Man City and Chelsea over their last five matches"],
        "tools": {"form_summary", "form_summary"},
        "teams": {"Man City", "Chelsea"},
        "contains": ["Goals scored", "Last 5 matches"],
    },
    {
        "name": "defensive_comparison_arsenal_chelsea",
        "questions": ["Who defended better over the last 10 matches, Arsenal or Chelsea?"],
        "tools": {"form_summary", "form_summary"},
        "teams": {"Arsenal", "Chelsea"},
        "contains": ["Goals conceded", "defensively stronger"],
    },
    {
        "name": "compound_away_defence",
        "questions": [
            "Compare Liverpool and Arsenal's defensive records over their last "
            "10 matches. Does that advantage hold away?"
        ],
        "tools": {"form_summary", "form_summary"},
        "teams": {"Liverpool", "Arsenal"},
        "contains": ["Away-only", "Away goals conceded per match"],
    },
    {
        "name": "explicit_opponent_strength",
        "questions": [
            "Did Arsenal face stronger opponents than Liverpool during their last 10 matches?"
        ],
        "tools": {"form_summary", "compare_opponent_strength"},
        "teams": {"Liverpool", "Arsenal"},
        "contains": ["Fixture difficulty", "prior points per match"],
    },
    {
        "name": "contextual_opponent_strength_other_clubs",
        "questions": [
            "Compare Chelsea and Tottenham over their last 10 matches",
            "Could the difference be explained by the opponents they faced?",
        ],
        "tools": {"form_summary", "compare_opponent_strength"},
        "teams": {"Chelsea", "Tottenham"},
        "contains": ["Opponent-strength", "pre-match"],
    },
    {
        "name": "five_match_context_followup",
        "questions": [
            "Compare Liverpool and Arsenal over their last 10 matches",
            "What about their last 5 matches?",
        ],
        "tools": {"form_summary"},
        "teams": {"Liverpool", "Arsenal"},
        "contains": ["Last 5 matches"],
    },
    {
        "name": "referential_defensive_why",
        "questions": [
            "Compare Liverpool and Arsenal over their last 10 matches",
            "Who has been defensively better over those 10 matches?",
            "Why is that?",
        ],
        "tools": {"form_summary"},
        "teams": {"Liverpool", "Arsenal"},
        "contains": ["descriptive patterns", "clean sheets"],
    },
    {
        "name": "single_team_recent_goals",
        "questions": ["How many goals have Liverpool scored in their last 5 Premier League matches?"],
        "tools": {"form_summary"},
        "teams": {"Liverpool"},
        "contains": ["Goals scored", "Liverpool"],
    },
    {
        "name": "single_team_defensive_trend",
        "questions": [
            "Has Arsenal conceded fewer goals in its last five games than in the five before?"
        ],
        "tools": {"form_summary"},
        "teams": {"Arsenal"},
        "contains": ["previous five", "Arsenal"],
    },
    {
        "name": "two_team_h2h_results",
        "questions": [
            "What were the last five head-to-head Premier League results between Liverpool and Arsenal?"
        ],
        "tools": {"head_to_head"},
        "teams": {"Liverpool", "Arsenal"},
        "contains": ["Liverpool", "Arsenal", "head-to-head"],
    },
    {
        "name": "other_clubs_brighton_brentford",
        "questions": ["Compare Brighton and Brentford over their last 5 league matches"],
        "tools": {"form_summary"},
        "teams": {"Brighton", "Brentford"},
        "contains": ["Goals conceded", "Brighton", "Brentford"],
    },
    {
        "name": "other_clubs_villa_newcastle",
        "questions": ["Compare Aston Villa with Newcastle over the last 5 Premier League matches"],
        "tools": {"form_summary"},
        "teams": {"Aston Villa", "Newcastle"},
        "contains": ["Points per game", "Aston Villa", "Newcastle"],
    },
    {
        "name": "contextual_shorthand_attack",
        "questions": [
            "Compare Liverpool and Arsenal over their last 10 games",
            "Which of them scored more?",
        ],
        "tools": {"form_summary"},
        "teams": {"Liverpool", "Arsenal"},
        "contains": ["Goals scored"],
    },
    {
        "name": "home_and_away_requested",
        "questions": [
            "Compare Chelsea and Tottenham's home and away defending over their last 10 matches"
        ],
        "tools": {"form_summary"},
        "teams": {"Chelsea", "Tottenham"},
        "contains": ["Home goals conceded per match", "Away goals conceded per match"],
    },
]


def screen():
    """Return auditable results, with no hidden calls, mocks or LLM substitutions."""
    rows = []
    for case in CASES:
        conversation = Conversation(planner=lambda _: [])
        try:
            result = None
            for question in case["questions"]:
                result = conversation.ask(question, thread_id=case["name"])
            if result is None:
                raise AssertionError("No answer")
            evidence = result.get("evidence", [])
            actual = {item["tool"] for item in evidence}
            team_args = set()
            for item in evidence:
                args = item.get("arguments", {})
                for key in ("team", "team1", "team2"):
                    if args.get(key):
                        team_args.add(args[key])
            answer = result.get("answer", "")
            checks = {
                "required_tools": case["tools"].issubset(actual),
                "team_recall": case["teams"].issubset(team_args),
                "answer_obligations": all(x.casefold() in answer.casefold()
                                           for x in case["contains"]),
                "no_tool_failure": bool(evidence) and all(
                    "error" not in item.get("result", {}) for item in evidence
                ),
            }
            rows.append({"case": case["name"], "pass": all(checks.values()),
                         "checks": checks, "actual_tools": sorted(actual),
                         "trace": result.get("trace", [])})
        except Exception as exc:
            rows.append({"case": case["name"], "pass": False,
                         "checks": {"runtime": False}, "error": type(exc).__name__})
    return rows


class V2BreadthScreen(unittest.TestCase):
    def test_non_gating_deterministic_coverage_screen(self):
        results = screen()
        self.assertEqual(len(results), len(CASES))
        successes = sum(r["pass"] for r in results)
        print(f"\nV2_BREADTH_SCREEN deterministic_only={successes}/{len(results)}"
              " (NOT Groq or whole-product accuracy)", flush=True)
        for result in results:
            print("V2_BREADTH_CASE", "PASS" if result["pass"] else "GAP",
                  result["case"], "checks=", result["checks"],
                  "tools=", result.get("actual_tools"),
                  "trace=", result.get("trace"), flush=True)
        # Do not turn current limitations into passing tests by changing the
        # expectations. This diagnostic is intentionally non-gating.
