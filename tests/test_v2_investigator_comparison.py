"""Contract tests for the isolated compiler-vs-Groq ablation (offline)."""
import unittest

from agent.investigator_v2 import parse_scope
from scripts.compare_v2_investigator import (
    CASES, CompilerPlanner, CountedPlanner, expected_calls,
    expected_scope, score,
)


class ComparisonContractTests(unittest.TestCase):
    def test_unique_cases_and_multiple_focus_types(self):
        self.assertEqual(len(CASES), 10)
        self.assertEqual(len({c.name for c in CASES}), len(CASES))
        self.assertTrue(any(c.focus == "attack" for c in CASES))
        self.assertTrue(any(c.opponent for c in CASES))
        self.assertTrue(any(c.h2h for c in CASES))
        self.assertTrue(any(c.followup for c in CASES))

    def test_expected_cases_have_independent_scopes(self):
        for case in CASES:
            with self.subTest(case=case.name):
                first = parse_scope(case.question)
                observed = (first.teams, first.games, first.focus,
                            first.venues, first.opposition, first.h2h)
                self.assertEqual(observed, expected_scope(case))

    def test_followup_scope_is_checked_separately(self):
        for case in CASES:
            if case.followup is None:
                continue
            with self.subTest(case=case.name):
                prev = parse_scope(case.question)
                follow = parse_scope(case.followup, prev)
                observed = (follow.teams, follow.games, follow.focus,
                            follow.venues, follow.opposition, follow.h2h)
                self.assertEqual(observed, expected_scope(case, True))

    def test_compiler_only_selects_authorised_calls(self):
        case = CASES[4]
        planner = CompilerPlanner()
        planner.set_question(case.question)
        expected = expected_calls(expected_scope(case))
        proposals = planner("First pass: independently select tools.")
        found = {(c["name"], tuple(sorted(c["arguments"].items())))
                 for c in proposals}
        self.assertEqual(found, expected)
        self.assertEqual(planner.requests, 1)

    def test_missing_evidence_never_passes(self):
        case = CASES[0]
        scope = parse_scope(case.question)
        result = {"evidence": [], "complete": False,
                  "answer": "Unsupported assertion.", "provider_blocked": False}
        scored = score(result, scope, expected_scope(case))
        self.assertFalse(scored["passed"])
        self.assertFalse(scored["coverage_ok"])
        self.assertFalse(scored["answer_audit_ok"])

    def test_counted_planner_counts_calls(self):
        planner = CountedPlanner(lambda _: [])
        planner("plan")
        planner("repair")
        self.assertEqual(planner.requests, 2)


if __name__ == "__main__":
    unittest.main()
