"""Benchmark integrity tests: no invented model achievements, no oracle leakage."""
import unittest

from agent.investigator_v2 import Investigator, parse_scope
from scripts import evaluate_v2_investigator as bench


class InvestigatorBenchmarkTests(unittest.TestCase):
    def test_all_cases_have_independently_defined_expected_scope(self):
        self.assertEqual({name for name, _, _ in bench.CASES}, set(bench.EXPECTED))
        self.assertEqual(len(bench.CASES), 8)
        for name, question, followup in bench.CASES:
            with self.subTest(name=name):
                initial = parse_scope(question)
                final = parse_scope(followup, initial) if followup else initial
                scope_tuple = (
                    final.teams, final.games, final.h2h,
                    final.opposition, final.venues
                )
                self.assertEqual(scope_tuple, bench.EXPECTED[name])

    def test_first_pass_does_not_leak_required_tool_names(self):
        prompts = []
        def planner(prompt):
            prompts.append(prompt)
            return []
        agent = Investigator(planner)
        agent.ask("Compare Arsenal and Liverpool over the last nine matches, "
                  "including opponents faced")
        self.assertEqual(len(prompts), 2)
        self.assertIn("First pass:", prompts[0])
        self.assertNotIn("Missing evidence requirements", prompts[0])
        self.assertNotIn("compare_opponent_strength", prompts[0])
        self.assertIn("Second and FINAL pass", prompts[1])
        self.assertIn("compare_opponent_strength", prompts[1])

    def test_only_guided_control_receives_expected_calls(self):
        plans = [
            {"name": "form_summary",
             "arguments": {"team": "Arsenal", "games": 6}},
            {"name": "form_summary",
             "arguments": {"team": "Chelsea", "games": 6}},
            {"name": "compare_opponent_strength",
             "arguments": {"team1": "Arsenal",
                           "team2": "Chelsea", "games": 6}},
        ]
        self.assertEqual(bench._oracle_planner("First pass:", plans), plans[:2])
        self.assertEqual(bench._oracle_planner("Second and FINAL pass", plans),
                         plans[2:])


if __name__ == "__main__":
    unittest.main()
