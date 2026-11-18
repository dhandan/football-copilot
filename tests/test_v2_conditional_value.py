"""Offline controls: conditional investigation must add grounded evidence."""
import unittest

from agent.investigator_v2 import (
    Investigator, conditional_opposition_evidence, parse_scope,
)
from scripts.evaluate_v2_conditional_value import CASES, form_only, score_context


class ConditionalValueTests(unittest.TestCase):
    def test_independent_real_source_cases_have_verified_differences(self):
        self.assertGreaterEqual(len(CASES), 3)
        for case in CASES:
            with self.subTest(case=case.name):
                scope, baseline, answer = form_only(case.question)
                self.assertEqual(scope.teams, case.teams)
                self.assertEqual(scope.games, case.games)
                self.assertTrue(all("result" in e for e in baseline))
                candidate = conditional_opposition_evidence(scope, baseline)
                self.assertEqual(candidate is not None, case.expect_context)
                self.assertIn("Verified football evidence", answer)

    def test_equal_defensive_totals_are_negative_control(self):
        case = CASES[0]
        scope, baseline, _ = form_only(case.question)
        self.assertFalse(case.expect_context)
        self.assertIsNone(conditional_opposition_evidence(scope, baseline))
        a, b = [item["result"]["data"][0]["GoalsAgainst"] for item in baseline]
        self.assertEqual(a, b)

    def test_form_only_does_not_claim_opponent_context(self):
        scope, evidence, answer = form_only(CASES[0].question)
        self.assertEqual(len(evidence), 2)
        self.assertNotIn("Pre-match opponent form", answer)

    def test_extra_data_without_causal_guardrail_is_not_a_pass(self):
        case = CASES[1]
        scope, baseline, _ = form_only(case.question)
        candidate = conditional_opposition_evidence(scope, baseline)
        self.assertIsNotNone(candidate)
        fake = {"evidence": baseline, "complete": True, "gaps": [],
                "answer": "It was entirely because of the opposition.",
                "provider_blocked": False}
        result = score_context(case, scope, baseline, fake, candidate)
        self.assertFalse(result["pass"])
        self.assertFalse(result["causal_guardrail"])

    def test_no_extra_opponent_tool_is_not_counted_as_value(self):
        case = CASES[0]
        scope, baseline, base_answer = form_only(case.question)
        candidate = conditional_opposition_evidence(scope, baseline)
        fake = {"evidence": baseline, "complete": True, "gaps": [],
                "answer": base_answer +
                          "\n\nCannot establish why and cannot prove causation.",
                "provider_blocked": False}
        scored = score_context(case, scope, baseline, fake, candidate)
        self.assertFalse(scored["pass"])
        self.assertFalse(scored["extra_context"])


if __name__ == "__main__":
    unittest.main()
