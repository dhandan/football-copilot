"""Offline tests for the bounded Groq shadow-screen evaluation contracts."""
import unittest
from unittest.mock import patch

from scripts import evaluate_v2_groq_shadow as shadow


class GroqShadowScreenTests(unittest.TestCase):
    def test_fixed_budget_stays_within_24_request_local_cap(self):
        self.assertEqual(len(shadow.MODELS), 2)
        self.assertLessEqual(
            len(shadow.MODELS) *
            (len(shadow.PLANNER_CASES) + len(shadow.SYNTHESIS_CASES)),
            24
        )
        self.assertEqual(len(shadow.PLANNER_CASES), 8)
        self.assertEqual(len(shadow.SYNTHESIS_CASES), 3)

    def test_plan_contract_compares_all_arguments(self):
        a = [{"name": "form_summary", "arguments": {"team": "Liverpool", "games": 5}}]
        b = [{"name": "form_summary", "arguments": {"team": "Liverpool", "games": 10}}]
        self.assertNotEqual(shadow.plan_key(a), shadow.plan_key(b))
        self.assertEqual(shadow.plan_key(a), shadow.plan_key(list(reversed(a))))

    def test_synthesis_rejects_incomplete_narrative(self):
        candidate = "Liverpool and Arsenal are Premier League teams."
        grounded, complete = shadow.grounded_and_complete(
            candidate, candidate, "form", "Compare Liverpool and Arsenal"
        )
        self.assertTrue(grounded)
        self.assertFalse(complete)

    def test_synthesis_rejects_novel_numbers(self):
        facts = "Liverpool scored 9 and Arsenal scored 8 goals."
        candidate = "Liverpool scored 100 and Arsenal scored 8 goals."
        grounded, complete = shadow.grounded_and_complete(
            candidate, facts, "form", "Compare Liverpool and Arsenal"
        )
        self.assertFalse(grounded)

    def test_bounded_screen_refuses_to_run_without_credentials(self):
        with patch.object(shadow.groq_v2, "groq_ready", return_value=False):
            with patch.object(shadow.groq_v2, "groq_planner",
                              side_effect=AssertionError("No network allowed")):
                self.assertEqual(shadow.run(), 2)


if __name__ == "__main__":
    unittest.main()
