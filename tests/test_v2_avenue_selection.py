"""Isolated avenue choice: independent labels, allowlist and evidence reconciliation."""
import unittest
from agent.avenue_selection_v2 import (
    CASES, CHOICES, observed_form, venue_evidence, opponent_evidence,
    select_keyword, investigate, choice_prompt,
)


class AvenueSelectionTests(unittest.TestCase):
    def test_cases_independently_labelled(self):
        self.assertEqual(len(CASES), 6)
        self.assertEqual(len({c.label for c in CASES}), len(CASES))
        self.assertEqual({c.expected for c in CASES}, set(CHOICES))

    def test_all_cases_have_independent_verified_source_evidence(self):
        for case in CASES:
            with self.subTest(case=case.label):
                scope, first = observed_form(case.question)
                self.assertEqual(len(first), 2)
                self.assertTrue(all("result" in e for e in first))
                expected = investigate(case.question, case.expected)
                if case.expected == "none":
                    self.assertIsNone(expected["extra"])
                else:
                    self.assertTrue(expected["extra"])
                self.assertIn("not proof", expected["caveat"])

    def test_venue_totals_reconcile_and_rates_are_sourced(self):
        case = CASES[0]
        scope, first = observed_form(case.question)
        data = venue_evidence(scope, first)["data"]
        for item, base in zip(data, first):
            row = base["result"]["data"][0]
            self.assertEqual(sum(x["games"] for x in item["venue_rates"]),
                             row["Played"])
            self.assertEqual(sum(x["goals"] for x in item["venue_rates"]),
                             row[item["metric"]])

    def test_unapproved_avenue_is_rejected(self):
        with self.assertRaises(ValueError):
            investigate(CASES[0].question, "erase_data")
        with self.assertRaises(ValueError):
            investigate(CASES[0].question, "prediction")

    def test_missing_venue_data_cannot_be_replaced_with_guess(self):
        from copy import deepcopy
        scope, first = observed_form(CASES[0].question)
        altered = deepcopy(first)
        altered[0]["result"]["data"][0]["VenueKnown"] = 0
        with self.assertRaises(ValueError):
            venue_evidence(scope, altered)

    def test_simple_baseline_not_misrepresented_as_ai_value(self):
        for case in CASES:
            with self.subTest(case=case.label):
                self.assertEqual(select_keyword(case.question), case.expected)

    def test_live_failure_classified_and_later_cases_continue_without_retry(self):
        """All six cases are scored even when one provider response is invalid."""
        from unittest.mock import patch
        from agent.groq_v2 import GroqBadRequestJsonValidation
        from scripts.evaluate_v2_avenue_choice import run
        outputs = ["venue", "opposition", "venue", "opposition", "venue",
                   GroqBadRequestJsonValidation("invalid")]
        observed = []
        def fake_select(question, scope, evidence):
            observed.append(question)
            item = outputs[len(observed) - 1]
            if isinstance(item, Exception):
                raise item
            return item
        with patch("agent.groq_v2.groq_ready", return_value=True), \\
             patch("scripts.evaluate_v2_avenue_choice.select_live",
                   side_effect=fake_select):
            scores, total = run(live_groq=True)
        self.assertEqual(total, 6)
        self.assertEqual(scores["baseline"], 6)
        self.assertEqual(scores["groq"], 4)
        self.assertEqual(len(observed), 6)

    def test_prompt_exposes_only_factual_first_round_rows(self):
        case = CASES[1]
        scope, first = observed_form(case.question)
        prompt = choice_prompt(case.question, scope, first)
        self.assertIn("Verified first-round evidence", prompt)
        self.assertIn("home/away", prompt)
        self.assertIn("pre-match", prompt)
        self.assertNotIn("expected=", prompt)


if __name__ == "__main__":
    unittest.main()
