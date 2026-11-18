"""Regression checks for experimental investigator evidence quality."""
import unittest

from agent.groq_v2 import GroqRateLimited
from agent.investigator_v2 import Investigator, _execute


def form_result(known=6, home=3, away=3):
    return {
        "data": [{
            "Played": 6, "Points": 9, "PointsPerGame": 1.5,
            "GoalsFor": 8, "GoalsAgainst": 7,
            "VenueKnown": known,
            "VenueSplit": {
                "Home": {"Played": home, "GoalsAgainst": 4},
                "Away": {"Played": away, "GoalsAgainst": 3},
            },
        }],
        "source": "fixture-test",
    }


class InvestigatorEvidenceQualityTests(unittest.TestCase):
    def test_venue_missing_is_rejected(self):
        def form_summary(**kwargs):
            return form_result(known=5)
        call = {"name": "form_summary", "arguments": {"team": "Arsenal", "games": 6}}
        result = _execute(call, {"form_summary": form_summary}, require_venues=True)
        self.assertIn("error", result)

    def test_venue_zero_away_is_rejected(self):
        def form_summary(**kwargs):
            return form_result(home=6, away=0)
        call = {"name": "form_summary", "arguments": {"team": "Arsenal", "games": 6}}
        result = _execute(call, {"form_summary": form_summary}, require_venues=True)
        self.assertIn("error", result)

    def test_valid_venue_is_accepted(self):
        def form_summary(**kwargs):
            return form_result()
        call = {"name": "form_summary", "arguments": {"team": "Arsenal", "games": 6}}
        result = _execute(call, {"form_summary": form_summary}, require_venues=True)
        self.assertIn("result", result)

    def test_general_form_does_not_require_venue(self):
        def form_summary(**kwargs):
            return form_result(known=0)
        call = {"name": "form_summary", "arguments": {"team": "Arsenal", "games": 6}}
        result = _execute(call, {"form_summary": form_summary})
        self.assertIn("result", result)

    def test_rate_limit_stops_after_one_planner_attempt(self):
        attempts = []
        def planner(prompt):
            attempts.append(prompt)
            raise GroqRateLimited("429")
        investigator = Investigator(planner, tools={})
        result = investigator.ask("Compare Arsenal and Liverpool over their last six matches")
        self.assertTrue(result["provider_blocked"])
        self.assertFalse(result["complete"])
        self.assertEqual(result["rounds"], 1)
        self.assertEqual(len(attempts), 1)
        self.assertIn("provider_blocked:GroqRateLimited", result["trace"])
        self.assertEqual(len(result["gaps"]), 2)


if __name__ == "__main__":
    unittest.main()
