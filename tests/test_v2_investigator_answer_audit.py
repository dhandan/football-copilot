"""Negative controls for the independent rendered-answer audit."""
import unittest

from agent.investigator_v2 import Investigator
from scripts.audit_v2_investigator_answers import audit_answer


class AnswerAuditTests(unittest.TestCase):
    def setUp(self):
        def form_summary(team, games):
            return {
                "source": "test fixtures",
                "data": [{
                    "Played": games, "Points": 9, "PointsPerGame": 1.5,
                    "GoalsFor": 8, "GoalsAgainst": 7,
                    "VenueKnown": games,
                    "VenueSplit": {
                        "Home": {"Played": 3, "GoalsAgainst": 4},
                        "Away": {"Played": 3, "GoalsAgainst": 3},
                    },
                }],
            }
        calls = [
            {"name": "form_summary",
             "arguments": {"team": "Arsenal", "games": 6}},
            {"name": "form_summary",
             "arguments": {"team": "Liverpool", "games": 6}},
        ]
        self.agent = Investigator(lambda _: calls, tools={"form_summary": form_summary})
        self.result = self.agent.ask(
            "Compare Arsenal and Liverpool defending at home and away "
            "over their last six matches", thread_id="audit"
        )
        self.scope = self.agent.context["audit"]

    def test_valid_rendered_evidence_passes(self):
        self.assertTrue(self.result["complete"])
        self.assertEqual(audit_answer(self.result["answer"], self.scope,
                                      self.result["evidence"]), [])

    def test_mutated_goal_total_is_rejected(self):
        answer = self.result["answer"].replace(
            "| Arsenal | 6 | 9 | 1.50 | 8 | 7 |",
            "| Arsenal | 6 | 9 | 1.50 | 8 | 99 |"
        )
        self.assertNotEqual(answer, self.result["answer"])
        self.assertIn("wrong_or_missing_form_row:Arsenal",
                      audit_answer(answer, self.scope, self.result["evidence"]))

    def test_mutated_away_rate_is_rejected(self):
        answer = self.result["answer"].replace(
            "Arsenal, away: 3 goals conceded in 3 matches (1.00 per match).",
            "Arsenal, away: 3 goals conceded in 3 matches (5.00 per match)."
        )
        self.assertNotEqual(answer, self.result["answer"])
        self.assertIn("wrong_or_missing_venue:Arsenal:Away",
                      audit_answer(answer, self.scope, self.result["evidence"]))

    def test_missing_venue_verdict_is_rejected(self):
        answer = self.result["answer"].replace(
            "**Away verdict:** Both teams conceded at the same rate",
            "**Away comparison:** Both teams conceded at the same rate"
        )
        self.assertNotEqual(answer, self.result["answer"])
        self.assertIn("missing_or_wrong_venue_verdict:Away",
                      audit_answer(answer, self.scope, self.result["evidence"]))

    def test_missing_overall_verdict_is_rejected(self):
        answer = self.result["answer"].replace(
            "**Overall verdict:** The teams conceded equally",
            "**Summary:** The teams conceded equally"
        )
        self.assertNotEqual(answer, self.result["answer"])
        self.assertIn("missing_or_wrong_overall_verdict",
                      audit_answer(answer, self.scope, self.result["evidence"]))

    def test_answer_starts_with_direct_conclusion(self):
        self.assertTrue(self.result["answer"].startswith(
            "### Answer\\nArsenal and Liverpool conceded the same number of goals"))
        
    def test_missing_direct_answer_is_rejected(self):
        answer = self.result["answer"].replace("### Answer", "### Data")
        self.assertIn("missing_or_wrong_lead",
                      audit_answer(answer, self.scope, self.result["evidence"]))

    def test_small_opponent_difference_is_calibrated(self):
        def form_summary(team, games):
            return {
                "source": "test fixtures",
                "data": [{"Played": games, "Points": 9, "PointsPerGame": 1.5,
                          "GoalsFor": 8, "GoalsAgainst": 7}],
            }
        def compare_opponent_strength(team1, team2, games):
            return {"source": "prior match form", "data": {
                team1: {"rated": 5, "played": 6, "mean_opponent_prior_ppg": 1.60},
                team2: {"rated": 6, "played": 6, "mean_opponent_prior_ppg": 1.56},
            }}
        from agent.investigator_v2 import requirements, parse_scope
        question = ("Compare Arsenal and Liverpool over their last six matches "
                    "including opponents faced")
        needed = requirements(parse_scope(question))
        agent = Investigator(lambda _: needed, tools={
            "form_summary": form_summary,
            "compare_opponent_strength": compare_opponent_strength})
        result = agent.ask(question)
        self.assertTrue(result["complete"])
        self.assertIn("difference 0.04", result["answer"])
        self.assertIn("rated coverage 5/6 versus 6/6", result["answer"])
        self.assertIn("not proof of materially tougher opposition", result["answer"])
        self.assertEqual(audit_answer(result["answer"], agent.context["default"],
                                      result["evidence"]), [])

    def test_missing_source_is_rejected(self):
        answer = self.result["answer"].replace("*Form source:", "*Unknown source:")
        self.assertIn("missing_form_attribution",
                      audit_answer(answer, self.scope, self.result["evidence"]))


if __name__ == "__main__":
    unittest.main()
