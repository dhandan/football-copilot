"""V2 why-reasoning acceptance: deterministic, sourced, no causal invention."""
import unittest
from unittest.mock import patch

from agent.defensive_explanation import defensive_evidence
from agent.langgraph_v2 import Conversation


def sample(team, games):
    conceded = 6 if team == "Arsenal" else 13
    return {
        "type": "form_summary",
        "season": "2026/27, 2025/26",
        "requested_games": games,
        "source": "verified 2026/27 results and 2025/26 DuckDB",
        "data": [{
            "Played": games, "Won": 4, "Drawn": 2,
            "Lost": games - 6, "Points": 14, "PointsPerGame": round(14 / games, 2),
            "GoalsFor": 16, "GoalsAgainst": conceded,
            "CleanSheets": 4 if team == "Arsenal" else 1,
            "Conceded2Plus": 1 if team == "Arsenal" else 4,
            "ConcededRecent5": 2 if team == "Arsenal" else 8,
            "ConcededPrevious5": 4 if team == "Arsenal" else 5,
        }],
    }


class DefensiveWhyTests(unittest.TestCase):
    def test_evidence_explicitly_explains_results_not_tactics(self):
        a = sample("Arsenal", 10)["data"][0]
        l = sample("Liverpool", 10)["data"][0]
        text = defensive_evidence("Arsenal", a, "Liverpool", l)
        self.assertIn("6 in 10 games (0.60 per match)", text)
        self.assertIn("13 (1.30 per match)", text)
        self.assertIn("4 clean sheets versus 1", text)
        self.assertIn("two or more in 1 match versus 4", text)
        self.assertIn("most recent five matches Arsenal conceded 2", text)
        self.assertIn("descriptive patterns", text)
        self.assertIn("cannot establish tactics", text)

    def test_why_direct_question_routes_to_two_tools_without_llm(self):
        def no_llm(_):
            raise AssertionError("Deterministic why comparison should not call remote planner")
        with patch.dict("agent.langgraph_v2.ALLOWED", {"form_summary": sample}):
            chat = Conversation(planner=no_llm)
            result = chat.ask(
                "Why is Arsenal defensively stronger than Liverpool over their last 10 matches?",
                thread_id="why-direct",
            )
        self.assertEqual({r["arguments"]["team"] for r in result["evidence"]},
                         {"Arsenal", "Liverpool"})
        self.assertTrue(all(r["arguments"]["games"] == 10 for r in result["evidence"]))
        self.assertIn("4 clean sheets versus 1", result["answer"])
        self.assertIn("13", result["answer"])
        self.assertIn("not proof of why", result["answer"])

    def test_short_why_followup_retains_teams_window(self):
        def no_llm(_):
            raise AssertionError("Contextual why should bypass the planner")
        with patch.dict("agent.langgraph_v2.ALLOWED", {"form_summary": sample}):
            chat = Conversation(planner=no_llm)
            chat.ask("Compare Liverpool and Arsenal over their last 10 matches",
                     thread_id="why-followup")
            chat.ask("Who has been defensively better over those 10 matches?",
                     thread_id="why-followup")
            result = chat.ask("Why is that?", thread_id="why-followup")
        self.assertEqual(len(result["evidence"]), 2)
        self.assertEqual({r["arguments"]["games"] for r in result["evidence"]}, {10})
        self.assertIn("1 clean sheet versus 4", result["answer"])
        self.assertIn("not proof of why", result["answer"])

    def test_real_readonly_result_data_has_computed_defensive_patterns(self):
        from agent.football_tools import form_summary
        from agent.langgraph_v2 import ask
        arsenal = form_summary("Arsenal", 10, season="2026/27")
        liverpool = form_summary("Liverpool", 10, season="2026/27")
        self.assertEqual(arsenal["data"][0]["Played"], 10)
        self.assertEqual(liverpool["data"][0]["Played"], 10)
        for team_result in (arsenal, liverpool):
            data = team_result["data"][0]
            self.assertGreaterEqual(data["CleanSheets"], 0)
            self.assertLessEqual(data["CleanSheets"], 10)
            self.assertGreaterEqual(data["Conceded2Plus"], 0)
            self.assertEqual(data["ConcededRecent5"] + data["ConcededPrevious5"],
                             data["GoalsAgainst"])
        answer = ask("Why is Arsenal defensively stronger than Liverpool over their last 10 matches?",
                     planner=lambda _: self.fail("Should use the deterministic comparison route"))
        self.assertIn("descriptive patterns", answer["answer"])
        self.assertIn("clean sheets", answer["answer"])
        self.assertIn("Source:", answer["answer"])
        self.assertIn("2026/27", answer["answer"])

    def test_why_does_not_invent_match_breakdown_when_missing(self):
        l = {"Played": 10, "GoalsAgainst": 14}
        a = {"Played": 10, "GoalsAgainst": 6}
        text = defensive_evidence("Liverpool", l, "Arsenal", a)
        self.assertIn("6 (0.60 per match)", text)
        self.assertIn("not available", text)
        self.assertNotIn("kept 0 clean sheets", text)

    def test_unequal_match_windows_do_not_claim_defensive_lead(self):
        self.assertIn("not have a comparable completed-match window",
                      defensive_evidence("Liverpool",
                                         {"Played": 9, "GoalsAgainst": 10},
                                         "Arsenal", {"Played": 10, "GoalsAgainst": 5}))


if __name__ == "__main__":
    unittest.main()
