"""Offline, no network test suite for experimental two-round investigation."""
import unittest

from agent.investigator_v2 import (
    Investigator, MAX_ROUNDS, MAX_TOOLS, parse_scope, requirements,
)


def form(team, games, **extra):
    return {"data": [{"Played": games, "Points": 2 * games,
                       "PointsPerGame": 2.0, "GoalsFor": games + 3,
                       "GoalsAgainst": games - 1,
                       "VenueKnown": games, "VenueSplit": {
                           "Home": {"Played": games // 2, "GoalsAgainst": 2},
                           "Away": {"Played": games - games // 2, "GoalsAgainst": 3}},
                       **extra}],
            "source": "offline verified fixtures"}


def opponents(team1, team2, games):
    return {"source": "offline prior matches", "data": {
        team1: {"played": games, "rated": games,
                "mean_opponent_prior_ppg": 1.4},
        team2: {"played": games, "rated": games - 1,
                "mean_opponent_prior_ppg": 1.7}}}


class InvestigatorTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.tools = {
            "form_summary": lambda team, games: form(team, games),
            "compare_opponent_strength": opponents,
            "head_to_head": lambda team1, team2: {
                "data": [{"Date": "2025-05-11", "Venue": "Home",
                          "GoalsFor": 2, "GoalsAgainst": 1}]},
        }

    def test_scope_extracts_general_comparison_and_opposition(self):
        q = ("Did Brighton and Brentford face tougher opponents "
             "in their last seven Premier League matches?")
        scope = parse_scope(q)
        self.assertEqual(scope.teams, ("Brighton", "Brentford"))
        self.assertEqual(scope.games, 7)
        self.assertEqual(len(requirements(scope)), 3)

    def test_question_requires_two_rounds_to_gather_missing_evidence(self):
        def planner(prompt):
            self.calls.append(prompt)
            if len(self.calls) == 1:
                return [{"name": "form_summary",
                         "arguments": {"team": "Arsenal", "games": 10}},
                        {"name": "form_summary",
                         "arguments": {"team": "Liverpool", "games": 10}}]
            return [{"name": "compare_opponent_strength", "arguments": {
                "team1": "Arsenal", "team2": "Liverpool", "games": 10}}]
        agent = Investigator(planner, tools=self.tools)
        result = agent.ask("Did Arsenal and Liverpool face weaker opponents in "
                           "their last 10 matches?")
        self.assertTrue(result["complete"])
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(len(result["evidence"]), 3)
        self.assertEqual(len(self.calls), 2)
        self.assertIn("Pre-match opponent form", result["answer"])
        self.assertIn("observational", result["answer"])
        self.assertEqual(result["trace"][-1], "complete")

    def test_invalid_tool_and_wrong_arguments_cannot_execute(self):
        def planner(_):
            return [{"name": "delete_predictions", "arguments": {}},
                    {"name": "form_summary", "arguments": {
                        "team": "Arsenal", "games": 10, "write": True}}]
        agent = Investigator(planner, tools=self.tools)
        result = agent.ask("How has Arsenal played in the last 10 games?")
        self.assertFalse(result["complete"])
        self.assertFalse(result["evidence"])
        self.assertEqual(result["rounds"], MAX_ROUNDS)
        self.assertIn("Unresolved evidence", result["answer"])

    def test_does_not_silently_fill_missing_opposition_data(self):
        def planner(prompt):
            if "compare_opponent_strength" in prompt:
                return [{"name": "compare_opponent_strength",
                         "arguments": {"team1": "Chelsea", "team2": "Tottenham",
                                       "games": 5}}]
            return []
        agent = Investigator(planner, tools=self.tools)
        result = agent.ask("Did Chelsea and Tottenham play stronger opponents "
                           "over their last five matches?")
        self.assertFalse(result["complete"])
        self.assertTrue(result["gaps"])
        self.assertIn("Unresolved evidence", result["answer"])

    def test_followup_inherits_teams_not_new_subject(self):
        agent = Investigator(lambda _: [], tools=self.tools)
        first = agent.ask("Compare Arsenal and Liverpool over their last 10 matches",
                          thread_id="thread-1")
        second = agent.ask("What about away?", thread_id="thread-1")
        self.assertFalse(first["complete"])
        self.assertIn("Arsenal", second["answer"])
        self.assertIn("Liverpool", second["answer"])
        self.assertEqual(second["rounds"], 2)
        unrelated = agent.ask("What about Brighton over their last five matches?",
                              thread_id="thread-1")
        self.assertIn("Brighton", unrelated["answer"])
        self.assertNotIn("Liverpool", unrelated["answer"])

    def test_context_is_separate_per_thread(self):
        agent = Investigator(lambda _: [], tools=self.tools)
        agent.ask("Arsenal last 10 matches", thread_id="A")
        response = agent.ask("What about away?", thread_id="B")
        self.assertEqual(response["trace"], ["scope_unresolved"])

    def test_rejects_destructive_and_unrelated_questions_before_planner(self):
        def planner(_):
            raise AssertionError("Must not plan")
        agent = Investigator(planner, tools=self.tools)
        for q in ("Delete Arsenal results", "What's the weather in London?"):
            with self.subTest(q=q):
                result = agent.ask(q)
                self.assertEqual(result["trace"], ["scope_rejected"])
                self.assertEqual(result["rounds"], 0)

    def test_venue_rates_come_from_verified_rows(self):
        calls = [{"name": "form_summary",
                  "arguments": {"team": "Brighton", "games": 6}}]
        agent = Investigator(lambda _: calls, tools=self.tools)
        result = agent.ask("Brighton home and away over their last six matches")
        self.assertTrue(result["complete"])
        self.assertIn("brighton", result["answer"].lower())
        self.assertIn("away: 3 goals", result["answer"])
        self.assertIn("home: 2 goals", result["answer"])

    def test_historical_h2h_is_labelled(self):
        calls = [{"name": "head_to_head", "arguments": {
            "team1": "Liverpool", "team2": "Arsenal"}}]
        agent = Investigator(lambda _: calls, tools=self.tools)
        result = agent.ask("Liverpool and Arsenal head-to-head results?")
        self.assertTrue(result["complete"])
        self.assertIn("historical Premier League", result["answer"])
        self.assertIn("2025-05-11", result["answer"])

    def test_real_repo_two_round_investigation_uses_verified_results(self):
        """Integration: real current-season form and pre-match opponent proxy."""
        seen = []
        def planner(prompt):
            seen.append(prompt)
            if len(seen) == 1:
                return [
                    {"name": "form_summary",
                     "arguments": {"team": "Arsenal", "games": 10}},
                    {"name": "form_summary",
                     "arguments": {"team": "Liverpool", "games": 10}},
                ]
            return [{"name": "compare_opponent_strength",
                     "arguments": {"team1": "Arsenal",
                                   "team2": "Liverpool", "games": 10}}]
        agent = Investigator(planner)
        result = agent.ask(
            "Did Arsenal and Liverpool have different defensive records "
            "because of the opponents they faced over their last 10 matches?",
            thread_id="real-data-smoke"
        )
        self.assertEqual(result["rounds"], 2)
        self.assertTrue(result["complete"], result["trace"])
        self.assertEqual(len(result["evidence"]), 3)
        self.assertIn("Pre-match opponent form", result["answer"])
        self.assertIn("descriptive, not causal", result["answer"].lower())
        self.assertIn("source:", result["answer"].lower())

    def test_two_round_and_six_call_bounds(self):
        self.assertEqual(MAX_ROUNDS, 2)
        self.assertEqual(MAX_TOOLS, 6)


if __name__ == "__main__":
    unittest.main()
