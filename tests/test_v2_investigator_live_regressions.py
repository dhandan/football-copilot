"""Regression tests from live Brighton/Brentford and Chelsea/Tottenham failures."""
import unittest

from agent.investigator_v2 import Investigator, parse_scope, requirements
from scripts.audit_v2_investigator_answers import audit_answer


class LiveFailureRegressions(unittest.TestCase):
    def test_switch_discards_named_old_clubs(self):
        q = ("Forget Brighton and Brentford. Compare Chelsea and Tottenham's "
             "last eight matches and their historical head-to-head record.")
        scope = parse_scope(q)
        self.assertEqual(scope.teams, ("Chelsea", "Tottenham"))
        self.assertEqual(scope.games, 8)
        self.assertTrue(scope.h2h)
        self.assertEqual(len(requirements(scope)), 3)

    def test_switch_in_same_thread_executes_new_teams(self):
        def form_summary(team, games):
            return {"source": "test", "data": [{
                "Played": games, "Points": 10, "PointsPerGame": 1.25,
                "GoalsFor": 12, "GoalsAgainst": 9}]}
        def head_to_head(team1, team2):
            return {"data": [{"Date": "2024-01-01", "Venue": "Home",
                              "GoalsFor": 1, "GoalsAgainst": 0}]}
        agent = Investigator(
            lambda prompt: requirements(parse_scope(
                "Compare Chelsea and Tottenham's last eight matches and "
                "historical head-to-head record"))
            if "Chelsea" in prompt else [],
            tools={"form_summary": form_summary, "head_to_head": head_to_head}
        )
        agent.ask("Brighton and Brentford last seven games", thread_id="same")
        result = agent.ask(
            "Forget Brighton and Brentford. Compare Chelsea and Tottenham's "
            "last eight matches and their historical head-to-head record.",
            thread_id="same")
        self.assertTrue(result["complete"], result["trace"])
        self.assertIn("Historical head-to-head results", result["answer"])
        self.assertNotIn("Brighton", result["answer"])

    def test_attack_question_reports_goals_for_and_points(self):
        q = ("Over their last seven Premier League matches, which team has been "
             "better going forward, Brighton or Brentford? "
             "Compare goals scored and points earned.")
        scope = parse_scope(q)
        self.assertEqual(scope.focus, "attack")
        def form_summary(team, games):
            gf, ga, pts = ((17, 3, 16) if team == "Brighton" else (12, 8, 10))
            return {"source": "test", "data": [{
                "Played": games, "Points": pts,
                "PointsPerGame": pts / games,
                "GoalsFor": gf, "GoalsAgainst": ga}]}
        agent = Investigator(lambda _: requirements(scope), tools={
            "form_summary": form_summary})
        result = agent.ask(q)
        self.assertTrue(result["complete"])
        self.assertTrue(result["answer"].startswith(
            "### Answer\nBrighton scored more goals"))
        self.assertIn("Brighton earned 16 points and Brentford earned 10", result["answer"])
        self.assertNotIn("Overall verdict: Brighton conceded", result["answer"])
        self.assertEqual(audit_answer(result["answer"], scope, result["evidence"]), [])

    def test_attack_away_followup_inherits_focus(self):
        first = parse_scope("Brighton versus Brentford goals scored last seven games")
        second = parse_scope("Does that advantage hold in away matches only?", first)
        self.assertEqual(second.teams, ("Brighton", "Brentford"))
        self.assertEqual(second.games, 7)
        self.assertEqual(second.focus, "attack")
        self.assertTrue(second.venues)

    def test_actual_result_snapshots_support_away_attack_followup(self):
        """End-to-end, real repository match data; no Groq/network calls."""
        from agent.football_tools import form_summary
        question = ("Over their last seven Premier League matches, "
                    "which team has been better going forward, Brighton or "
                    "Brentford? Compare goals scored and points earned.")
        agent = Investigator(
            lambda prompt: [
                {"name": "form_summary", "arguments": {"team": team, "games": 7}}
                for team in ("Brighton", "Brentford")
            ])
        first = agent.ask(question, thread_id="real-away-attack")
        self.assertTrue(first["complete"], first["trace"])
        followup = agent.ask(
            "Does that attacking advantage still hold when comparing "
            "their away matches only?", thread_id="real-away-attack"
        )
        self.assertTrue(followup["complete"], followup["trace"])
        self.assertIn("**Away attacking verdict:**", followup["answer"])
        self.assertIn("goals scored", followup["answer"])
        self.assertNotIn("Unresolved evidence:", followup["answer"])
        scope = agent.context["real-away-attack"]
        self.assertEqual(scope.focus, "attack")
        self.assertEqual(scope.games, 7)
        for team in scope.teams:
            row = form_summary(team, 7, season="2026/27")["data"][0]
            self.assertEqual(row["VenueKnown"], row["Played"])
            self.assertEqual(
                sum(row["VenueSplit"][venue]["GoalsFor"]
                    for venue in ("Home", "Away")), row["GoalsFor"])
            self.assertEqual(
                sum(row["VenueSplit"][venue]["GoalsAgainst"]
                    for venue in ("Home", "Away")), row["GoalsAgainst"])
        self.assertEqual(audit_answer(followup["answer"], scope,
                                      followup["evidence"]), [])

    def test_no_literal_backslash_n_in_attack_answer(self):
        q = "Compare Brighton and Brentford goals scored over last seven games"
        def form_summary(team, games):
            return {"source": "test", "data": [{
                "Played": games, "Points": 10, "PointsPerGame": 10/games,
                "GoalsFor": 12, "GoalsAgainst": 7}]}
        scope = parse_scope(q)
        agent = Investigator(lambda _: requirements(scope),
                             tools={"form_summary": form_summary})
        result = agent.ask(q)
        self.assertTrue(result["complete"])
        self.assertNotIn(chr(92) + "n", result["answer"])

    def test_attack_away_missing_scored_goals_does_not_pass(self):
        first = "Compare Brighton and Brentford goals scored over last seven games"
        def form_summary(team, games):
            return {"source": "test", "data": [{
                "Played": games, "Points": 10, "PointsPerGame": 10/games,
                "GoalsFor": 12, "GoalsAgainst": 7, "VenueKnown": games,
                "VenueSplit": {"Home": {"Played": 3, "GoalsAgainst": 4},
                               "Away": {"Played": 4, "GoalsAgainst": 3}}}]}
        agent = Investigator(
            lambda prompt: requirements(parse_scope(
                "Compare Brighton and Brentford goals scored over last seven games")),
            tools={"form_summary": form_summary})
        agent.ask(first, thread_id="x")
        second = agent.ask("Does that attacking advantage still hold away?", thread_id="x")
        self.assertFalse(second["complete"])
        self.assertTrue(second["gaps"])
        self.assertNotIn("Away attacking verdict:** Brighton", second["answer"])


if __name__ == "__main__":
    unittest.main()
