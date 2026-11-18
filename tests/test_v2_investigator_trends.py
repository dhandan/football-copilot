"""Offline and real-data regression for two distinct five-match windows."""
import unittest

from agent.investigator_v2 import Investigator, parse_scope, requirements
from scripts.audit_v2_investigator_answers import audit_answer


class TwoWindowInvestigationTests(unittest.TestCase):
    def test_five_before_requires_ten_and_preserves_metric(self):
        question = ("Has Arsenal's defence improved over its most recent five "
                    "matches compared with the five before?")
        scope = parse_scope(question)
        self.assertEqual(scope.teams, ("Arsenal",))
        self.assertEqual(scope.games, 10)
        self.assertEqual(scope.focus, "defence")
        self.assertTrue(scope.trend)
        self.assertEqual(requirements(scope), [
            {"name": "form_summary", "arguments": {"team": "Arsenal", "games": 10}}])
        self.assertFalse(parse_scope("Arsenal last five matches").trend)

    def test_real_data_defensive_and_attacking_trends(self):
        questions = [
            ("Has Arsenal's defence improved in its most recent five matches "
             "compared with the previous five?", "defence"),
            ("Has Liverpool's scoring improved in its most recent five matches "
             "compared with the five before?", "attack"),
        ]
        for question, focus in questions:
            with self.subTest(focus=focus):
                scope = parse_scope(question)
                self.assertEqual(scope.games, 10)
                self.assertTrue(scope.trend)
                self.assertEqual(scope.focus, focus)
                agent = Investigator(lambda _: requirements(scope))
                result = agent.ask(question)
                self.assertTrue(result["complete"], result["trace"])
                self.assertIn("Two-window form investigation", result["answer"])
                self.assertIn("the preceding five", result["answer"])
                self.assertEqual(
                    audit_answer(result["answer"], scope, result["evidence"]), [])

    def test_rejects_incomplete_or_inconsistent_two_window_metrics(self):
        question = ("Has Arsenal's defence improved in its most recent five "
                    "matches compared with the five before?")
        scope = parse_scope(question)
        def broken(team, games):
            return {"source": "test", "data": [{
                "Played": games, "Points": 15, "PointsPerGame": 1.5,
                "GoalsFor": 12, "GoalsAgainst": 14,
                "ConcededRecent5": 2, "ConcededPrevious5": 3,
                "PointsRecent5": 9, "PointsPrevious5": 6}]}
        agent = Investigator(lambda _: requirements(scope), tools={
            "form_summary": broken})
        result = agent.ask(question)
        self.assertFalse(result["complete"])
        self.assertTrue(result["gaps"])
        self.assertNotIn("### Answer", result["answer"])

    def test_mutated_trend_verdict_is_detected(self):
        question = ("Has Arsenal's defence improved in its most recent five "
                    "matches compared with the previous five?")
        scope = parse_scope(question)
        agent = Investigator(lambda _: requirements(scope))
        result = agent.ask(question)
        self.assertTrue(result["complete"])
        mutated = result["answer"].replace("most recent five versus",
                                            "most recent six versus")
        self.assertNotEqual(mutated, result["answer"])
        self.assertIn("missing_or_wrong_trend:Arsenal",
                      audit_answer(mutated, scope, result["evidence"]))


if __name__ == "__main__":
    unittest.main()
