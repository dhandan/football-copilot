"""Evidence-conditioned investigation: two rounds, not preauthorised context."""
import unittest

from agent.investigator_v2 import (
    Investigator, parse_scope, requirements, conditional_opposition_evidence,
)


QUESTION = ("Why did Arsenal concede fewer than Liverpool over their "
            "last seven Premier League matches?")


def form(team, games):
    return {"source": "offline fixtures", "data": [{
        "Played": games, "Points": 14,
        "PointsPerGame": 2.0, "GoalsFor": 11,
        "GoalsAgainst": 4 if team == "Arsenal" else 9}]}


def opponent(team1, team2, games):
    return {"source": "offline prior form", "data": {
        team1: {"played": games, "rated": games,
                "mean_opponent_prior_ppg": 1.6},
        team2: {"played": games, "rated": games,
                "mean_opponent_prior_ppg": 1.5}}}


class ConditionalInvestigationTests(unittest.TestCase):
    def test_why_question_does_not_pre_authorise_opposition(self):
        scope = parse_scope(QUESTION)
        self.assertTrue(scope.investigate_why)
        self.assertFalse(scope.opposition)
        self.assertEqual(len(requirements(scope)), 2)
        self.assertEqual([x["name"] for x in requirements(scope)],
                         ["form_summary", "form_summary"])

    def test_second_round_uses_real_evidence_to_expand_tools(self):
        seen = []
        def planner(prompt):
            seen.append(prompt)
            if "Second and FINAL pass" in prompt:
                self.assertEqual(len(seen), 2)
                self.assertIn("compare_opponent_strength", prompt)
                return [{"name": "compare_opponent_strength", "arguments": {
                    "team1": "Arsenal", "team2": "Liverpool", "games": 7}}]
            self.assertNotIn("Missing evidence requirements", prompt)
            return requirements(parse_scope(QUESTION))

        agent = Investigator(planner, tools={
            "form_summary": form,
            "compare_opponent_strength": opponent})
        result = agent.ask(QUESTION)
        self.assertTrue(result["complete"], result["trace"])
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(len(result["evidence"]), 3)
        self.assertIn("context_tool_offered_after_verified_gap", result["trace"])
        self.assertIn("Pre-match opponent form", result["answer"])
        self.assertIn("cannot establish", result["answer"])

    def test_equal_goals_do_not_trigger_context_tool(self):
        def equal_form(team, games):
            row = form(team, games)
            row["data"][0]["GoalsAgainst"] = 5
            return row
        seen = []
        def planner(prompt):
            seen.append(prompt)
            return requirements(parse_scope(QUESTION))
        agent = Investigator(planner, tools={
            "form_summary": equal_form,
            "compare_opponent_strength": opponent})
        result = agent.ask(QUESTION)
        self.assertTrue(result["complete"], result["trace"])
        self.assertEqual(len(seen), 1)
        self.assertEqual(result["rounds"], 1)
        self.assertNotIn("context_tool_offered_after_verified_gap", result["trace"])

    def test_missing_verified_form_never_enables_context(self):
        scope = parse_scope(QUESTION)
        self.assertIsNone(conditional_opposition_evidence(scope, []))
        self.assertIsNone(conditional_opposition_evidence(scope, [{
            "tool": "form_summary", "arguments": {"team": "Arsenal", "games": 7},
            "result": form("Arsenal", 7)}]))

    def test_second_round_cannot_execute_unapproved_tool(self):
        def planner(prompt):
            if "Second and FINAL pass" in prompt:
                return [{"name": "head_to_head", "arguments": {
                    "team1": "Arsenal", "team2": "Liverpool"}}]
            return requirements(parse_scope(QUESTION))
        agent = Investigator(planner, tools={
            "form_summary": form,
            "compare_opponent_strength": opponent})
        result = agent.ask(QUESTION)
        self.assertFalse(result["complete"])
        self.assertEqual([e["tool"] for e in result["evidence"]],
                         ["form_summary", "form_summary"])
        self.assertEqual(len(result["gaps"]), 1)
        self.assertEqual(result["gaps"][0]["name"], "compare_opponent_strength")


if __name__ == "__main__":
    unittest.main()
