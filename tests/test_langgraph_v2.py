"""Run with python -m unittest tests.test_langgraph_v2 (no Ollama needed)."""
import unittest

from agent.langgraph_v2 import ask


class AgentGraphTests(unittest.TestCase):
    def test_h07_experiment(self):
        result = ask("Why was H07 rejected?", planner=lambda q: [
            {"name": "experiment_history", "arguments": {"experiment": "H07"}}
        ])
        self.assertIn("H07 parked", result["answer"])
        self.assertEqual(len(result["evidence"]), 1)

    def test_gw8_returns_actual_fixture_rows(self):
        result = ask("Show GW8 forecasts", planner=lambda q: [
            {"name": "frozen_gameweek_predictions", "arguments": {"gameweek": 8}}
        ], synthesizer=lambda q, facts: "This must not replace the fixture table.")
        rows = result["evidence"][0]["result"]["data"]
        self.assertEqual(len(rows), 10)
        self.assertIn("forecast_rows_preserved", result["trace"])
        self.assertIn("| Fixture | 40/60 champion | Model 2 | Adaptive challenger |", result["answer"])
        for row in rows:
            self.assertIn(f"{row['HomeTeam']} vs {row['AwayTeam']}", result["answer"])
            self.assertIn(str(row["PredictedResult"]), result["answer"])

    def test_two_team_form_comparison_uses_existing_tool(self):
        from unittest.mock import patch
        def fake_form(team, games):
            return {"data": [{"Played": games, "Won": 3, "Drawn": 1,
                              "Lost": 1, "Points": 10, "PointsPerGame": 2.0,
                              "GoalsFor": 9, "GoalsAgainst": 4}]}
        plan = lambda q: [
            {"name": "form_summary", "arguments": {"team": "Liverpool", "games": 5}},
            {"name": "form_summary", "arguments": {"team": "Arsenal", "games": 5}},
        ]
        with patch.dict("agent.langgraph_v2.ALLOWED", {"form_summary": fake_form}):
            result = ask("Compare Liverpool and Arsenal's last five games", planner=plan)
        self.assertEqual(len(result["evidence"]), 2)
        self.assertIn("Liverpool recent form", result["answer"])
        self.assertIn("Arsenal recent form", result["answer"])
        self.assertIn("Source:", result["answer"])

    def test_form_comparison_is_calculated_not_invented(self):
        from unittest.mock import patch
        values = {
            "Man City": {"Played": 5, "Won": 2, "Drawn": 2, "Lost": 1,
                         "Points": 8, "PointsPerGame": 1.6, "GoalsFor": 11,
                         "GoalsAgainst": 6},
            "Chelsea": {"Played": 5, "Won": 1, "Drawn": 1, "Lost": 3,
                        "Points": 4, "PointsPerGame": 0.8, "GoalsFor": 5,
                        "GoalsAgainst": 10},
        }
        def fake_form(team, games):
            return {"data": [values[team]]}
        with patch.dict("agent.langgraph_v2.ALLOWED", {"form_summary": fake_form}):
            result = ask("Compare Manchester City and Chelsea", planner=lambda _: [
                {"name": "form_summary", "arguments": {"team": "Man City", "games": 5}},
                {"name": "form_summary", "arguments": {"team": "Chelsea", "games": 5}},
            ], synthesizer=lambda *_: "Chelsea is better, without evidence.")
        answer = result["answer"]
        self.assertIn("Man City leads on points per game by 0.8", answer)
        self.assertIn("| Metric | Man City | Chelsea |", answer)
        self.assertIn("| Points per game | 1.6 | 0.8 |", answer)
        self.assertIn("**Key differences**", answer)
        self.assertIn("Man City conceded 4 fewer goals", answer)
        self.assertNotIn("Chelsea is better", answer)
        self.assertIn("form_facts_preserved", result["trace"])

    def test_form_comparison_does_not_claim_lead_without_two_valid_rows(self):
        from unittest.mock import patch
        def fake_form(team, games):
            if team == "Chelsea":
                return {"data": []}
            return {"data": [{"Played": 5, "Won": 2, "Drawn": 1, "Lost": 2,
                              "Points": 7, "PointsPerGame": 1.4,
                              "GoalsFor": 6, "GoalsAgainst": 5}]}
        with patch.dict("agent.langgraph_v2.ALLOWED", {"form_summary": fake_form}):
            result = ask("Compare Man City and Chelsea", planner=lambda _: [
                {"name": "form_summary", "arguments": {"team": "Man City", "games": 5}},
                {"name": "form_summary", "arguments": {"team": "Chelsea", "games": 5}},
            ])
        self.assertNotIn("**Comparison:", result["answer"])
        self.assertIn("no completed 2026/27 matches", result["answer"])

    def test_explicit_two_team_form_bypasses_slow_planner(self):
        from unittest.mock import patch
        def fail_planner(_):
            raise AssertionError("Ollama planner must not run for explicit two-team form")
        def fake_form(team, games):
            return {"data": [{"Played": games, "Won": 2, "Drawn": 1,
                              "Lost": games - 3, "Points": 7,
                              "PointsPerGame": round(7 / games, 2),
                              "GoalsFor": 5, "GoalsAgainst": 3}]}
        with patch.dict("agent.langgraph_v2.ALLOWED", {"form_summary": fake_form}):
            result = ask("Compare Liverpool and Arsenal's form over their last 5 matches",
                         planner=fail_planner)
        self.assertIn("deterministic_form_route", result["trace"])
        self.assertEqual(len(result["evidence"]), 2)
        self.assertIn("2026/27", result["answer"])

    def test_current_season_form_data_does_not_fallback_to_historical(self):
        from unittest.mock import patch
        from agent import langgraph_v2
        def fake_current_form(team, games, season=None):
            self.assertEqual(season, "2026/27")
            return {"data": [], "season": season}
        with patch.object(langgraph_v2, "form_summary", fake_current_form), \
             patch.dict(langgraph_v2.ALLOWED, {"form_summary": fake_current_form}):
            result = ask("Compare Liverpool and Arsenal's last 5 matches",
                         planner=lambda _: [])
        self.assertIn("no completed 2026/27 matches", result["answer"])
        self.assertNotIn("**Comparison:", result["answer"])

    def test_zero_match_nan_form_does_not_create_comparison(self):
        from unittest.mock import patch
        def empty_form(team, games):
            return {"data": [{"Played": 0, "Points": None,
                              "PointsPerGame": float("nan"),
                              "GoalsFor": float("nan"),
                              "GoalsAgainst": float("nan")}]}
        with patch.dict("agent.langgraph_v2.ALLOWED", {"form_summary": empty_form}):
            result = ask("Compare Liverpool and Arsenal's last 5 matches",
                         planner=lambda _: [])
        self.assertNotIn("**Comparison:", result["answer"])
        self.assertNotIn("nan", result["answer"].lower())
        self.assertIn("no completed 2026/27 matches", result["answer"])

    def test_2026_27_form_reads_verified_results_without_legacy_database(self):
        from agent.football_tools import _current_season_form
        data = _current_season_form("Liverpool", 5)
        self.assertEqual(data["season"], "2026/27")
        self.assertIn("2026/27", data["season"])
        self.assertEqual(data["requested_games"], 5)
        # The committed GW8 results record Liverpool 2-0 Brighton.
        self.assertGreaterEqual(data["data"][0]["Played"], 1)
        self.assertGreaterEqual(data["data"][0]["GoalsFor"], 2)

    def test_recent_form_fills_ten_with_previous_season(self):
        import pandas as pd
        from unittest.mock import patch
        from agent.football_tools import _current_season_form
        history = pd.DataFrame([
            {"Season": "2025/26", "Date": f"2026-05-{24-i:02d}",
             "Opponent": f"Opponent {i}", "GoalsFor": 1, "GoalsAgainst": 0}
            for i in range(10)
        ])
        with patch("agent.football_tools.get_team_form", return_value=history):
            result = _current_season_form("Liverpool", 10)
        self.assertEqual(result["data"][0]["Played"], 10)
        self.assertIn("2025/26", result["season"])
        self.assertIn("2026/27", result["season"])

    def test_defensive_question_answers_defence_not_generic_form(self):
        from unittest.mock import patch
        from agent.langgraph_v2 import Conversation
        def fake_form(team, games):
            return {"data": [{"Played": games, "Won": 4, "Drawn": 0,
                              "Lost": games - 4, "Points": 12,
                              "PointsPerGame": 1.2,
                              "GoalsFor": 14,
                              "GoalsAgainst": 13 if team == "Liverpool" else 6}]}
        def no_llm(_):
            raise AssertionError("Unambiguous defensive follow-up must not invoke Qwen")
        with patch.dict("agent.langgraph_v2.ALLOWED", {"form_summary": fake_form}):
            chat = Conversation(planner=lambda _: [
                {"name": "form_summary", "arguments": {"team": "Liverpool", "games": 10}},
                {"name": "form_summary", "arguments": {"team": "Arsenal", "games": 10}}],
                )
            chat.ask("Compare Liverpool and Arsenal's last 10 matches", thread_id="defence")
            result = chat.ask("who has been defensively better over the last 10 matches",
                              thread_id="defence")
            again = chat.ask("i asked specifically on defence who has been better in the last 10 matches????",
                             thread_id="defence")
        for response in (result, again):
            self.assertEqual(len(response["evidence"]), 2)
            self.assertIn("Arsenal have been defensively stronger", response["answer"])
            self.assertIn("13 for Liverpool", response["answer"])
            self.assertNotIn("**Key differences**\\n\\nArsenal leads on points", response["answer"])

    def test_form_validation_error_is_explained(self):
        from unittest.mock import patch
        def invalid_team(team, games):
            raise ValueError("Team 'Manchester City' was not found. Did you mean: Man City?")
        with patch.dict("agent.langgraph_v2.ALLOWED", {"form_summary": invalid_team}):
            result = ask("Show Manchester City form", planner=lambda _: [
                {"name": "form_summary", "arguments": {"team": "Manchester City", "games": 5}}
            ])
        self.assertIn("Did you mean: Man City?", result["answer"])
        self.assertIn("Tool execution failed: ValueError", result["answer"])

    def test_manchester_city_alias_and_grounded_form(self):
        from unittest.mock import patch
        seen = []
        def fake_form(team, games):
            seen.append(team)
            return {"data": [{"Played": 5, "Won": 1, "Drawn": 2, "Lost": 2,
                              "Points": 5, "PointsPerGame": 1.0,
                              "GoalsFor": 9, "GoalsAgainst": 10}]}
        with patch.dict("agent.langgraph_v2.ALLOWED", {"form_summary": fake_form}):
            result = ask("Manchester City form", planner=lambda q: [
                {"name": "form_summary", "arguments": {"team": "Manchester City", "games": 5}}
            ], synthesizer=lambda q, facts: "Invented better performance.")
        self.assertEqual(seen, ["Man City"])
        self.assertIn("form_facts_preserved", result["trace"])
        self.assertNotIn("Invented better", result["answer"])

    def test_two_team_followup_restores_missing_team(self):
        from unittest.mock import patch
        def fake_form(team, games):
            return {"data": [{"Played": games, "Won": 2, "Drawn": 1, "Lost": games - 3,
                              "Points": 7, "PointsPerGame": 1.4,
                              "GoalsFor": 5, "GoalsAgainst": 5}]}
        def planner(q):
            if "Current question:" in q:
                return [{"name": "form_summary", "arguments": {"team": "Man City", "games": 10}}]
            return [{"name": "form_summary", "arguments": {"team": "Man City", "games": 5}},
                    {"name": "form_summary", "arguments": {"team": "Chelsea", "games": 5}}]
        from agent.langgraph_v2 import Conversation
        with patch.dict("agent.langgraph_v2.ALLOWED", {"form_summary": fake_form}):
            chat = Conversation(planner)
            chat.ask("Compare Manchester City and Chelsea form", thread_id="pair")
            result = chat.ask("How does their last 10 matches compare?", thread_id="pair")
        self.assertEqual({x["arguments"]["team"] for x in result["evidence"]}, {"Man City", "Chelsea"})

    def test_form_schema_rejects_invalid_game_count(self):
        from pydantic import ValidationError
        from agent.langgraph_v2 import _Plan
        with self.assertRaises(ValidationError):
            _Plan.model_validate({"calls": [
                {"name": "form_summary", "arguments": {"team": "Liverpool", "games": 100}}
            ]})

    def test_multi_tool(self):
        result = ask("GW8 forecasts and H07?", planner=lambda q: [
            {"name": "frozen_gameweek_predictions", "arguments": {"gameweek": 8}},
            {"name": "experiment_history", "arguments": {"experiment": "H07"}},
        ])
        self.assertEqual(len(result["evidence"]), 2)
        self.assertIn("10 fixtures", result["answer"])
        self.assertIn("H07 parked", result["answer"])

    def test_conversation_followup_uses_prior_turn(self):
        from agent.langgraph_v2 import Conversation
        prompts = []
        def planner(prompt):
            prompts.append(prompt)
            if "Current question: How does that compare with champion performance?" in prompt:
                self.assertIn("Earlier user: Show GW8 forecasts", prompt)
                return [{"name": "model_performance", "arguments": {}}]
            return [{"name": "frozen_gameweek_predictions", "arguments": {"gameweek": 8}}]
        chat = Conversation(planner)
        first = chat.ask("Show GW8 forecasts", thread_id="a")
        second = chat.ask("How does that compare with champion performance?", thread_id="a")
        self.assertEqual(first["evidence"][0]["tool"], "frozen_gameweek_predictions")
        self.assertEqual(second["evidence"][0]["tool"], "model_performance")
        self.assertEqual(len(second["history"]), 2)

    def test_conversation_threads_are_isolated(self):
        from agent.langgraph_v2 import Conversation
        prompts = []
        def planner(prompt):
            prompts.append(prompt)
            return []
        chat = Conversation(planner)
        chat.ask("What is H07?", thread_id="a")
        chat.ask("What are GW8 predictions?", thread_id="b")
        self.assertNotIn("Earlier user:", prompts[1])
        self.assertEqual(len(chat.graph.get_state({"configurable": {"thread_id": "b"}}).values["history"]), 1)

    def test_followup_repairs_missing_champion_tool(self):
        from agent.langgraph_v2 import Conversation
        def planner(question):
            return ([{"name": "frozen_gameweek_predictions", "arguments": {"gameweek": 8}}]
                    if "Current question:" not in question else
                    [{"name": "frozen_gameweek_predictions", "arguments": {"gameweek": 8}}])
        chat = Conversation(planner)
        chat.ask("Show me the GW8 forecasts", thread_id="repair")
        result = chat.ask("How does that compare with champion performance?", thread_id="repair")
        self.assertEqual({x["tool"] for x in result["evidence"]},
                         {"frozen_gameweek_predictions", "model_performance"})
        self.assertIn("plan_completed_missing_comparison_evidence", result["trace"])

    def test_synthesis_rejects_invented_numeric_claims(self):
        from agent.langgraph_v2 import ask
        result = ask(
            "Compare GW8 forecasts and model performance",
            planner=lambda q: [
                {"name": "frozen_gameweek_predictions", "arguments": {"gameweek": 8}},
                {"name": "model_performance", "arguments": {}},
            ],
            synthesizer=lambda q, facts: "GW8 forecasts achieved 99% accuracy and are not yet evaluated.",
        )
        self.assertIn("synthesis_rejected_grounding", result["trace"])
        self.assertIn("GW6-GW7", result["answer"])

    def test_followup_synthesis_receives_previous_forecast_context(self):
        from agent.langgraph_v2 import Conversation
        captured = []
        def planner(question):
            if "Current question:" in question:
                return [{"name": "model_performance", "arguments": {}}]
            return [{"name": "frozen_gameweek_predictions", "arguments": {"gameweek": 8}}]
        def synthesizer(question, facts):
            captured.append(facts)
            if "champion" in question:
                return ("GW8 contains evaluated results: market only scored 40%, "
                        "the provisional 40/60 blend and Model 2 both scored 30%. "
                        "One gameweek is a small sample; formal promotion remains pending.")
            return "GW8 frozen forecasts: 10 fixtures, saved and not yet evaluated."
        conversation = Conversation(planner=planner, synthesizer=synthesizer)
        conversation.ask("Show GW8 forecasts", thread_id="followup")
        result = conversation.ask("Compare with champion performance", thread_id="followup")
        self.assertIn("Previous turn", captured[-1])
        self.assertIn("GW8 frozen forecasts", captured[-1])
        self.assertIn("synthesis_accepted", result["trace"])
        self.assertIn("40%", result["answer"])

    def test_performance_uses_completed_gw8_scorecard(self):
        result = ask(
            "Which model performed best in GW8?",
            planner=lambda q: [{"name": "model_performance", "arguments": {}}],
        )
        self.assertIn("GW8 (10 finished matches)", result["answer"])
        self.assertIn("adaptive 3/10 (30%)", result["answer"])
        self.assertIn("market only 4/10 (40%)", result["answer"])
        self.assertIn("docs/gameweeks/GW08.md", result["answer"])
        self.assertNotIn("GW8 forecast accuracy is unknown", result["answer"])

    def test_synthesis_accepts_small_20_match_sample_phrasing(self):
        result = ask(
            "Which model is performing best so far and why?",
            planner=lambda q: [{"name": "model_performance", "arguments": {}}],
            synthesizer=lambda q, facts: (
                "GW8: the 40/60 provisional candidate and adaptive each scored 30%, "
                "while market only scored 40%. This is one gameweek, a small sample; "
                "formal promotion remains pending."
            ),
        )
        self.assertIn("synthesis_accepted", result["trace"])

    def test_synthesis_rejects_reversed_performance_winner(self):
        result = ask(
            "Which model is performing best and why?",
            planner=lambda q: [{"name": "model_performance", "arguments": {}}],
            synthesizer=lambda q, facts: (
                "Model 2 is the best performing model with 55% accuracy versus 40% "
                "for the 40/60 blend across 20 matches. "
                "This is a small sample and promotion remains provisional."
            ),
        )
        self.assertIn("synthesis_rejected_grounding", result["trace"])
        self.assertIn("40/60", result["answer"])

    def test_synthesis_rejects_denial_of_available_performance(self):
        from agent.langgraph_v2 import ask
        result = ask(
            "How does that compare with the champion performance?",
            planner=lambda q: [{"name": "model_performance", "arguments": {}}],
            synthesizer=lambda q, facts: (
                "The evidence does not provide clear information about champion performance."
            ),
        )
        self.assertIn("synthesis_rejected_grounding", result["trace"])
        self.assertIn("55%", result["answer"])
        self.assertIn("40%", result["answer"])

    def test_synthesis_accepts_grounded_comparison(self):
        from agent.langgraph_v2 import ask
        result = ask(
            "Compare GW8 forecasts and model performance",
            planner=lambda q: [
                {"name": "frozen_gameweek_predictions", "arguments": {"gameweek": 8}},
                {"name": "model_performance", "arguments": {}},
            ],
            synthesizer=lambda q, facts: (
                "GW8 contains 10 saved forecasts and 10 completed results. "
                "GW8: market only scored 40% while the 40/60 provisional candidate "
                "and Model 2 both scored 30%. One gameweek is a small sample; "
                "formal promotion remains pending."
            ),
        )
        self.assertIn("synthesis_accepted", result["trace"])

    def test_synthesis_error_falls_back_to_facts(self):
        from agent.langgraph_v2 import ask
        def fail(q, facts):
            raise ValueError("invalid model output")
        result = ask(
            "H07", planner=lambda q: [{"name": "experiment_history", "arguments": {"experiment": "H07"}}],
            synthesizer=fail,
        )
        self.assertIn("synthesis_fallback: ValueError", result["trace"])
        self.assertIn("H07 parked", result["answer"])

    def test_strict_planner_schema(self):
        from pydantic import ValidationError
        from agent.langgraph_v2 import _Plan
        good = _Plan.model_validate({
            "calls": [
                {"name": "model_performance", "arguments": {}},
                {"name": "frozen_gameweek_predictions", "arguments": {"gameweek": 8}},
            ]
        })
        self.assertEqual(len(good.calls), 2)
        for bad in (
            {"calls": [{"name": "made_up_tool", "arguments": {}}]},
            {"calls": [{"name": "model_performance", "arguments": {"champion": "x"}}]},
            {"calls": [{"name": "frozen_gameweek_predictions", "arguments": {"gameweek": 40}}]},
        ):
            with self.assertRaises(ValidationError):
                _Plan.model_validate(bad)

    def test_weather_does_not_call_football_tools(self):
        def forbidden(question):
            raise AssertionError("Weather must not reach the football planner")
        result = ask("Can you tell me today's weather in London?", planner=forbidden)
        self.assertEqual(result["evidence"], [])
        self.assertIn("rejected_out_of_scope", result["trace"])

    def test_h07_synthesis_requires_rejection_explanation(self):
        result = ask(
            "Why was H07 rejected?",
            planner=lambda q: [{"name": "experiment_history", "arguments": {"experiment": "H07"}}],
            synthesizer=lambda q, facts: "H07 was evaluated in the historical season.",
        )
        self.assertIn("synthesis_rejected_grounding", result["trace"])
        self.assertIn("H07 parked", result["answer"])

    def test_destructive_request_never_calls_planner(self):
        def forbidden_planner(question):
            raise AssertionError("Planner should not run for destructive intent")
        result = ask("Delete our prediction history and retrain the model", planner=forbidden_planner)
        self.assertEqual(result["evidence"], [])
        self.assertIn("rejected_destructive_intent", result["trace"])

    def test_reject_hallucinated_or_writing_tools(self):
        result = ask("Test", planner=lambda q: [
            {"name": "overwrite_predictions", "arguments": {}},
            {"name": "frozen_gameweek_predictions", "arguments": {"gameweek": -1}},
            {"name": "model_performance", "arguments": {}},
        ])
        self.assertEqual([x["tool"] for x in result["evidence"]], ["model_performance"])


if __name__ == "__main__":
    unittest.main()
