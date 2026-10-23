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
                return ("GW8 frozen forecasts are not yet evaluated. "
                        "GW6-GW7 champion performance is 55% versus 40% for Model 2.")
            return "GW8 frozen forecasts: 10 fixtures, saved and not yet evaluated."
        conversation = Conversation(planner=planner, synthesizer=synthesizer)
        conversation.ask("Show GW8 forecasts", thread_id="followup")
        result = conversation.ask("Compare with champion performance", thread_id="followup")
        self.assertIn("Previous turn", captured[-1])
        self.assertIn("GW8 frozen forecasts", captured[-1])
        self.assertIn("synthesis_accepted", result["trace"])
        self.assertIn("55%", result["answer"])

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
                "GW8 contains 10 frozen forecasts. These are not yet evaluated. "
                "GW6-GW7 performance: champion 55%, Model 2 40%."
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
