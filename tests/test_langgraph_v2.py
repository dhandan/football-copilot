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
