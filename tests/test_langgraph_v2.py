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

    def test_reject_hallucinated_or_writing_tools(self):
        result = ask("Test", planner=lambda q: [
            {"name": "overwrite_predictions", "arguments": {}},
            {"name": "frozen_gameweek_predictions", "arguments": {"gameweek": -1}},
            {"name": "model_performance", "arguments": {}},
        ])
        self.assertEqual([x["tool"] for x in result["evidence"]], ["model_performance"])


if __name__ == "__main__":
    unittest.main()
