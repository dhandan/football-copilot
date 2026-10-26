"""Answer-quality benchmark for 'which model is performing best so far and why?'

Uses the GW7 prospective scorecard and 23 Oct governance clarification.
Run: python -m unittest tests.test_v2_answer_quality -v
No Ollama is required; candidate answers are scored with explicit criteria.
"""
import re
import unittest

from agent.langgraph_v2 import ask

QUESTION = "Which model is performing the best so far and why?"
SOURCE = "docs/gameweeks/GW07.md"

def assess(answer: str):
    """Return (passed, failures). High-precision regression for this scenario."""
    t = answer.lower()
    checks = {
        "identifies_40_60": bool(re.search(r"40/60", answer)),
        "reports_sample": bool(re.search(r"20\s+(?:prospective\s+)?matches|20\s+fixtures|11/20", t)),
        "accuracies": "55%" in answer and "40%" in answer,
        "proper_scores": all(x in answer for x in ("0.9453", "1.0621", "0.5615", "0.6490")),
        "limits_claim": any(x in t for x in ("small sample", "too small", "limited sample", "insufficient evidence")),
        "provisional_status": any(x in t for x in ("provisional", "pending", "not yet formally", "not yet officially")),
        "does_not_claim_confirmed_promotion": not bool(re.search(
            r"(officially|formally) (promoted|switched|adopted)|confirmed (as |the )?champion", t)),
        "clean_rendering": "canvascanvas" not in t,
        "source": SOURCE.lower() in t or "gw7" in t,
    }
    return all(checks.values()), [k for k, good in checks.items() if not good]


class AnswerQualityTests(unittest.TestCase):
    def test_reference_answer_passes(self):
        reference = (
            "The 40/60 provisional leader is performing best across 20 prospective matches "
            "(11/20, 55% accuracy versus 40% for Model 2). Its Log Loss of 0.9453 "
            "beats 1.0621 and its Brier 0.5615 beats 0.6490. "
            "This is a small sample, not established superiority or formal promotion. "
            "Source: docs/gameweeks/GW07.md."
        )
        self.assertEqual(assess(reference), (True, []))

    def test_actual_user_example_exposes_weaknesses(self):
        response = (
            "Based on GW6-GW7 (20 prospective matches), the 40/60 model is performing better. "
            "Accuracy: 11 out of 20 matches (55%) versus 8/20 (40%). "
            "Log loss 0.9453 vs 1.0621; Brier 0.5615 vs 0.6490. "
            "Stronger recent consistency. canvascanvas Source: docs/gameweeks/GW07.md."
        )
        passed, failures = assess(response)
        self.assertFalse(passed)
        self.assertIn("provisional_status", failures)
        self.assertIn("limits_claim", failures)
        self.assertIn("clean_rendering", failures)

    def test_current_agent_response_is_scored_not_assumed_good(self):
        result = ask(QUESTION, planner=lambda _: [
            {"name": "model_performance", "arguments": {}}
        ])
        passed, failures = assess(result["answer"])
        self.assertFalse(passed)  # Known gap: current answer omits formal status.
        self.assertIn("provisional_status", failures)
        self.assertIn("55%", result["answer"])


if __name__ == "__main__":
    unittest.main()
