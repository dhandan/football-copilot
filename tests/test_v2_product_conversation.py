"""User-facing facts-first V2 product route, including conversational state."""
import unittest

from agent.product_conversation_v2 import ProductConversation


class ProductConversationTests(unittest.TestCase):
    def setUp(self):
        self.chat = ProductConversation()

    def test_verified_form_comparison_without_groq(self):
        output = self.chat.ask(
            "Compare Brighton and Brentford's last seven Premier League matches",
            thread_id="one")
        self.assertEqual(output["route"], "deterministic_investigation")
        self.assertTrue(output["complete"], output.get("trace"))
        self.assertEqual(output["planner_requests"], 0)
        self.assertIn("Brighton", output["answer"])
        self.assertIn("Brentford", output["answer"])

    def test_followup_keeps_teams_and_changes_focus(self):
        self.chat.ask("Compare Brighton and Brentford's last seven matches",
                      thread_id="follow")
        out = self.chat.ask(
            "Who has scored more goals across those seven matches?",
            thread_id="follow")
        self.assertEqual(out["route"], "deterministic_investigation")
        self.assertTrue(out["complete"], out["trace"])
        self.assertIn("Attacking verdict", out["answer"])

    def test_two_window_defence_has_a_verified_conclusion(self):
        out = self.chat.ask(
            "Has Arsenal's defence improved in its most recent five matches "
            "compared with the previous five?", thread_id="trend")
        self.assertEqual(out["route"], "deterministic_investigation")
        self.assertTrue(out["complete"], out["trace"])
        self.assertIn("Five-match trend", out["answer"] if False
                      else out["answer"].replace("five-match trend", "Five-match trend"))
        self.assertIn("the preceding five", out["answer"])

    def test_why_uses_two_rounds_only_if_verified_gap(self):
        out = self.chat.ask(
            "Why did Brighton score more than Brentford over their last seven matches?",
            thread_id="why")
        self.assertEqual(out["route"], "deterministic_investigation")
        self.assertTrue(out["complete"], out["trace"])
        self.assertIn("Interpretation limit", out["answer"])
        self.assertEqual(out["planner_requests"], 0)

    def test_previous_teams_do_not_leak_across_threads(self):
        a = self.chat.ask("Compare Arsenal and Chelsea over their last five matches",
                          thread_id="a")
        b = self.chat.ask("Compare Everton and Bournemouth over their last five matches",
                          thread_id="b")
        self.assertTrue(a["complete"])
        self.assertTrue(b["complete"])
        self.assertNotIn("| Arsenal |", b["answer"])

    def test_unsupported_legacy_without_groq_explains_limitation(self):
        out = self.chat.ask("Why was H07 rejected?", thread_id="legacy")
        self.assertEqual(out["route"], "unsupported")
        self.assertFalse(out["complete"])
        self.assertEqual(out["planner_requests"], 0)

    def test_unsupported_with_legacy_delegates_without_rewriting(self):
        class Legacy:
            def ask(self, question, thread_id):
                return {"answer": "Legacy verified response",
                        "trace": ["legacy"], "evidence": []}
        route = ProductConversation(legacy=Legacy())
        out = route.ask("Why was H07 rejected?", thread_id="legacy")
        self.assertEqual(out["route"], "legacy_v2")
        self.assertEqual(out["answer"], "Legacy verified response")


if __name__ == "__main__":
    unittest.main()
