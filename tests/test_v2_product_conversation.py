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
        self.assertIn("five-match trend", out["answer"])
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

    def test_product_surfaces_clickable_followups_without_groq(self):
        out = self.chat.ask(
            "Compare Brighton and Brentford's last seven matches",
            thread_id="suggestions")
        self.assertTrue(out["complete"], out["trace"])
        self.assertEqual(len(out["followups"]), 2)
        self.assertTrue(all("Brighton" in q and "Brentford" in q
                            for q in out["followups"]))
        next_out = self.chat.ask(out["followups"][0],
                                 thread_id="suggestions")
        self.assertEqual(next_out["route"], "deterministic_investigation")
        self.assertTrue(next_out["complete"], next_out["trace"])
        self.assertEqual(next_out["planner_requests"], 0)

    def test_followup_selection_reuses_conversation_scope(self):
        first = self.chat.ask(
            "Compare Brighton and Brentford's last seven matches",
            thread_id="followup-venue")
        query = first["followups"][1]
        out = self.chat.ask(query, thread_id="followup-venue")
        self.assertTrue(out["complete"], out["trace"])
        self.assertIn("Home", out["answer"])
        self.assertIn("Away", out["answer"])
        self.assertEqual(out["planner_requests"], 0)

    def test_one_team_trend_followup_uses_verified_evidence(self):
        first = self.chat.ask(
            "Has Arsenal's defence improved in its most recent five matches "
            "compared with the previous five?", thread_id="team-trend")
        self.assertTrue(first["complete"])
        self.assertEqual(len(first["followups"]), 1)
        later = self.chat.ask(first["followups"][0],
                              thread_id="team-trend")
        self.assertTrue(later["complete"], later["trace"])
        self.assertIn("goals scored", later["answer"])
        self.assertEqual(later["planner_requests"], 0)

    def test_supported_why_followup_has_conditional_context(self):
        first = self.chat.ask(
            "Compare Brighton and Brentford's last seven matches",
            thread_id="why-chain")
        self.assertTrue(first["complete"])
        why = self.chat.ask("Why is that?", thread_id="why-chain")
        self.assertEqual(why["route"], "deterministic_investigation")
        self.assertTrue(why["complete"], why["trace"])
        self.assertIn("Interpretation limit", why["answer"])
        self.assertEqual(why["planner_requests"], 0)

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
