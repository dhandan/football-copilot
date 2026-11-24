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

    def test_open_comparison_leads_with_points_attack_and_defence(self):
        question = "Compare Arsenal and Liverpool over their last 10 Premier League matches."
        out = self.chat.ask(question, thread_id="overall")
        self.assertTrue(out["complete"], out["trace"])
        rows = {e["arguments"]["team"]: e["result"]["data"][0]
                for e in out["evidence"] if e["tool"] == "form_summary"}
        lead = out["answer"].split("### Verified football evidence")[0]
        self.assertTrue(lead.startswith("### Answer\nOver their last 10 matches"))
        for team in ("Arsenal", "Liverpool"):
            self.assertIn(team, lead)
            self.assertIn(str(int(rows[team]["Points"])), lead)
            self.assertIn(str(int(rows[team]["GoalsFor"])), lead)
            self.assertIn(str(int(rows[team]["GoalsAgainst"])), lead)
        self.assertIn("scored more goals", lead)
        self.assertIn("conceded fewer", lead)
        self.assertEqual(out["planner_requests"], 0)

    def test_specialised_defence_question_keeps_defensive_lead(self):
        out = self.chat.ask(
            "Compare Arsenal and Liverpool's defensive records over their last 10 matches.",
            thread_id="defence-focus")
        self.assertTrue(out["complete"], out["trace"])
        self.assertIn("better goals-conceded record", out["answer"].split(
            "### Verified football evidence")[0])
        self.assertNotIn("Over their last 10 matches, ", out["answer"].split(
            "### Verified football evidence")[0])

    def test_specialised_attack_question_keeps_attacking_lead(self):
        out = self.chat.ask(
            "Compare Arsenal and Liverpool's goals scored over their last 10 matches.",
            thread_id="attack-focus")
        self.assertTrue(out["complete"], out["trace"])
        self.assertIn("scored more goals", out["answer"].split(
            "### Verified football evidence")[0])
        self.assertNotIn("This compares observed results", out["answer"])

    def test_followup_keeps_teams_and_changes_focus(self):
        self.chat.ask("Compare Brighton and Brentford's last seven matches",
                      thread_id="follow")
        out = self.chat.ask(
            "Who has scored more goals across those seven matches?",
            thread_id="follow")
        self.assertEqual(out["route"], "deterministic_investigation")
        self.assertTrue(out["complete"], out["trace"])
        self.assertIn("Attacking verdict", out["answer"])

    def test_defensively_followup_overrides_previous_attack_focus(self):
        self.chat.ask(
            "Compare Arsenal and Liverpool's goals scored over their last 10 matches",
            thread_id="switch-focus")
        out = self.chat.ask("Who has been better defensively over those 10 matches?",
                            thread_id="switch-focus")
        self.assertTrue(out["complete"], out["trace"])
        lead = out["answer"].split("### Verified football evidence")[0]
        self.assertIn("goals-conceded record", lead)
        self.assertNotIn("scored more goals", lead)
        self.assertEqual(out["planner_requests"], 0)

    def test_live_opponent_strength_followup_after_venue_attack(self):
        """Regression for the fourth real user acceptance question."""
        thread = "live-opponent-strength"
        self.chat.ask("Compare Arsenal and Liverpool over their last 10 Premier League matches.", thread_id=thread)
        self.chat.ask("Who has been better defensively over those 10 matches?", thread_id=thread)
        self.chat.ask("How does their home and away scoring compare over those 10 matches?", thread_id=thread)
        question = "Could the strength of their opponents explain the difference in their recent performances?"
        result = self.chat.ask(question, thread_id=thread)
        if not result["complete"]:
            from agent.investigator_v2 import parse_scope
            from scripts.audit_v2_investigator_answers import audit_answer
            from dataclasses import replace
            scope = parse_scope(question, self.chat.investigator.context[thread])
            render_scope = replace(scope, opposition=True)
            raw = self.chat.investigator.ask(question, thread_id="opponent-diagnostic", )
            print("OPPONENT_REGRESSION_DIAGNOSTIC", result["trace"], raw.get("trace"),
                  result.get("evidence"), result.get("answer")[:700], flush=True)
        self.assertEqual(result["route"], "deterministic_investigation", result["trace"])
        self.assertTrue(result["complete"], result["trace"])
        self.assertEqual(result["planner_requests"], 0)
        self.assertIn("Pre-match opponent form", result["answer"])
        self.assertIn("not a validated defensive adjustment or causal finding", result["answer"])

    def test_two_window_defence_has_a_verified_conclusion(self):
        out = self.chat.ask(
            "Has Arsenal's defence improved in its most recent five matches "
            "compared with the previous five?", thread_id="trend")
        self.assertEqual(out["route"], "deterministic_investigation")
        self.assertTrue(out["complete"], out["trace"])
        self.assertIn("five-match trend", out["answer"])
        self.assertIn("the preceding five", out["answer"])

    def test_live_report_home_away_attack_headline(self):
        """Acceptance: exact follow-up must show the actual venue winners."""
        self.chat.ask("Compare Arsenal and Liverpool over their last 10 Premier League matches",
                      thread_id="live-review")
        self.chat.ask("Compare Arsenal and Liverpool's goals scored over their last 10 matches.",
                      thread_id="live-review")
        out = self.chat.ask(
            "How does Arsenal versus Liverpool home and away scoring compare over their last 10 matches?",
            thread_id="live-review")
        self.assertTrue(out["complete"], out["trace"])
        lead = out["answer"].split("### Verified football evidence")[0]
        self.assertIn("Home: Arsenal scored at a higher rate", lead)
        self.assertIn("Away: Liverpool scored at a higher rate", lead)
        # The PR merge checkout can contain a different committed fixture
        # snapshot. Rates must match whichever real rows were executed.
        rows = {e["arguments"]["team"]: e["result"]["data"][0]
                for e in out["evidence"] if e["tool"] == "form_summary"}
        for venue in ("Home", "Away"):
            for team in ("Arsenal", "Liverpool"):
                split = rows[team]["VenueSplit"][venue]
                expected = int(split["GoalsFor"]) / int(split["Played"])
                self.assertIn(f"{team} {expected:.2f}", lead)
        self.assertEqual(out["planner_requests"], 0)

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
