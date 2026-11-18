"""Real Streamlit widget acceptance, not just direct backend calls.

Runs headlessly via Streamlit AppTest with Groq deliberately disabled.
Checks that a user can click a question, see verified evidence, then click an
actual follow-up button. No credentials, model requests, browser or Codespace.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "v2_groq_preview.py")
FIRST = "Compare Liverpool and Arsenal over their last 10 matches"


class PrivateV2StreamlitAcceptance(unittest.TestCase):
    def setUp(self):
        self.environ = patch.dict(os.environ, {"FOOTBALL_GROQ_ENABLED": "0"})
        self.environ.start()
        self.addCleanup(self.environ.stop)
        self.app = AppTest.from_file(APP_PATH, default_timeout=30).run()
        self.assertEqual(len(self.app.exception), 0, list(self.app.exception))

    def test_offline_preview_is_not_blocked_by_missing_key(self):
        self.assertTrue(any("Football Copilot V2" in x.value for x in self.app.title))
        self.assertEqual(len(self.app.chat_input), 1)
        self.assertTrue(any("Recent form" == x.label for x in self.app.button))
        self.assertTrue(any("Groq is off" in x.message for x in self.app.info))

    def test_sample_then_actual_followup_button_and_reset(self):
        self.app.button(key="sample_Recent form").click().run()
        self.assertEqual(len(self.app.exception), 0, list(self.app.exception))
        messages = self.app.session_state["v2_private_messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["content"], FIRST)
        self.assertIn("Verified football evidence", messages[1]["content"])
        self.assertIn("0 Groq request", messages[1]["status"])
        self.assertGreaterEqual(len(messages[1]["followups"]), 1)
        suggestion = messages[1]["followups"][0]
        self.assertIn("Liverpool", suggestion)
        self.assertIn("Arsenal", suggestion)
        keys = [b.key for b in self.app.button if b.label == suggestion]
        self.assertEqual(len(keys), 1)
        self.app.button(key=keys[0]).click().run()
        self.assertEqual(len(self.app.exception), 0, list(self.app.exception))
        messages = self.app.session_state["v2_private_messages"]
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[2]["content"], suggestion)
        self.assertIn("Verified football evidence", messages[3]["content"])
        self.assertIn("0 Groq request", messages[3]["status"])
        clear = next(b for b in self.app.button if b.label == "Clear conversation")
        clear.click().run()
        self.assertEqual(len(self.app.exception), 0, list(self.app.exception))
        self.assertEqual(self.app.session_state["v2_private_messages"], [])

    def test_unsupported_question_is_explained_not_hidden(self):
        self.app.chat_input[0].set_value("Why was H07 rejected?").run()
        self.assertEqual(len(self.app.exception), 0, list(self.app.exception))
        messages = self.app.session_state["v2_private_messages"]
        self.assertEqual(len(messages), 2)
        self.assertIn("isn't currently supported", messages[1]["content"])
        self.assertEqual(messages[1]["followups"], [])


if __name__ == "__main__":
    unittest.main()
