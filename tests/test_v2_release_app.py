"""Production promotion smoke: original app remains intact with a V2 rollback."""
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest
from agent.football_tools import frozen_gameweek_predictions


APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


class ProductionPromotionSmoke(unittest.TestCase):
    def test_production_app_defaults_to_v2_and_can_roll_back(self):
        app = AppTest.from_file(APP, default_timeout=40).run()
        self.assertEqual(len(app.exception), 0, list(app.exception))
        toggles = [t for t in app.get("toggle") if t.label == "V2 facts-first chat"]
        self.assertEqual(len(toggles), 1)
        self.assertTrue(toggles[0].value)
        toggles[0].set_value(False).run()
        self.assertEqual(len(app.exception), 0, list(app.exception))
        self.assertFalse(next(t for t in app.get("toggle")
                              if t.label == "V2 facts-first chat").value)

    def test_production_app_answers_with_verified_tools(self):
        app = AppTest.from_file(APP, default_timeout=40).run()
        self.assertEqual(len(app.exception), 0, list(app.exception))
        app.chat_input[0].set_value(
            "Compare Arsenal and Liverpool over their last 10 Premier League matches"
        ).run()
        self.assertEqual(len(app.exception), 0, list(app.exception))
        self.assertTrue(any(x.get("routing") == "product_v2_verified"
                            for x in app.session_state["debug_info"]))

    def test_frozen_prediction_assets_unchanged_and_available(self):
        result = frozen_gameweek_predictions(8)
        self.assertNotIn("error", result)
        self.assertEqual(len(result["data"]), 10)
        self.assertIn("Model2_v1.0", result["models"])


if __name__ == "__main__":
    unittest.main()
