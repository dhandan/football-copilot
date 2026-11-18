"""Opponent strength must use only earlier results and expose sparse coverage."""
import unittest
from agent.opponent_strength import rate_opponents

def fixture(date, team, opponent, points, gf, ga):
    return {"date": date, "team": team, "opponent": opponent,
            "points": points, "gf": gf, "ga": ga, "venue": "Away"}

class OpponentStrengthTests(unittest.TestCase):
    def test_pre_match_excludes_same_day_and_future_results(self):
        rows = [
            fixture("2026-08-01", "Chelsea", "A", 0, 0, 1),
            fixture("2026-08-08", "Chelsea", "B", 3, 2, 0),
            fixture("2026-08-15", "Chelsea", "C", 3, 4, 0),
            fixture("2026-08-22", "Chelsea", "D", 3, 5, 0),
            fixture("2026-08-22", "Arsenal", "Chelsea", 3, 1, 0),
            fixture("2026-08-29", "Chelsea", "E", 3, 6, 0),
            fixture("2026-08-22", "Liverpool", "West Ham", 1, 1, 1),
        ]
        data = rate_opponents(rows, ("Arsenal", "Liverpool"), 1)
        arsenal = data["Arsenal"]["fixtures"][0]
        self.assertEqual(arsenal["prior_games"], 3)
        self.assertAlmostEqual(arsenal["opponent_prior_ppg"], 2.0)
        self.assertAlmostEqual(arsenal["opponent_prior_gf_pg"], 2.0)
        self.assertIsNone(data["Liverpool"]["fixtures"][0]["opponent_prior_ppg"])
        self.assertEqual(data["Liverpool"]["rated"], 0)

    def test_all_future_results_cannot_change_existing_rating(self):
        base = [
            fixture("2026-08-01", "Chelsea", "A", 3, 2, 0),
            fixture("2026-08-08", "Chelsea", "B", 3, 2, 0),
            fixture("2026-08-15", "Chelsea", "C", 0, 0, 1),
            fixture("2026-08-22", "Arsenal", "Chelsea", 3, 1, 0),
        ]
        before = rate_opponents(base, ("Arsenal", "Liverpool"), 1)["Arsenal"]["fixtures"][0]
        after = rate_opponents(base + [
            fixture("2026-08-29", "Chelsea", "D", 0, 0, 8)
        ], ("Arsenal", "Liverpool"), 1)["Arsenal"]["fixtures"][0]
        self.assertEqual(before["opponent_prior_ppg"], after["opponent_prior_ppg"])
        self.assertEqual(before["opponent_prior_gf_pg"], after["opponent_prior_gf_pg"])
