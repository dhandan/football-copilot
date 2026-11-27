"""GW12 four-way forecast integrity controls. No remote calls or outcomes."""
import unittest
import numpy as np
import pandas as pd

from scripts.freeze_gw12_adaptive import adaptive_probabilities, build

KEYS = [
    ("2026-11-28", "12:30", "Man City", "Fulham"),
    ("2026-11-28", "15:00", "Newcastle", "Arsenal"),
] + [(f"2026-11-{28+i//3:02d}", "15:00", f"Home{i}", f"Away{i}") for i in range(8)]


def samples():
    base = pd.DataFrame(KEYS, columns=["FixtureDate","FixtureTime","HomeTeam","AwayTeam"])
    model = base.assign(Gameweek=12, PredictionStatus="Predicted",
                        HomeWinProbability=45., DrawProbability=25., AwayWinProbability=30.,
                        PredictionTimestamp="2026-11-27T08:50:00+00:00")
    market = base.assign(MarketHomeProbability=.40, MarketDrawProbability=.27,
                         MarketAwayProbability=.33, SnapshotType="OFFICIAL",
                         SnapshotTimestamp="2026-11-27T08:52:00+00:00")
    return model, market


class GW12AdaptiveFreezeTests(unittest.TestCase):
    def test_registered_adaptive_rule_exact(self):
        m = np.array([[.50,.25,.25],[.30,.25,.45]])
        k = np.array([[.60,.20,.20],[.40,.27,.33]])
        z, mw, kw = adaptive_probabilities(m,k)
        self.assertAlmostEqual(kw[0], .54)
        self.assertAlmostEqual(kw[1], .66)
        np.testing.assert_allclose(z[0], mw[0]*m[0]+kw[0]*k[0])
        np.testing.assert_allclose(z.sum(axis=1), [1,1])

    def test_ten_match_freeze_matches_model_and_market(self):
        model, market = samples()
        out = build(model,market,12)
        self.assertEqual(len(out), 10)
        self.assertTrue(out["ForecastRole"].eq("ProspectiveChallenger").all())
        self.assertTrue(out["ExperimentVersion"].eq("AdaptiveMarketConfidence_v1.0").all())
        self.assertTrue(out["PredictedResult"].eq("H").all())
        np.testing.assert_allclose(out[["HomeProbability","DrawProbability","AwayProbability"]].sum(axis=1), 1)

    def test_reject_unmatched_fixture(self):
        model, market = samples()
        market.loc[0,"AwayTeam"]="Other"
        with self.assertRaisesRegex(ValueError, "do not match"):
            build(model, market,12)

    def test_reject_duplicates(self):
        model, market = samples()
        market.loc[1,["FixtureDate","FixtureTime","HomeTeam","AwayTeam"]] = market.loc[0,["FixtureDate","FixtureTime","HomeTeam","AwayTeam"]].to_numpy()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build(model,market,12)

    def test_reject_invalid_probability(self):
        model, market = samples()
        market.loc[0,"MarketHomeProbability"]=.9
        with self.assertRaisesRegex(ValueError, "sum to one"):
            build(model,market,12)

    def test_reject_wrong_week_or_non_official_market(self):
        model, market = samples()
        model["Gameweek"]=10
        with self.assertRaisesRegex(ValueError, "Wrong gameweek"):
            build(model,market,12)
        model["Gameweek"]=11
        market["SnapshotType"]="TEST"
        with self.assertRaisesRegex(ValueError, "Only official"):
            build(model,market,12)

    def test_champion_remains_fixed_40_60_not_adaptive(self):
        model, market = samples()
        h = .4 * (.45) + .6 * (.40)
        d = .4 * (.25) + .6 * (.27)
        a = .4 * (.30) + .6 * (.33)
        model["ShadowHomeProbability"]=h
        model["ShadowDrawProbability"]=d
        model["ShadowAwayProbability"]=a
        out = build(model,market,12)
        self.assertEqual(len(out),10)
        model.loc[0,"ShadowHomeProbability"]=.90
        with self.assertRaisesRegex(ValueError,"do not reconcile"):
            build(model,market,12)


if __name__ == "__main__":
    unittest.main()
