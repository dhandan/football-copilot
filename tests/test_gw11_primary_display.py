"""GW11 release gate: displayed 1X2 primary must be the frozen fixed blend."""
import unittest
from pathlib import Path
import pandas as pd
from app.live_predictions import display_provisional_primary


ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT/"data/live/predictions/2026_27_gw11_predictions.csv"
BLEND=ROOT/"data/live/challenger_predictions/2026_27_gw11_market_shadow_predictions.csv"


class GW11PrimaryDisplayTests(unittest.TestCase):
    def test_gw11_default_1x2_from_saved_40_60(self):
        model=pd.read_csv(BASE)
        frozen=pd.read_csv(BLEND)
        displayed, using_primary=display_provisional_primary(model,"2026/27",11)
        self.assertTrue(using_primary)
        self.assertEqual(len(displayed),10)
        keys=["FixtureDate","FixtureTime","HomeTeam","AwayTeam"]
        compare=displayed.merge(frozen[keys+[
            "ShadowHomeProbability","ShadowDrawProbability",
            "ShadowAwayProbability","ShadowPredictedResult"
        ]],on=keys,validate="one_to_one")
        for display_col, frozen_col in zip(
            ["HomeWinProbability","DrawProbability","AwayWinProbability"],
            ["ShadowHomeProbability","ShadowDrawProbability","ShadowAwayProbability"]):
            self.assertTrue(
                ((compare[display_col] - compare[frozen_col]*100).abs() < .000001).all()
            )
        self.assertTrue((displayed["MostLikelyScore"]==model["MostLikelyScore"]).all())
        self.assertTrue((displayed["Model2PredictedResult"]==model["PredictedResult"]).all())
        picks=[]
        for _, row in compare.iterrows():
            picks.append(row["HomeTeam"] if row["ShadowPredictedResult"]=="H" else
                         row["AwayTeam"] if row["ShadowPredictedResult"]=="A" else "Draw")
        self.assertEqual(compare["PredictedResult"].tolist(),picks)

    def test_reject_fixture_mismatch_without_substituting(self):
        model=pd.read_csv(BASE)
        model.loc[0,"HomeTeam"]="Not a team in freeze"
        with self.assertRaisesRegex(ValueError,"fixture mismatch"):
            display_provisional_primary(model,"2026/27",11)

    def test_old_gameweek_keeps_model2(self):
        model=pd.read_csv(BASE)
        displayed, using_primary=display_provisional_primary(model,"2026/27",7)
        self.assertFalse(using_primary)
        self.assertTrue(displayed.equals(model))


if __name__=="__main__":
    unittest.main()
