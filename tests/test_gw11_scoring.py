"""Tests for prospectively scored GW11 four-way experiment."""
import unittest
import numpy as np
import pandas as pd

from scripts.freeze_gw11_adaptive import build
from scripts.score_gw11_four_way import validate_and_score, MODELS
from test_gw11_adaptive_freeze import samples


class GW11ScoringTests(unittest.TestCase):
    def setUp(self):
        model, market = samples()
        self.adaptive = build(model, market, 11)
        self.fixed = model.merge(market, on=["FixtureDate","FixtureTime","HomeTeam","AwayTeam"])
        p = self.fixed[["HomeWinProbability","DrawProbability","AwayWinProbability"]].to_numpy()/100
        k = self.fixed[["MarketHomeProbability","MarketDrawProbability","MarketAwayProbability"]].to_numpy()
        z = .4*p+.6*k
        self.fixed[["ShadowHomeProbability","ShadowDrawProbability","ShadowAwayProbability"]] = z
        self.fixed["ShadowPredictedResult"] = "H"
        self.fixed["SnapshotType"] = "OFFICIAL"
        self.fixed["KickoffUTC"] = "2026-11-21T12:30:00+00:00"
        self.results=self.fixed[["FixtureDate","HomeTeam","AwayTeam"]].copy()
        self.results["Status"]="FINISHED"
        self.results["ActualHomeGoals"]=2
        self.results["ActualAwayGoals"]=1
        self.results["ActualResultCode"]="H"

    def test_four_way_scores_reconcile(self):
        scores,summary=validate_and_score(self.fixed,self.adaptive,self.results)
        self.assertEqual(len(scores),40)
        self.assertEqual(set(summary["Approach"]),set(MODELS))
        self.assertTrue(summary["Matches"].eq(10).all())
        self.assertTrue(summary["Correct"].eq(10).all())
        self.assertTrue(summary["Accuracy"].eq(1.0).all())
        self.assertTrue(summary["LogLoss"].between(0,2).all())
        self.assertTrue(summary["MulticlassBrier"].between(0,2).all())

    def test_refuses_inconsistent_final_result(self):
        self.results.loc[0,"ActualResultCode"]="D"
        with self.assertRaisesRegex(ValueError, "not consistent"):
            validate_and_score(self.fixed,self.adaptive,self.results)

    def test_refuses_unfinished_fixture(self):
        self.results.loc[0,"Status"]="SCHEDULED"
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_and_score(self.fixed,self.adaptive,self.results)

    def test_refuses_unmatched_adaptive(self):
        self.adaptive.loc[0,"AwayTeam"]="Different"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_and_score(self.fixed,self.adaptive,self.results)

    def test_refuses_nonofficial_forecasts(self):
        self.fixed["SnapshotType"]="TEST"
        with self.assertRaisesRegex(ValueError, "test forecasts"):
            validate_and_score(self.fixed,self.adaptive,self.results)


if __name__=="__main__":
    unittest.main()
