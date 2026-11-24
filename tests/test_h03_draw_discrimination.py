"""H03 draw-aware research guardrails and pipeline contracts."""
import unittest
import numpy as np
import pandas as pd
from scripts.h03_draw_discrimination import (
    ROOT, BASE, KEY, VERSION, adjust_draws, load_development, load_holdout, run, score,
)


class H03DrawDiscriminationTests(unittest.TestCase):
    def test_neutral_rule_preserves_probability_mass_and_win_ratio(self):
        base=np.array([[.40,.25,.35],[.70,.20,.10],[.25,.40,.35]])
        new,signal=adjust_draws(base,np.array([1.1,2.5,.8]),np.array([1.0,.7,.9]))
        np.testing.assert_allclose(new.sum(axis=1),1.0,atol=1e-12)
        np.testing.assert_allclose(new[:,0]/new[:,2],base[:,0]/base[:,2],atol=1e-12)
        self.assertTrue(np.isfinite(signal).all())
        self.assertTrue((new>=0).all())
        self.assertTrue((new<=1).all())

    def test_more_balanced_lower_total_xg_means_higher_draw_probability(self):
        p=np.tile(np.array([.40,.25,.35]),(3,1))
        new,_=adjust_draws(p,np.array([1.0,2.1,1.0]),np.array([1.0,.5,3.0]))
        self.assertGreater(new[0,1],new[1,1])
        self.assertGreater(new[0,1],new[2,1])

    def test_bad_probabilities_and_missing_goals_rejected(self):
        with self.assertRaises(ValueError):
            adjust_draws(np.array([[.4,.2,.2]]),np.array([1]),np.array([1]))
        with self.assertRaises(ValueError):
            adjust_draws(np.array([[.4,.3,.3]]),np.array([float("nan")]),np.array([1]))

    def test_dev_and_heldout_are_strict_and_disjoint(self):
        dev=load_development(ROOT)
        test=load_holdout(ROOT)
        self.assertEqual(len(dev),50)
        self.assertEqual(len(test),60)
        self.assertEqual(set(dev.Gameweek),set(range(1,6)))
        self.assertEqual(set(test.Gameweek),set(range(6,12)))
        self.assertTrue(set(map(tuple,dev[KEY].to_numpy())).isdisjoint(
            set(map(tuple,test[KEY].to_numpy()))))
        self.assertTrue(dev["Provenance"].eq("retrospective_opening_odds_comparison").all())
        self.assertTrue(test["Provenance"].eq("official_frozen_prematch").all())

    def test_scores_derived_from_unmodified_source_vectors(self):
        fixtures,summary=run(ROOT)
        self.assertEqual(len(fixtures),220)
        self.assertEqual(len(summary),4)
        self.assertEqual(set(summary.Approach),{"Fixed_40_60",VERSION})
        self.assertTrue(summary.Matches.sort_values().to_list()==[50,50,60,60])
        self.assertTrue(fixtures["Correct"].isin([0,1]).all())
        self.assertFalse((fixtures["DrawProbability"]<0).any())
        self.assertFalse((fixtures["DrawProbability"]>1).any())

    def test_original_historical_cohort_reproduces_594_of_1100(self):
        from scripts.h03_historical_diagnostics import diagnostics, SOURCE
        summary, bins = diagnostics(pd.read_csv(ROOT/SOURCE))
        fixed = summary.loc[summary["Method"].eq("Fixed40_60")].iloc[0]
        self.assertEqual(int(fixed["Matches"]), 1100)
        self.assertEqual(int(fixed["Correct"]), 594)
        self.assertEqual(int(fixed["ActualDraws"]), 268)
        self.assertEqual(int(bins["Matches"].sum()), 1100)
        self.assertEqual(int(bins["ActualDraws"].sum()), 268)

    def test_outcomes_not_an_input_to_probability_transform(self):
        df=load_development(ROOT).copy()
        scores_a,_=score(df,"dev")
        df["Actual"]=df["Actual"].map({"H":"A","A":"D","D":"H"})
        scores_b,_=score(df,"dev")
        a=scores_a.loc[scores_a.Approach.eq(VERSION),"DrawProbability"].to_numpy()
        b=scores_b.loc[scores_b.Approach.eq(VERSION),"DrawProbability"].to_numpy()
        np.testing.assert_array_equal(a,b)


if __name__=="__main__":
    unittest.main()
