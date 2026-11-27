"""Score all four PRE-REGISTERED GW12 probability vectors after final results.

Never regenerate forecasts from outcomes.  Release refuses unmatched fixture
keys, absent verified final scores, future kickoffs, and non-OFFICIAL snapshots.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
KEYS=["FixtureDate","HomeTeam","AwayTeam"]
OUTCOMES={"H":0,"D":1,"A":2}
MODELS={
    "Model2_v1.0":["HomeWinProbability","DrawProbability","AwayWinProbability"],
    "Market_only_GW12":["MarketHomeProbability","MarketDrawProbability","MarketAwayProbability"],
    "Model2_OpeningMarket_40_60_v1.0":["ShadowHomeProbability","ShadowDrawProbability","ShadowAwayProbability"],
    "AdaptiveMarketConfidence_v1.0":["HomeProbability","DrawProbability","AwayProbability"],
}


def validate_and_score(frozen, adaptive, results):
    if len(frozen)!=10 or len(adaptive)!=10 or len(results)!=10:
        raise ValueError("Expected exactly ten fixtures in every table")
    for d in (frozen,adaptive,results):
        if not set(KEYS).issubset(d):
            raise ValueError("Missing fixture identity")
        if d.duplicated(KEYS).any():
            raise ValueError("Duplicate fixture")
    for required in ("ShadowPredictedResult", "SnapshotType", "PredictionTimestamp", "KickoffUTC"):
        if required not in frozen:
            raise ValueError("Missing official frozen provenance")
    if not frozen["SnapshotType"].eq("OFFICIAL").all():
        raise ValueError("Refusing to score test forecasts")
    if not set(["ActualResultCode","ActualHomeGoals","ActualAwayGoals","Status"]).issubset(results):
        raise ValueError("Unverified result input")
    if not results["Status"].eq("FINISHED").all():
        raise ValueError("Incomplete fixtures")
    if results[["ActualHomeGoals","ActualAwayGoals"]].isna().any().any():
        raise ValueError("Missing final scores")
    if not results["ActualResultCode"].isin(OUTCOMES).all():
        raise ValueError("Invalid result outcomes")
    expected=np.where(results["ActualHomeGoals"]>results["ActualAwayGoals"],"H",
             np.where(results["ActualHomeGoals"]<results["ActualAwayGoals"],"A","D"))
    if not np.array_equal(expected,results["ActualResultCode"].to_numpy()):
        raise ValueError("Actual outcome not consistent with final score")
    for label,d in (("adaptive",adaptive),("results",results)):
        check=frozen[KEYS].merge(d[KEYS],on=KEYS,how="outer",indicator=True)
        if len(check)!=10 or not check["_merge"].eq("both").all():
            raise ValueError("Fixture identities mismatch: "+label)
    actual_map=results.set_index(KEYS)["ActualResultCode"]
    actual=np.array([OUTCOMES[actual_map.loc[tuple(row)]] for row in frozen[KEYS].itertuples(index=False,name=None)])
    frames=[]
    for name, cols in MODELS.items():
        source=adaptive.set_index(KEYS).loc[
            pd.MultiIndex.from_frame(frozen[KEYS])].reset_index() if name.startswith("Adaptive") else frozen
        if not set(cols).issubset(source):
            raise ValueError("Missing probability columns: "+name)
        probabilities=source[cols].to_numpy(dtype=float)
        if name=="Model2_v1.0":
            probabilities=probabilities/100
            probabilities=probabilities/probabilities.sum(axis=1,keepdims=True)
        if not (np.isfinite(probabilities).all() and (probabilities>=0).all()
                and np.allclose(probabilities.sum(axis=1),1,atol=1e-6)):
            raise ValueError("Invalid probability vector: "+name)
        pred=probabilities.argmax(axis=1)
        correct=(pred==actual).astype(int)
        actual_prob=probabilities[np.arange(10),actual]
        loss=-np.log(np.maximum(actual_prob,1e-15))
        target=np.eye(3)[actual]
        brier=np.square(probabilities-target).sum(axis=1)
        d=frozen[KEYS].copy()
        d["Approach"]=name
        d[["HomeProbability","DrawProbability","AwayProbability"]]=probabilities
        d["ActualResultCode"]=np.array(["H","D","A"])[actual]
        d["PredictedResultCode"]=np.array(["H","D","A"])[pred]
        d["Correct"]=correct
        d["LogLoss"]=loss
        d["MulticlassBrier"]=brier
        # Pre-registered diagnostic strata, derived only AFTER final results.
        # These fields do not alter any frozen forecast probability.
        market_pick = np.array(["H","D","A"])[
            frozen[["MarketHomeProbability","MarketDrawProbability",
                    "MarketAwayProbability"]].to_numpy(dtype=float).argmax(axis=1)]
        d["DisagreesWithMarket"] = d["PredictedResultCode"].ne(market_pick).astype(int)
        d["ColdStartUsed"] = (frozen["ColdStartUsed"].astype(str).str.lower()
                              .isin(("true","1","yes")).to_numpy()
                              if "ColdStartUsed" in frozen else False)
        d["ActualIsDraw"] = (actual == OUTCOMES["D"]).astype(int)
        draw_prob = probabilities[:, OUTCOMES["D"]]
        d["BinaryDrawBrier"] = np.square(draw_prob - d["ActualIsDraw"].to_numpy())
        d["BinaryDrawLogLoss"] = -(d["ActualIsDraw"].to_numpy()*np.log(np.maximum(draw_prob,1e-15))
                              + (1-d["ActualIsDraw"].to_numpy())*np.log(np.maximum(1-draw_prob,1e-15)))
        d["HighestProbability"] = probabilities.max(axis=1)
        d["ConfidenceBucket"] = pd.cut(
            d["HighestProbability"], bins=[0,.4,.6,1.000001],
            labels=["low <=40%","medium 40-60%","high >60%"],
            include_lowest=True)
        if name=="Model2_v1.0":
            if {"MostLikelyScore","ExpectedHomeGoals","ExpectedAwayGoals"}.issubset(frozen):
                got=(results.set_index(KEYS).loc[pd.MultiIndex.from_frame(frozen[KEYS])])
                actual_scores=(got["ActualHomeGoals"].astype(int).astype(str) + "-" +
                               got["ActualAwayGoals"].astype(int).astype(str)).to_numpy()
                d["ExactScoreCorrect"] = (frozen["MostLikelyScore"].astype(str).to_numpy()==actual_scores).astype(int)
                d["HomeGoalMAE"] = np.abs(frozen["ExpectedHomeGoals"].to_numpy(dtype=float)-got["ActualHomeGoals"].to_numpy(dtype=float))
                d["AwayGoalMAE"] = np.abs(frozen["ExpectedAwayGoals"].to_numpy(dtype=float)-got["ActualAwayGoals"].to_numpy(dtype=float))
        frames.append(d)
    scores=pd.concat(frames,ignore_index=True)
    grouped=scores.groupby("Approach",sort=False).agg(
        Matches=("Correct","size"),Correct=("Correct","sum"),
        Accuracy=("Correct","mean"),LogLoss=("LogLoss","mean"),
        MulticlassBrier=("MulticlassBrier","mean"),
        PredictedDraws=("PredictedResultCode",lambda x: int(x.eq("D").sum())),
        ActualDraws=("ActualIsDraw","sum"),
        MeanDrawProbability=("DrawProbability","mean"),
        DrawBrier=("BinaryDrawBrier","mean"),
        DrawLogLoss=("BinaryDrawLogLoss","mean"),
        MarketDisagreements=("DisagreesWithMarket","sum"),
        ColdStartFixtures=("ColdStartUsed","sum")).reset_index()
    # Small-sample calibration diagnostics; not a trained calibration model.
    diagnostics=[]
    for approach, data in scores.groupby("Approach",sort=False):
        pred_draw=data["DrawProbability"].to_numpy(dtype=float)
        actual_draw=data["ActualIsDraw"].to_numpy(dtype=float)
        positives=pred_draw[actual_draw==1]
        negatives=pred_draw[actual_draw==0]
        bins=data.groupby("ConfidenceBucket",observed=True).agg(
            n=("Correct","size"),accuracy=("Correct","mean"),
            confidence=("HighestProbability","mean"))
        ece=float((bins["n"]*abs(bins["confidence"]-bins["accuracy"])).sum()/len(data))
        disagree=data.loc[data["DisagreesWithMarket"].eq(1)]
        cold=data.loc[data["ColdStartUsed"].eq(True)]
        diagnostics.append(dict(Approach=approach,
            DrawCalibrationGap=abs(float(pred_draw.mean()-actual_draw.mean())),
            DrawDiscrimination=float(positives.mean()-negatives.mean())
                if len(positives) and len(negatives) else np.nan,
            ConfidenceECE_3Bins=ece,
            MarketDisagreementAccuracy=float(disagree["Correct"].mean()) if len(disagree) else np.nan,
            ColdStartAccuracy=float(cold["Correct"].mean()) if len(cold) else np.nan,
            ExactScoreHits=int(data["ExactScoreCorrect"].sum())
                if "ExactScoreCorrect" in data and approach=="Model2_v1.0" else np.nan,
            HomeGoalMAE=float(data["HomeGoalMAE"].mean())
                if "HomeGoalMAE" in data and approach=="Model2_v1.0" else np.nan,
            AwayGoalMAE=float(data["AwayGoalMAE"].mean())
                if "AwayGoalMAE" in data and approach=="Model2_v1.0" else np.nan))
    grouped=grouped.merge(pd.DataFrame(diagnostics),on="Approach",validate="one_to_one")
    grouped.insert(0,"Gameweek",12)
    if len(grouped)!=4 or not grouped["Matches"].eq(10).all():
        raise RuntimeError("Four-way GW12 summary incomplete")
    return scores, grouped


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--gameweek",type=int,default=12)
    args=parser.parse_args()
    if args.gameweek!=12:
        raise ValueError("Only GW12 supported")
    d=ROOT/"data/live"
    prefix="2026_27_gw12"
    frozen=pd.read_csv(d/"challenger_predictions"/f"{prefix}_market_shadow_predictions.csv")
    adaptive=pd.read_csv(d/"challenger_predictions"/f"{prefix}_adaptive_confidence_v1.csv")
    results=pd.read_csv(d/"results"/f"{prefix}_results.csv")
    scores, summary=validate_and_score(frozen,adaptive,results)
    folder=d/"evaluations"
    output1=folder/f"{prefix}_four_way_fixture_scores.csv"
    output2=folder/f"{prefix}_four_way_summary.csv"
    if output1.exists() or output2.exists():
        raise FileExistsError("GW12 evaluation exists; immutable comparison")
    with output1.open("x",encoding="utf-8") as f:
        scores.to_csv(f,index=False)
    with output2.open("x",encoding="utf-8") as f:
        summary.to_csv(f,index=False)
    print(summary.to_string(index=False))
    print("GW12 FOUR-WAY POST-MATCH EVALUATION RECORDED; no model promoted")


if __name__=="__main__":
    main()
