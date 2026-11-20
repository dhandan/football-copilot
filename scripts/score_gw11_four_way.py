"""Score all four PRE-REGISTERED GW11 probability vectors after final results.

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
    "Market_only_GW11":["MarketHomeProbability","MarketDrawProbability","MarketAwayProbability"],
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
        frames.append(d)
    scores=pd.concat(frames,ignore_index=True)
    grouped=scores.groupby("Approach",sort=False).agg(
        Matches=("Correct","size"),Correct=("Correct","sum"),
        Accuracy=("Correct","mean"),LogLoss=("LogLoss","mean"),
        MulticlassBrier=("MulticlassBrier","mean")).reset_index()
    grouped.insert(0,"Gameweek",11)
    if len(grouped)!=4 or not grouped["Matches"].eq(10).all():
        raise RuntimeError("Four-way GW11 summary incomplete")
    return scores, grouped


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--gameweek",type=int,default=11)
    args=parser.parse_args()
    if args.gameweek!=11:
        raise ValueError("Only GW11 supported")
    d=ROOT/"data/live"
    prefix="2026_27_gw11"
    frozen=pd.read_csv(d/"challenger_predictions"/f"{prefix}_market_shadow_predictions.csv")
    adaptive=pd.read_csv(d/"challenger_predictions"/f"{prefix}_adaptive_confidence_v1.csv")
    results=pd.read_csv(d/"results"/f"{prefix}_results.csv")
    scores, summary=validate_and_score(frozen,adaptive,results)
    folder=d/"evaluations"
    output1=folder/f"{prefix}_four_way_fixture_scores.csv"
    output2=folder/f"{prefix}_four_way_summary.csv"
    if output1.exists() or output2.exists():
        raise FileExistsError("GW11 evaluation exists; immutable comparison")
    with output1.open("x",encoding="utf-8") as f:
        scores.to_csv(f,index=False)
    with output2.open("x",encoding="utf-8") as f:
        summary.to_csv(f,index=False)
    print(summary.to_string(index=False))
    print("GW11 FOUR-WAY POST-MATCH EVALUATION RECORDED; no model promoted")


if __name__=="__main__":
    main()
