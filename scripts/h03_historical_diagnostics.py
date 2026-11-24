"""H03 historical diagnostic on the ORIGINAL immutable 1,100-match benchmark.

Diagnostic only. No training, tuning, winner switching, model promotion or
retrospective fitting. This cohort was seen in earlier model development.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE = "reports/model6/model6_walk_forward_predictions.csv"
OUTCOMES = np.array(["H","D","A"])
SOURCE_COLS = {
    "model2": ["Model2HomeProb","Model2DrawProb","Model2AwayProb"],
    "opening_market": ["MarketOpenHomeProb","MarketOpenDrawProb","MarketOpenAwayProb"],
}
EXPECTED_N = 1100
EXPECTED_BASELINE_HITS = 594
EXPECTED_DRAWS = 268


def diagnostics(raw: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    if len(raw) != EXPECTED_N:
        raise ValueError(f"Original H03 cohort must have {EXPECTED_N} fixtures")
    needed = ["Season","Date","HomeTeam","AwayTeam","ActualResult"]
    for cols in SOURCE_COLS.values():
        needed.extend(cols)
    if not set(needed).issubset(raw):
        raise ValueError("Missing original probability/identity/outcome columns")
    keys = ["Date","HomeTeam","AwayTeam"]
    if raw[keys].isna().any().any() or raw.duplicated(keys).any():
        raise ValueError("Historical match identity failure")
    truth = raw["ActualResult"].to_numpy()
    if not np.isin(truth,OUTCOMES).all():
        raise ValueError("Invalid actual H/D/A")
    if (truth=="D").sum()!=EXPECTED_DRAWS:
        raise ValueError("Unexpected original draw count; fail closed")
    model=raw[SOURCE_COLS["model2"]].to_numpy(dtype=float)
    market=raw[SOURCE_COLS["opening_market"]].to_numpy(dtype=float)
    for x in (model,market):
        if not np.isfinite(x).all() or (x<0).any() or not np.allclose(x.sum(axis=1),1,atol=1e-6):
            raise ValueError("Corrupt saved model/market vectors")
    blend=.4*model+.6*market
    pick=OUTCOMES[blend.argmax(axis=1)]
    if int((pick==truth).sum()) != EXPECTED_BASELINE_HITS:
        raise ValueError("Original fixed 40/60 reference is not 594/1100")
    output=[]
    for name,p in [("Model2",model),("OpeningMarket",market),("Fixed40_60",blend)]:
        selected=OUTCOMES[p.argmax(axis=1)]
        actual_draw=truth=="D"
        predicted_draw=selected=="D"
        output.append({
            "Method":name,"Matches":len(truth),
            "Correct":int((selected==truth).sum()),
            "Accuracy":float((selected==truth).mean()),
            "ActualDraws":int(actual_draw.sum()),
            "PredictedDraws":int(predicted_draw.sum()),
            "TrueDrawSelections":int((actual_draw & predicted_draw).sum()),
            "MeanDrawProbability":float(p[:,1].mean()),
            "ActualDrawRate":float(actual_draw.mean()),
            "DrawBrier":float(np.mean((p[:,1]-actual_draw.astype(float))**2)),
            "LogLoss":float(np.mean(-np.log(np.maximum(
                p[np.arange(len(p)),np.array([{"H":0,"D":1,"A":2}[x] for x in truth])],1e-15)))),
        })
    # Descriptive fixed bins. No optimisation or threshold selection.
    frame=raw[["Season","Date","HomeTeam","AwayTeam","ActualResult"]].copy()
    frame["BaselineDrawProbability"]=blend[:,1]
    frame["ActualDraw"]=(truth=="D").astype(int)
    frame["PredictedDraw"]=(pick=="D").astype(int)
    frame["BaselineCorrect"]=(pick==truth).astype(int)
    frame["DrawBin"]=pd.cut(blend[:,1],bins=[-0.001,.15,.20,.25,.30,1.001],
                            labels=["<=15%","15-20%","20-25%","25-30%",">30%"])
    by_bin=frame.groupby("DrawBin",observed=False).agg(
        Matches=("ActualDraw","size"),ActualDraws=("ActualDraw","sum"),
        SelectedDraws=("PredictedDraw","sum"),
        MeanDrawProbability=("BaselineDrawProbability","mean"),
        DrawRate=("ActualDraw","mean"),
        OverallAccuracy=("BaselineCorrect","mean")).reset_index()
    return pd.DataFrame(output),by_bin


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=ROOT/"artifacts/h03")
    args=parser.parse_args()
    summary,bins=diagnostics(pd.read_csv(ROOT/SOURCE))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    summary.to_csv(args.output_dir/"h03_historical_original_1100_baselines.csv",index=False)
    bins.to_csv(args.output_dir/"h03_historical_draw_bins.csv",index=False)
    print("H03 ORIGINAL 1100-MATCH DIAGNOSTIC",flush=True)
    print(summary.to_string(index=False),flush=True)
    print("H03 PREDEFINED DRAW-PROBABILITY BINS",flush=True)
    print(bins.to_string(index=False),flush=True)
    print("Diagnostic only: historical cohort previously used for model development.",flush=True)


if __name__=="__main__":
    main()
