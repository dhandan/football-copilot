"""H12: fixed Asian handicap / totals pre-match signals for football 1X2.
Bounded 3-variant chronological screen; no tuning against test seasons.
All fixtures in existing 1100-match benchmark retained.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"reports/h12_cross_market"
OUT.mkdir(parents=True,exist_ok=True)
YEARS={"2122":"2021/22","2223":"2022/23","2324":"2023/24","2425":"2024/25","2526":"2025/26"}
KEY=["Date","HomeTeam","AwayTeam"]
LABEL={"H":0,"D":1,"A":2}
TEST=["2023/24","2024/25","2025/26"]
def load():
    tables=[]
    for suffix,season in YEARS.items():
        r=pd.read_csv(ROOT/f"data/raw/E0_{suffix}.csv",encoding="utf-8-sig",low_memory=False)
        r["Season"]=season
        r["Date"]=pd.to_datetime(r.Date,dayfirst=True).dt.strftime("%Y-%m-%d")
        tables.append(r)
    raw=pd.concat(tables,ignore_index=True)
    assert len(raw)==1900
    ref=pd.read_csv(ROOT/"reports/model6/model6_walk_forward_predictions.csv")
    ref.Date=pd.to_datetime(ref.Date).dt.strftime("%Y-%m-%d")
    assert len(ref)==1100
    ref["y"]=ref.ActualResult.map(LABEL)
    # ensure raw source is unique and exact matching.
    assert not raw.duplicated(KEY).any()
    assert raw[KEY].notna().all().all()
    joined=ref.merge(raw,on=KEY,how="left",suffixes=("","_raw"),validate="one_to_one",indicator=True)
    assert len(joined)==1100 and joined._merge.eq("both").all()
    assert joined.Season.eq(joined.Season_raw).all()
    p2=joined[["Model2HomeProb","Model2DrawProb","Model2AwayProb"]].to_numpy(float)
    market=joined[["MarketOpenHomeProb","MarketOpenDrawProb","MarketOpenAwayProb"]].to_numpy(float)
    baseline=.4*p2+.6*market
    assert int((baseline.argmax(1)==joined.y.to_numpy()).sum())==594
    return raw,joined,p2,market,baseline

def no_vig(frame,names):
    d=frame[names].apply(pd.to_numeric,errors="coerce")
    inv=1/d
    inv=inv.where(d>1)
    inv=inv.div(inv.sum(axis=1),axis=0)
    return inv

def prepare(frame):
    df=frame.copy()
    # Under/over market pre-match consensus and Asian handicap, no closing lines.
    df["p_over25"]=no_vig(df,["Avg>2.5","Avg<2.5"]).iloc[:,0]
    df["p_asian_home"]=no_vig(df,["AvgAHH","AvgAHA"]).iloc[:,0]
    df["ah_line"]=pd.to_numeric(df["AHh"],errors="coerce")
    df["p_open_home"]=no_vig(df,["AvgH","AvgD","AvgA"]).iloc[:,0]
    df["p_open_draw"]=no_vig(df,["AvgH","AvgD","AvgA"]).iloc[:,1]
    df["p_open_away"]=no_vig(df,["AvgH","AvgD","AvgA"]).iloc[:,2]
    # Fixed engineered alternatives.  Use odds consensus only, no future outcomes.
    return df

def score(p,y):
    p=np.asarray(p,float);y=np.asarray(y,int)
    assert p.shape==(len(y),3)
    assert np.isfinite(p).all() and (p>=0).all()
    p=p/p.sum(1,keepdims=True)
    cor=p.argmax(1)==y
    return dict(n=len(y),correct=int(cor.sum()),accuracy_pct=round(float(100*cor.mean()),3),
        log_loss=round(float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()),5),
        brier=round(float(((p-np.eye(3)[y])**2).sum(1).mean()),5),
        draws=int((p.argmax(1)==1).sum()))
def fit(train,test,columns):
    # Fixed strongly regularized multinomial fit; no hyperparameter selection.
    model=make_pipeline(SimpleImputer(strategy="median",add_indicator=False),
        StandardScaler(),LogisticRegression(C=.1,max_iter=1300,solver="lbfgs"))
    model.fit(train[columns],train.y.astype(int))
    return model.predict_proba(test[columns])
def ci(p,b,y):
    d=(p.argmax(1)==y).astype(int)-(b.argmax(1)==y).astype(int)
    rng=np.random.default_rng(71)
    a=np.array([100*rng.choice(d,len(d),replace=True).mean() for _ in range(2500)])
    return [round(float(t),3) for t in np.percentile(a,[2.5,97.5])]
def main():
    raw,original,p2,market,base=load()
    raw=prepare(raw)
    # strict same match join in reference fixture order.
    frame=original[KEY+["Season","y"]].merge(raw[KEY+[
        "p_over25","p_asian_home","ah_line","p_open_home","p_open_draw",
        "p_open_away","Season"]],on=KEY,how="left",validate="one_to_one")
    assert frame.Season_x.eq(frame.Season_y).all()
    frame=frame.rename(columns={"Season_x":"Season"}).drop(columns="Season_y")
    assert frame.y.notna().all()
    for c in ["ah_line","p_asian_home","p_over25"]:
        print("FEATURE",c,"coverage",int(frame[c].notna().sum()),len(frame),flush=True)
    full=original[KEY+["Season","y"]].copy()
    # raw 2021/22 and 2022/23 needed for season walk-forward training.
    trainpool=prepare(raw)
    trainpool["y"]=trainpool.FTR.map(LABEL)
    assert trainpool.y.notna().all()
    plans={
      "market_only_recalibration":["p_open_home","p_open_draw","p_open_away"],
      "market_plus_ah_ou":["p_open_home","p_open_draw","p_open_away","ah_line","p_asian_home","p_over25"],
      "ah_ou_without_1x2":["ah_line","p_asian_home","p_over25"],
    }
    results={"baseline":score(base,original.y),"variants":{},"per_season":{},
             "design":"Fixed C=0.1 regularized multinomial, train on all earlier seasons; replace 40% football component and keep exact archived 60% market",
             "source":"Original Football-Data.co.uk raw opening odds and 1100 baseline"}
    y=original.y.to_numpy(int)
    for arm,cols in plans.items():
        allp=np.empty((len(original),3));stats={}
        for season in TEST:
            training=trainpool[trainpool.Season<season]
            mask=frame.Season.eq(season).to_numpy()
            test=frame.loc[mask]
            pr=fit(training,test,cols)
            allp[mask]=pr
            stats[season]=score(pr,test.y.to_numpy(int))
            print("SPLIT",arm,season,stats[season],flush=True)
        res={}
        for name,p in {"standalone":allp,"fixed_blend":.4*allp+.6*market}.items():
            z=score(p,y)
            z["uplift_pp"]=round(z["accuracy_pct"]-results["baseline"]["accuracy_pct"],3)
            z["net_correct"]=z["correct"]-594
            z["paired_95pct_bootstrap"]=ci(p,base,y)
            z["fixes"]=int(((p.argmax(1)==y)&(base.argmax(1)!=y)).sum())
            z["damages"]=int(((p.argmax(1)!=y)&(base.argmax(1)==y)).sum())
            res[name]=z
            full[[arm+"_"+name+"H",arm+"_"+name+"D",arm+"_"+name+"A"]]=p
        results["variants"][arm]=res
        results["per_season"][arm]=stats
    results["passing"]= [k for k,z in results["variants"].items()
        if z["fixed_blend"]["correct"]>=616
        and z["fixed_blend"]["log_loss"]<=results["baseline"]["log_loss"]
        and z["fixed_blend"]["brier"]<=results["baseline"]["brier"]]
    results["decision"]="PASS HISTORICAL SCREEN (not prospective)" if results["passing"] else "NO-GO: +2pp all-fixture target unmet"
    full.to_csv(OUT/"predictions.csv",index=False)
    (OUT/"results.json").write_text(json.dumps(results,indent=2)+"\n")
    lines=["# H12 cross-market Asian handicap/total goals screen","",
        "**Decision: "+results["decision"]+"**","",
        "This is a fixed, bounded evaluation, not a claim of independent performance.",
        "Models were fitted chronologically with previous seasons only. No closing odds, results or same-match event features used.",
        "Historical 1100 fixture cohort has been inspected repeatedly, so positive findings would require new prospective confirmation.",
        "The 40%/60% blend holds the existing 60% archived market probabilities fixed and replaces only the 40% football component.",
        "Markets used: pre-match AvgH/AvgD/AvgA, AvgAHH/AvgAHA, AHh, Avg over/under 2.5.",
        "All wagers and implied probabilities are used only for forecasting research, not betting advice.","",
        "| Candidate | Standalone | 40/60 substituted | Uplift | LL | Brier | paired 95% CI |",
        "|---|---|---|---|---|---|---|"]
    for name,arm in results["variants"].items():
        a=arm["standalone"];b=arm["fixed_blend"]
        lines.append(f"| {name} | {a['correct']}/1100 ({a['accuracy_pct']}%) | {b['correct']}/1100 ({b['accuracy_pct']}%) | {b['uplift_pp']:+.3f}pp | {b['log_loss']} | {b['brier']} | {b['paired_95pct_bootstrap']} |")
    lines.extend(["","## Reproduce", "- Script: experiments/h12_cross_market/screen.py",
        "- Output: reports/h12_cross_market/results.json",
        "- Fixture probabilities: reports/h12_cross_market/predictions.csv",
        "","## Deployment constraint",
        "The project's live bookmaker feed supplies 1X2 odds only; sourcing equivalent Asian handicap and goal-total prices at the matching T-24 freeze remains unverified. Even a historical PASS would not justify production.",
        ""])
    (ROOT/"docs/experiments/H12_CROSS_MARKET_SCREEN.md").write_text("\n".join(lines)+"\n")
    print("H12_RESULT",json.dumps(results),flush=True)
if __name__=="__main__":
    main()
