from __future__ import annotations
import ast, io, json, os, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from catboost import CatBoostClassifier
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = ROOT / "reports/model7_screen"
OUT.mkdir(parents=True, exist_ok=True)
SEASONS = ["2021/22","2022/23","2023/24","2024/25","2025/26"]
METRICS = ["totalScoringAtt","attemptsIbox","attemptsConcededIbox","touchesInOppBox",
           "totalFinalThirdPasses","finalThirdEntries","possWonAtt3rd","possessionPercentage"]
MAP = {"Man Utd":"Man United","Spurs":"Tottenham","Sheff Utd":"Sheffield United",
       "Coventry":"Coventry City","Hull":"Hull City"}
CATS = dict(iterations=350, depth=4, learning_rate=.035, l2_leaf_reg=8,
            loss_function="MultiClass", random_seed=42, thread_count=2,
            verbose=False, allow_writing_files=False)

def base_cols():
    tree = ast.parse(Path("prediction/train_model_v2.py").read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(
            isinstance(t,ast.Name) and t.id=="MODEL_2_FEATURES" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Cannot locate established Model 2 features")

BASE = base_cols()
assert len(BASE)==25, len(BASE)
EXTRA = ["EventDelta_"+m for m in METRICS]
ALL = BASE+EXTRA

def event_data():
    tables=[]
    for yy in ["2021-22","2022-23","2023-24","2024-25","2025-26","2026-27"]:
        url="https://raw.githubusercontent.com/imadeddine-belkat/Premier-League-Stats/main/pl_stats/_merged/events/"+yy+"_events_stats.csv"
        res=requests.get(url,timeout=75)
        res.raise_for_status()
        e=pd.read_csv(io.BytesIO(res.content),low_memory=False)
        assert set(METRICS).issubset(e.columns),(yy,len(e))
        e["team"]=e["team"].replace(MAP)
        e["Date"]=pd.to_datetime(e["kickoff"]).dt.strftime("%Y-%m-%d")
        e["MatchId"]=e["matchId"]
        if yy!="2026-27": assert len(e)==760,(yy,len(e))
        tables.append(e)
    events=pd.concat(tables,ignore_index=True)
    events=events.sort_values(["Date","MatchId","team"],kind="stable")
    for col in METRICS:
        events[col]=pd.to_numeric(events[col],errors="coerce")
        events["lag_"+col]=events.groupby("team")[col].transform(
            lambda x:x.shift(1).rolling(5,min_periods=1).mean()
        )
    return events

def add_event(frame,events):
    frame=frame.copy()
    frame["Date"]=pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d")
    cols=["Date","team","venue"]+["lag_"+x for x in METRICS]
    hm=events.loc[events["venue"]=="Home",cols].drop(columns="venue")
    aw=events.loc[events["venue"]=="Away",cols].drop(columns="venue")
    hm=hm.rename(columns={"team":"HomeTeam",**{"lag_"+x:"H_"+x for x in METRICS}})
    aw=aw.rename(columns={"team":"AwayTeam",**{"lag_"+x:"A_"+x for x in METRICS}})
    assert not hm.duplicated(["Date","HomeTeam"]).any()
    assert not aw.duplicated(["Date","AwayTeam"]).any()
    frame=frame.merge(hm,on=["Date","HomeTeam"],how="left",validate="many_to_one",indicator="joinedH")
    frame=frame.merge(aw,on=["Date","AwayTeam"],how="left",validate="many_to_one",indicator="joinedA")
    for x in METRICS: frame["EventDelta_"+x]=frame["H_"+x]-frame["A_"+x]
    return frame

def from_goals(h,a):
    h=np.clip(np.asarray(h,float),.1,8)
    a=np.clip(np.asarray(a,float),.1,8)
    goals=np.arange(9)
    ph=poisson.pmf(goals[None,:],h[:,None])
    pa=poisson.pmf(goals[None,:],a[:,None])
    prod=ph[:,:,None]*pa[:,None,:]
    i,j=np.indices((9,9))
    p=np.stack([prod[:,i>j].sum(axis=1),prod[:,i==j].sum(axis=1),
                prod[:,i<j].sum(axis=1)],axis=1)
    return p/p.sum(axis=1,keepdims=True)

def predict(train,test,arm):
    if arm in ("A","C"):
        cols=BASE if arm=="A" else ALL
        x=train[cols].fillna(-999)
        v=test[cols].fillna(-999)
        clf=CatBoostClassifier(**CATS)
        clf.fit(x,train["Target"].astype(int))
        return clf.predict_proba(v)
    def make_model():
        return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                             PoissonRegressor(alpha=.2,max_iter=1500))
    h,a=make_model(),make_model()
    h.fit(train[ALL],train["HomeGoals"])
    a.fit(train[ALL],train["AwayGoals"])
    return from_goals(h.predict(test[ALL]),a.predict(test[ALL]))

def score(p,y):
    p=np.asarray(p,float)
    p=p/p.sum(axis=1,keepdims=True)
    y=np.asarray(y,int)
    acc=p.argmax(1)==y
    return dict(n=len(y),correct=int(acc.sum()),accuracy_pct=round(100*acc.mean(),3),
                logloss=round(float(-np.log(np.maximum(1e-12,p[np.arange(len(y)),y])).mean()),5),
                brier=round(float(np.sum((p-np.eye(3)[y])**2,axis=1).mean()),5),
                draws=int((p.argmax(1)==1).sum()))

def paired_ci(p,b,y):
    delta=(p.argmax(1)==y).astype(int)-(b.argmax(1)==y).astype(int)
    rng=np.random.default_rng(42)
    sim=[100*rng.choice(delta,size=len(delta),replace=True).mean() for _ in range(2500)]
    return [round(v,3) for v in np.percentile(sim,[2.5,97.5])]

def main():
    events=event_data()
    subprocess.run(["python","prediction/build_features_v2.py"],check=True,stdout=subprocess.DEVNULL)
    built=pd.read_csv("data/processed/prediction_features_v2.csv")
    built=add_event(built,events)
    matched=((built.joinedH=="both")&(built.joinedA=="both")).mean()
    print("Event identity matched",round(float(matched),5),"of",len(built),flush=True)
    assert matched>=.99
    built["Target"]=np.where(built.HomeGoals>built.AwayGoals,0,
                       np.where(built.HomeGoals==built.AwayGoals,1,2))
    base=pd.read_csv("reports/model6/model6_walk_forward_predictions.csv")
    base["Date"]=pd.to_datetime(base["Date"]).dt.strftime("%Y-%m-%d")
    base["Target"]=base["ActualResult"].map({"H":0,"D":1,"A":2})
    assert len(base)==1100
    keys=["Date","HomeTeam","AwayTeam"]
    check=base.merge(built[keys+["Season"]],on=keys,how="left",indicator=True,validate="one_to_one")
    assert check._merge.eq("both").all()
    assert check.Season_x.eq(check.Season_y).all()
    order=base[keys].reset_index(drop=True)
    seasoncol=base.Season.tolist()
    p2=base[["Model2HomeProb","Model2DrawProb","Model2AwayProb"]].to_numpy(float)
    market=base[["MarketOpenHomeProb","MarketOpenDrawProb","MarketOpenAwayProb"]].to_numpy(float)
    bench=.4*p2+.6*market
    y=base.Target.to_numpy(int)
    benchmark=score(bench,y)
    assert benchmark["correct"]==594,benchmark
    outcomes={"benchmark":benchmark,"arms":{},"season_results":{},
              "source_match_rate":round(float(matched),6),
              "parameters":CATS,"feature_names":ALL}
    predictions=base[keys+["Season","ActualResult"]].copy()
    predictions[["BenchmarkH","BenchmarkD","BenchmarkA"]]=bench
    for arm in ["A","B","C"]:
        subs=[];perseason={}
        for yr in ["2023/24","2024/25","2025/26"]:
            train=built.loc[built.Season<yr]
            left=base.loc[base.Season==yr,keys]
            test=left.merge(built,on=keys,how="left",validate="one_to_one")
            assert test.Season.eq(yr).all(),(arm,yr)
            p=predict(train,test,arm)
            subs.append(p)
            perseason[yr]=score(p,test.Target.to_numpy(int))
            print("Evaluated",arm,yr,perseason[yr],flush=True)
        p=np.concatenate(subs)
        assert np.array_equal(
            pd.concat([base.loc[base.Season==yr,keys] for yr in ["2023/24","2024/25","2025/26"]]).reset_index(drop=True).to_numpy(),
            order.to_numpy())
        forms={"standalone":p,"fixed_blend":.4*p+.6*market}
        outcomes["arms"][arm]={}
        for name,proba in forms.items():
            z=score(proba,y)
            z["uplift_pp"]=round(z["accuracy_pct"]-benchmark["accuracy_pct"],3)
            z["paired_95pct_bootstrap_ci_pp"]=paired_ci(proba,bench,y)
            z["additional_correct"]=z["correct"]-benchmark["correct"]
            z["fixes"]=int(((proba.argmax(1)==y)&(bench.argmax(1)!=y)).sum())
            z["damages"]=int(((proba.argmax(1)!=y)&(bench.argmax(1)==y)).sum())
            outcomes["arms"][arm][name]=z
            predictions[[arm+"_"+name+"H",arm+"_"+name+"D",arm+"_"+name+"A"]]=proba
        outcomes["season_results"][arm]=perseason
    best=max([z["fixed_blend"]["correct"] for z in outcomes["arms"].values()])
    passed=any(z["fixed_blend"]["correct"]>=616 and
               z["fixed_blend"]["logloss"]<=benchmark["logloss"] and
               z["fixed_blend"]["brier"]<=benchmark["brier"]
               for z in outcomes["arms"].values())
    outcomes["verdict"]="PASS HISTORICAL SCREEN, not independent validation" if passed else "NO-GO full model build: no qualified +2pp historical result"
    outcomes["best_fixed_blend_correct"]=best
    predictions.to_csv(OUT/"predictions.csv",index=False)
    (OUT/"results.json").write_text(json.dumps(outcomes,indent=2)+"\n")
    lines=["# Model 7 A/B/C historical empirical screen","",
           "**Status:** "+outcomes["verdict"],"",
           "Evaluated on the identical 1,100 saved historical fixtures, with earlier-season-only training.",
           "Results are **exploratory**: earlier historical tests have repeatedly examined this cohort.",
           "Hyperparameters and event features were fixed in the script before testing; no outcome-directed retuning was performed.",
           "The raw external event source provides post-match records; all team feature averages shift by one full match before rolling.","",
           "**Fixed comparator:** 40% original Model 2 plus 60% saved market; "+
           str(benchmark["correct"])+"/1100, "+str(benchmark["accuracy_pct"])+"%","",
           "| Arm | Forecast | Correct | Accuracy | Uplift | Log Loss | Brier | Paired uplift 95% bootstrap interval |",
           "|---|---|---:|---:|---:|---:|---:|---|"]
    for arm in ["A","B","C"]:
        for form in ["standalone","fixed_blend"]:
            z=outcomes["arms"][arm][form]
            lines.append(f"| {arm} | {form} | {z['correct']}/1100 | {z['accuracy_pct']}% | {z['uplift_pp']:+.3f}pp | {z['logloss']} | {z['brier']} | {z['paired_95pct_bootstrap_ci_pp']} |")
    lines += ["","## Interpretation","",
              "The screening threshold is 616 correct of 1,100 while retaining Log Loss and Brier quality. "
              "A pass is only a reason to set up new prospective validation, not proof of future accuracy.",
              "This is not an untouched holdout and does not establish the expected uplift with confidence.",
              "The shared 2026/27 100-match archive has been feasibility-audited, but 2026/27 outcomes were not used for training or scoring this historical experiment.",
              "Data source rights, publication lag, and stable live supply must be resolved before production deployment.",
              "","## Audit files","- reports/model7_screen/results.json",
              "- reports/model7_screen/predictions.csv",
              "- experiments/model7_abc/bounded_screen.py"]
    Path("docs/experiments/MODEL7_ABC_EMPIRICAL_RESULTS.md").write_text("\n".join(lines)+"\n")
    print("MODEL7_RESULT",json.dumps(outcomes),flush=True)

if __name__=="__main__":
    main()
