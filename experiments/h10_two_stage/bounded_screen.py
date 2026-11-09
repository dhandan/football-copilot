"""Chronological two-stage shots/corners -> 1X2 screen; no live production edits.
Inspired by Wheatcroft (2021), but not an exact GAP replication.
"""
import ast, json, subprocess, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"reports/h10_two_stage"
OUT.mkdir(parents=True,exist_ok=True)
SEASONS=["2021/22","2022/23","2023/24","2024/25","2025/26"]
TEST=SEASONS[2:]
STATS=["SOT_H","SOT_A","NONTARGET_H","NONTARGET_A","CORN_H","CORN_A"]
FIELDS=["sot_for","sot_against","non_for","non_against","corner_for","corner_against"]
EXTRAS=["home_"+x+"_lag5" for x in FIELDS]+["away_"+x+"_lag5" for x in FIELDS]
LABEL={"H":0,"D":1,"A":2}

def get_original_features():
    tree=ast.parse((ROOT/"prediction/train_model_v2.py").read_text())
    for statement in tree.body:
        if isinstance(statement,ast.Assign):
            if any(isinstance(t,ast.Name) and t.id=="MODEL_2_FEATURES" for t in statement.targets):
                a=ast.literal_eval(statement.value)
                assert len(a)==25
                return a
    raise RuntimeError("Original 25 model features not found")

BASE=get_original_features()
ALL=BASE+EXTRAS
KEY=["Date","HomeTeam","AwayTeam"]

def historical_stats():
    m=pd.read_csv(ROOT/"data/processed/matches_clean.csv")
    assert len(m)==1900,len(m)
    m.Date=pd.to_datetime(m.Date).dt.strftime("%Y-%m-%d")
    for k in ["HS","AS","HST","AST","HC","AC"]:
        m[k]=pd.to_numeric(m[k],errors="coerce")
    m["SOT_H"]=m.HST
    m["SOT_A"]=m.AST
    m["NONTARGET_H"]=(m.HS-m.HST).clip(lower=0)
    m["NONTARGET_A"]=(m.AS-m.AST).clip(lower=0)
    m["CORN_H"]=m.HC
    m["CORN_A"]=m.AC
    chunks=[]
    for side in ("home","away"):
        r=m[KEY].copy()
        r["Team"]=m.HomeTeam if side=="home" else m.AwayTeam
        r["side"]=side
        for name,h,a in [
            ("sot_for","SOT_H","SOT_A"),("sot_against","SOT_A","SOT_H"),
            ("non_for","NONTARGET_H","NONTARGET_A"),("non_against","NONTARGET_A","NONTARGET_H"),
            ("corner_for","CORN_H","CORN_A"),("corner_against","CORN_A","CORN_H")]:
            r[name]=m[h].values if side=="home" else m[a].values
        chunks.append(r)
    sides=pd.concat(chunks,ignore_index=True).sort_values(["Team","Date"],kind="stable")
    assert not sides.duplicated(["Team","Date"]).any()
    for field in FIELDS:
        sides[field+"_lag5"]=sides.groupby("Team")[field].transform(
            lambda z:z.shift(1).rolling(5,min_periods=1).mean())
    for side in ("home","away"):
        block=sides[sides.side==side][KEY+[x+"_lag5" for x in FIELDS]].rename(
            columns={x+"_lag5":side+"_"+x+"_lag5" for x in FIELDS})
        m=m.merge(block,on=KEY,how="left",validate="one_to_one",indicator=True)
        assert m._merge.eq("both").all()
        m=m.drop(columns="_merge")
    return m

def build_features():
    subprocess.run(["python","prediction/build_features_v2.py"],cwd=ROOT,
                   check=True,stdout=subprocess.DEVNULL)
    p=pd.read_csv(ROOT/"data/processed/prediction_features_v2.csv")
    p.Date=pd.to_datetime(p.Date).dt.strftime("%Y-%m-%d")
    hist=historical_stats()
    p=p.merge(hist[KEY+STATS+EXTRAS],on=KEY,how="left",indicator=True,validate="one_to_one")
    assert p._merge.eq("both").all()
    p=p.drop(columns="_merge")
    assert p[STATS].notna().all().all(),"Missing target match statistics"
    p["Target"]=np.where(p.HomeGoals>p.AwayGoals,0,
                         np.where(p.HomeGoals==p.AwayGoals,1,2))
    return p.sort_values(["Season","Date","HomeTeam","AwayTeam"]).reset_index(drop=True)

def stage1(p,season):
    train=p[p.Season<season]
    test=p[p.Season==season]
    x=train[ALL].fillna(0)
    xv=test[ALL].fillna(0)
    output=test[["Season"]+KEY+["Target"]+STATS].copy()
    for stat in STATS:
        reg=make_pipeline(StandardScaler(),PoissonRegressor(alpha=.5,max_iter=500))
        reg.fit(x,train[stat])
        output["pred_"+stat]=reg.predict(xv)
    return output

def projected_features(frame):
    arr=np.column_stack([
        frame.pred_SOT_H-frame.pred_SOT_A,
        frame.pred_NONTARGET_H-frame.pred_NONTARGET_A,
        frame.pred_CORN_H-frame.pred_CORN_A,
        frame.pred_SOT_H+frame.pred_SOT_A,
        frame.pred_NONTARGET_H+frame.pred_NONTARGET_A,
        frame.pred_CORN_H+frame.pred_CORN_A])
    assert np.isfinite(arr).all()
    return arr

def stage2(train,test,mode):
    x=projected_features(train)
    v=projected_features(test)
    mu=x.mean(axis=0)
    std=np.where(x.std(axis=0)<.001,1,x.std(axis=0))
    x=(x-mu)/std
    v=(v-mu)/std
    if mode=="ordinal":
        y=2-train.Target.to_numpy(int)
        model=OrderedModel(y,x,distr="logit")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit=model.fit(method="bfgs",disp=False,maxiter=500)
        if not fit.mle_retvals.get("converged",False):
            raise RuntimeError("Ordinal regression did not converge")
        return np.asarray(model.predict(fit.params,exog=v),dtype=float)[:,::-1]
    clf=LogisticRegression(C=.7,max_iter=800,solver="lbfgs")
    clf.fit(x,train.Target.astype(int))
    return clf.predict_proba(v)

def direct(train,test):
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=.7,max_iter=800,solver="lbfgs"))
    clf.fit(train[ALL].fillna(0),train.Target.astype(int))
    return clf.predict_proba(test[ALL].fillna(0))

def metric(p,y):
    p=np.asarray(p,float)
    y=np.asarray(y,int)
    assert p.shape==(len(y),3)
    assert np.isfinite(p).all() and (p>=0).all()
    p=p/p.sum(1,keepdims=True)
    correct=p.argmax(1)==y
    return {"n":len(y),"correct":int(correct.sum()),"accuracy_pct":round(100*correct.mean(),3),
            "logloss":round(float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()),5),
            "brier":round(float(((p-np.eye(3)[y])**2).sum(1).mean()),5),
            "draw_picks":int((p.argmax(1)==1).sum())}

def interval(p,baseline,y):
    a=(p.argmax(1)==y).astype(int)-(baseline.argmax(1)==y).astype(int)
    rng=np.random.default_rng(42)
    sim=[rng.choice(a,size=len(a),replace=True).mean()*100 for _ in range(3000)]
    return [round(float(k),3) for k in np.percentile(sim,[2.5,97.5])]

def run():
    p=build_features()
    print("Feature rows:",len(p),p.Season.value_counts().sort_index().to_dict(),flush=True)
    ref=pd.read_csv(ROOT/"reports/model6/model6_walk_forward_predictions.csv")
    ref.Date=pd.to_datetime(ref.Date).dt.strftime("%Y-%m-%d")
    ref["Target"]=ref.ActualResult.map(LABEL)
    assert len(ref)==1100
    check=ref.merge(p[KEY+["Season"]],on=KEY,validate="one_to_one",indicator=True)
    assert len(check)==1100 and check._merge.eq("both").all()
    assert check.Season_x.eq(check.Season_y).all()
    f2=ref[["Model2HomeProb","Model2DrawProb","Model2AwayProb"]].to_numpy(float)
    fm=ref[["MarketOpenHomeProb","MarketOpenDrawProb","MarketOpenAwayProb"]].to_numpy(float)
    fixed=.4*f2+.6*fm
    y=ref.Target.to_numpy(int)
    baseline=metric(fixed,y)
    assert baseline["correct"]==594,baseline

    oof={}
    stage1_mae={}
    for season in SEASONS[1:]:
        frame=stage1(p,season)
        oof[season]=frame
        stage1_mae[season]={k:round(float(abs(frame[k]-frame["pred_"+k]).mean()),3) for k in STATS}
        print("Stage 1",season,len(frame),stage1_mae[season],flush=True)

    preds={k:[] for k in ["ordinal","multinomial","direct"]}
    year={}
    for season in TEST:
        train_oof=pd.concat([oof[s] for s in SEASONS[1:] if s<season],ignore_index=True)
        test_oof=oof[season]
        train=p[p.Season<season]
        test=p[p.Season==season]
        assert train_oof.Season.lt(season).all()
        assert test[KEY].reset_index(drop=True).equals(test_oof[KEY].reset_index(drop=True))
        row={}
        for name in preds:
            forecast=direct(train,test) if name=="direct" else stage2(train_oof,test_oof,name)
            preds[name].append(forecast)
            row[name]=metric(forecast,test_oof.Target.to_numpy(int))
            print("Stage 2",season,name,row[name],flush=True)
        year[season]=row

    expected_keys=pd.concat([oof[s][KEY] for s in TEST],ignore_index=True)
    assert expected_keys.equals(ref[KEY].reset_index(drop=True)), "Reference ordering mismatch"
    results={"baseline":baseline,"stage1_mae":stage1_mae,"year_standalone":year,
             "models":{},"method":"six stage-1 Poisson shot/corner regressions; ordered logit and multinomial stage 2, chronological out-of-season stage-1 predictions; direct logistic comparator",
             "source":"2021/22-2025/26 Football-Data match statistics",
             "test_fixture_count":1100}
    save=ref[KEY+["Season","ActualResult"]].copy()
    save[["BaselineH","BaselineD","BaselineA"]]=fixed
    for name,parts in preds.items():
        forecast=np.concatenate(parts)
        models={}
        for form,prob in [("standalone",forecast),("fixed_blend",.4*forecast+.6*fm)]:
            val=metric(prob,y)
            val["uplift_pp"]=round(val["accuracy_pct"]-baseline["accuracy_pct"],3)
            val["net_more_correct"]=val["correct"]-baseline["correct"]
            val["paired_bootstrap_95pct_pp"]=interval(prob,fixed,y)
            val["fixes"]=int(((prob.argmax(1)==y)&(fixed.argmax(1)!=y)).sum())
            val["damages"]=int(((prob.argmax(1)!=y)&(fixed.argmax(1)==y)).sum())
            models[form]=val
            save[[name+"_"+form+"_H",name+"_"+form+"_D",name+"_"+form+"_A"]]=prob
        results["models"][name]=models
    passed=[name for name,m in results["models"].items() if
            m["fixed_blend"]["correct"]>=616 and
            m["fixed_blend"]["logloss"]<=baseline["logloss"] and
            m["fixed_blend"]["brier"]<=baseline["brier"]]
    results["passed"]=passed
    results["decision"]="HISTORICAL DEVELOPMENT SCREEN PASS only; not independent proof" if passed else "NO-GO: +2pp development accuracy hurdle not met"
    (OUT/"results.json").write_text(json.dumps(results,indent=2)+"\n")
    save.to_csv(OUT/"predictions.csv",index=False)
    lines=["# H10 two-stage match-statistic screen","",
           "Status: "+results["decision"],"",
           "Inspired by Wheatcroft (2021), but not an exact implementation of GAP ratings.",
           "The test builds strictly lagged shots/corners features and trains six stage-1 forecasts out of season. Stage 2 is trained only on out-of-season stage-1 predictions, never actual match statistics for those fixtures.",
           "Baseline fixed 40% Model 2 / 60% market: 594/1100 (54.00%). Target >=616/1100 with no Log Loss or Brier deterioration.",
           "Historical 1100-fixture cohort previously used in many tests, so this is a development screen, not pristine out-of-sample validation.","",
           "| Method | Standalone | Fixed 40/60 blend | Accuracy uplift | 95% paired bootstrap CI | Log Loss | Brier |",
           "|---|---:|---:|---:|---|---:|---:|"]
    for name,model in results["models"].items():
        a=model["standalone"];b=model["fixed_blend"]
        lines.append(f"| {name} | {a['accuracy_pct']}% | {b['accuracy_pct']}% ({b['correct']}/1100) | {b['uplift_pp']:+.3f}pp | {b['paired_bootstrap_95pct_pp']}pp | {b['logloss']} | {b['brier']} |")
    lines+=["","## Caveats",
            "- Off-target proxy is total shot attempts minus on-target attempts; blocked attempts are included.",
            "- Stage 1 uses original 25 pre-match football features plus 12 prior-five-match shot/corner features.",
            "- Predictions within a test season may use team match statistics from earlier fixtures in that season, not future matches.",
            "- No hyperparameter search on test outcomes. Paired bootstrap intervals do not correct for prior historical research or season clustering.",
            "- Source data and reference predictions unchanged. No 2026/27 results were used.",
            "- The referenced study primarily demonstrates useful probabilistic and betting signals, not +2pp 1X2 accuracy.",
            "- Full technical reproduction of GAP ratings is outside this bounded screen; the core two-stage causal question is tested.",
            "","## Files","- experiments/h10_two_stage/bounded_screen.py",
            "- reports/h10_two_stage/results.json",
            "- reports/h10_two_stage/predictions.csv",
            "- https://journals.sagepub.com/doi/full/10.3233/JSA-200462"]
    (ROOT/"docs/experiments/H10_TWO_STAGE_STAT_SCREEN.md").write_text("\n".join(lines)+"\n")
    print("H10_RESULT",json.dumps(results),flush=True)

if __name__=="__main__":
    run()
