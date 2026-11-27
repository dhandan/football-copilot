"""Generate the GW12 pre-match journal directly from frozen, immutable CSVs."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
GW=12
P="2026_27_gw12"
KEYS=["FixtureDate","FixtureTime","HomeTeam","AwayTeam"]
LABEL={"H":"Home","D":"Draw","A":"Away"}

def main():
    root=ROOT/"data/live"
    model=pd.read_csv(root/"predictions"/f"{P}_predictions.csv")
    fixed=pd.read_csv(root/"challenger_predictions"/f"{P}_market_shadow_predictions.csv")
    adaptive=pd.read_csv(root/"challenger_predictions"/f"{P}_adaptive_confidence_v1.csv")
    markets=list((root/"market").glob(f"{P}_opening_market_official_*.csv"))
    if len(markets)!=1:raise RuntimeError("Exactly one official GW12 market snapshot required")
    market=pd.read_csv(markets[0])
    for label,d in [("model",model),("fixed",fixed),("market",market),("adaptive",adaptive)]:
        if len(d)!=10 or d.duplicated(KEYS).any() or not set(map(tuple,d[KEYS].to_numpy()))==set(map(tuple,model[KEYS].to_numpy())):
            raise RuntimeError("Fixture identity mismatch: "+label)
    if not (model.ModelVersion.eq("Model2_v1.0").all() and model.PredictionStatus.eq("Predicted").all()):
        raise RuntimeError("Model 2 prediction version/status mismatch")
    if not (fixed.ChallengerVersion.eq("Model2_OpeningMarket_40_60_v1.0").all() and
            fixed.SnapshotType.eq("OFFICIAL").all() and fixed.ModelWeight.eq(.4).all() and
            fixed.MarketWeight.eq(.6).all() and fixed.ForecastRole.eq("ProvisionalChampion").all() and
            model.ForecastRole.eq("Benchmark").all() and adaptive.ForecastRole.eq("ProspectiveChallenger").all()):
        raise RuntimeError("Unapproved champion or challenger version")
    if not adaptive.ExperimentVersion.eq("AdaptiveMarketConfidence_v1.0").all():
        raise RuntimeError("Adaptive challenger version mismatch")
    m=model[["HomeWinProbability","DrawProbability","AwayWinProbability"]].to_numpy(dtype=float)/100
    m/=m.sum(axis=1,keepdims=True)
    k=fixed[["MarketHomeProbability","MarketDrawProbability","MarketAwayProbability"]].to_numpy(dtype=float)
    got=fixed[["ShadowHomeProbability","ShadowDrawProbability","ShadowAwayProbability"]].to_numpy(dtype=float)
    if not (np.allclose(got,.4*m+.6*k,atol=1e-8) and np.allclose(got.sum(axis=1),1,atol=1e-8)):
        raise RuntimeError("Recorded 40/60 vectors do not reconcile")
    earliest=pd.to_datetime(model.KickoffUTC,utc=True).min()
    captures=pd.to_datetime(model.PredictionTimestamp,utc=True)
    odds_captures=pd.to_datetime(market.SnapshotTimestamp,utc=True)
    if not ((captures.to_numpy()<pd.to_datetime(model.KickoffUTC,utc=True).to_numpy()).all() and
            (odds_captures.to_numpy()<pd.to_datetime(market.KickoffUTC,utc=True).to_numpy()).all()):
        raise RuntimeError("At least one forecast after kickoff; refuse journal")
    fixtures=model[KEYS+["MostLikelyScore","LeadHours","T24Eligible","ColdStartUsed"]].merge(
        fixed[KEYS+["ShadowHomeProbability","ShadowDrawProbability","ShadowAwayProbability","ShadowPredictedResult",
                    "MarketHomeProbability","MarketDrawProbability","MarketAwayProbability","BookmakerCount"]],
        on=KEYS,validate="one_to_one").merge(adaptive[KEYS+["HomeProbability","DrawProbability","AwayProbability","PredictedResult"]],
        on=KEYS,validate="one_to_one")
    fixtures=fixtures.sort_values(["FixtureDate","FixtureTime"]).reset_index(drop=True)
    def cell(row,cols):
        arr=np.asarray([row[c] for c in cols])
        ix=int(arr.argmax())
        return f"{'HDA'[ix]} {arr[ix]*100:.1f}%"
    def model_cell(row):
        arr=np.array([row["HomeWinProbability"],row["DrawProbability"],row["AwayWinProbability"]],float)
        return f"{'HDA'[arr.argmax()]} {arr.max():.1f}%"
    by_model=model.set_index(KEYS)
    rows=[]
    for _,r in fixtures.iterrows():
        key=tuple(r[x] for x in KEYS)
        mod=by_model.loc[key]
        rows.append(f'| {r.FixtureDate} {r.FixtureTime} | {r.HomeTeam} vs {r.AwayTeam} | {mod.MostLikelyScore} | '
                    f'{model_cell(mod)} | {cell(r,["MarketHomeProbability","MarketDrawProbability","MarketAwayProbability"])} | '
                    f'{cell(r,["ShadowHomeProbability","ShadowDrawProbability","ShadowAwayProbability"])} | '
                    f'{cell(r,["HomeProbability","DrawProbability","AwayProbability"])} |')
    lead=float(model.LeadHours.min())
    eligible=int(model.T24Eligible.astype(str).str.lower().eq("true").sum())
    cold=fixtures.loc[fixtures.ColdStartUsed.astype(str).str.lower().eq("true"),["HomeTeam","AwayTeam"]]
    cold_text=", ".join(f"{x.HomeTeam} vs {x.AwayTeam}" for x in cold.itertuples(index=False)) or "none"
    capture=str(captures.iloc[0])
    market_capture=str(odds_captures.iloc[0])
    statuses="\n".join(f"- {'PASS' if flag else 'EXCEPTION'}: {label}" for label,flag in [
        ("Exactly 10 matching fixture IDs/teams/date/time",True),
        ("Both model and odds captured before all 10 kickoffs",True),
        ("Model2 v1.0 unchanged, 40/60 provisional champion unchanged",True),
        ("Registered adaptive v1.0 frozen independently",True),
        ("All 40/60 probability vectors reconciled",True),
        ("24-hour target met across all 10",eligible==10),
    ])
    text=f"""# Football Copilot 2026/27: Gameweek 12

**Status: OFFICIAL PRE-MATCH FROZEN — {capture}. Results and post-match scoring PENDING.**

## Release authority and explicit GW12 exception

[Operating checklist](OPERATING_CHECKLIST.md) applied with user-authorised exception to the **24-hour target only**. Never waive pre-kickoff capture, fixture completeness, genuine official market sourcing or immutable evidence. Actual earliest kickoff: **{earliest.tz_convert("Europe/London").strftime("%A %d %B %Y %H:%M %Z")}**. Earliest lead: **{lead:.2f} hours**. **{eligible}/10** met T−24. The latest live odds are a **Friday capture, not necessarily opening odds**, and may not be comparable to prior T−24 snapshots. No odds were backdated or backfilled.

## Forecast sources and roles

- [Frozen Model 2 benchmark](../../data/live/predictions/{P}_predictions.csv), captured {capture}
- [Official live UK market snapshot](../../data/live/market/{markets[0].name}), captured {market_capture}, {int(market.BookmakerCount.min())}–{int(market.BookmakerCount.max())} bookmakers/fixture
- [Frozen fixed 40/60 provisional champion](../../data/live/challenger_predictions/{P}_market_shadow_predictions.csv): 40% Model 2 + 60% same-market UK no-vig 1X2
- [Frozen adaptive v1.0 challenger](../../data/live/challenger_predictions/{P}_adaptive_confidence_v1.csv) and [version/hash manifest](../../data/live/challenger_predictions/{P}_adaptive_confidence_v1.json). This is **not promoted**
- [Latest point-in-time FPL player/fixture snapshot](../../data/live/players/h07_snapshots/). H07 observational, no unproven player/injury model adjustment

## All ten pre-match predictions

Each pick is the most likely **1X2 class** and its probability, not a guaranteed result. The Model 2 modal exact score is a separate output.

| Kickoff UK GMT | Fixture | Model 2 modal score | Model 2 1X2 | Market-only 1X2 | Fixed 40/60 1X2 (primary) | Adaptive 1X2 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Full pre-match 40/60 probability vectors

| Fixture | Home | Draw | Away |
|---|---:|---:|---:|
{chr(10).join(f'| {r.HomeTeam} vs {r.AwayTeam} | {r.ShadowHomeProbability*100:.2f}% | {r.ShadowDrawProbability*100:.2f}% | {r.ShadowAwayProbability*100:.2f}% |' for r in fixtures.itertuples(index=False))}

## Diagnostics and freeze controls

- Selected draws (fixed 40/60): **{int(fixed.ShadowPredictedResult.eq("D").sum())}/10**. Do not reinterpret Model 2's modal 1–1 as its 1X2 draw prediction.
- Modal 1–1 scorelines (Model 2): **{int(model.MostLikelyScore.astype(str).eq("1-1").sum())}/10**.
- Cold-start fixtures: **{len(cold)}/10**: {cold_text}.
- All four versions and probability inputs are frozen, with no GW12 outcomes involved.
{statuses}
- No unsupported claims of a ≥2pp improvement: historical fixed-blend benchmark is **594/1,100 (54%)**, unchanged target **616/1,100 (56%)**.

## After GW12 (NOT YET PERFORMED)

- Verify 10/10 final scores, align stable fixture IDs and confirm status with independent sources.
- Score exactly the frozen Model 2, market, fixed 40/60, and adaptive vectors on the same matches: accuracy, Log Loss, multiclass Brier, draw recall, cold starts, exact scores, fixture disagreements and diagnostics.
- Update cumulative like-for-like GW6–GW12, write an independent retrospective, and decide keep/change/reject without modifying frozen forecasts.
- Sprint scope: retain 40/60 provisional champion; prioritise point-in-time source feasibility for truly independent information (Model 7 A/B/C and H09), while keeping H03 v1 rejected and V2 conversational functions operational.

**Lifecycle:** Only pre-match release is complete. Not Gameweek complete until final verified results and retrospective.
"""
    output=ROOT/"docs/gameweeks/GW12.md"
    if output.exists(): raise FileExistsError("GW12 journal already exists; do not overwrite")
    output.write_text(text,encoding="utf-8")
    print("GW12 JOURNAL GENERATED:",output, "rows",len(rows), "T24",eligible,flush=True)


if __name__=="__main__":
    main()
