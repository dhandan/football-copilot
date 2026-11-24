"""H03 v1: point-in-time draw-discrimination experiment. No fitted parameters.

Read-only use of archived predictions and verified results. GW1-5 development
market probabilities originate in a retrospective opening-odds comparison;
GW6-11 use official frozen pre-match 40/60 snapshots. Never mix their provenance.
No forecast or official result file is overwritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KEY = ["Gameweek", "FixtureDate", "HomeTeam", "AwayTeam"]
BASE = ["ShadowHomeProbability", "ShadowDrawProbability", "ShadowAwayProbability"]
LABELS = np.array(["H", "D", "A"])
OUTCOME = {"H": 0, "D": 1, "A": 2}
VERSION = "H03_DrawBalanceLowGoals_v1_experimental"
# Pre-specified research hypothesis, NOT chosen by searching the held-out results.
# A balanced, low-scoring expected-goals profile should raise draw likelihood.
# If the signal is weak, move probability back to home/away in their prior ratio.
DRAW_ADJUSTMENT = 0.12
NEUTRAL_SIGNAL = 0.35
BALANCE_SCALE = 0.60
LOW_GOALS_CEILING = 3.20
LOW_GOALS_SPAN = 1.60


def _probabilities(a: np.ndarray, *, tolerance: float = 0.003) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or not len(a):
        raise ValueError("Expected nonempty N x 3 probability array")
    if not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
        raise ValueError("Invalid probability value")
    if not np.allclose(a.sum(axis=1), 1.0, atol=tolerance):
        raise ValueError("Probability rows do not sum to one")
    return a / a.sum(axis=1, keepdims=True)


def adjust_draws(blend: np.ndarray, xg_home: np.ndarray,
                 xg_away: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Never uses outcomes; leaves relative H:A win odds invariant."""
    blend = _probabilities(blend, tolerance=1e-6)
    h = np.asarray(xg_home, dtype=float)
    a = np.asarray(xg_away, dtype=float)
    if h.shape != (len(blend),) or a.shape != (len(blend),):
        raise ValueError("Expected one home/away xG pair for every fixture")
    if not np.isfinite(h).all() or not np.isfinite(a).all() or (h < 0).any() or (a < 0).any():
        raise ValueError("xG input missing, negative or nonfinite")
    balance = np.exp(-np.abs(h-a) / BALANCE_SCALE)
    low_goals = np.clip((LOW_GOALS_CEILING - (h+a))/LOW_GOALS_SPAN, 0.0, 1.0)
    signal = balance * (0.5 + 0.5*low_goals)
    old_d = blend[:, 1]
    draw = np.clip(old_d + DRAW_ADJUSTMENT*(signal-NEUTRAL_SIGNAL), 0.02, 0.60)
    remaining = 1.0-draw
    ratio = blend[:, 0] / (blend[:, 0]+blend[:, 2])
    out = np.column_stack((remaining*ratio, draw, remaining*(1-ratio)))
    if not np.allclose(out.sum(axis=1), 1, atol=1e-12):
        raise AssertionError("Draw adjustment broke probability mass")
    return out, signal


def _validate_match_keys(data: pd.DataFrame, name: str, n: int) -> None:
    if len(data) != n or not set(KEY).issubset(data.columns):
        raise ValueError(f"{name}: expected {n} uniquely identified fixtures")
    if data[KEY].isna().any().any() or data.duplicated(KEY).any():
        raise ValueError(f"{name}: missing or duplicate fixture keys")


def load_development(root: Path) -> pd.DataFrame:
    """GW1-5 retrospective odds comparison, excluded from prospective scorecard."""
    path = root/"data/live/challengers/opening_market_vs_model2_gw01_gw05_matches.csv"
    df = pd.read_csv(path)
    _validate_match_keys(df, "development", 50)
    if set(df["Gameweek"].unique()) != set(range(1, 6)):
        raise ValueError("Wrong development gameweeks")
    model = _probabilities(df[["Model_H", "Model_D", "Model_A"]].to_numpy())
    market = _probabilities(df[["Market_H", "Market_D", "Market_A"]].to_numpy())
    out = df[KEY+["ExpectedHomeGoals","ExpectedAwayGoals","Actual"]].copy()
    out[BASE] = 0.40*model+0.60*market
    out["Provenance"] = "retrospective_opening_odds_comparison"
    return out


def load_holdout(root: Path) -> pd.DataFrame:
    """GW6-11 point-in-time official forecasts with separately verified results."""
    frames = []
    for gw in range(6, 12):
        prefix = f"2026_27_gw{gw:02d}"
        frozen = pd.read_csv(root/"data/live/challenger_predictions"/f"{prefix}_market_shadow_predictions.csv")
        model = pd.read_csv(root/"data/live/predictions"/f"{prefix}_predictions.csv")
        results = pd.read_csv(root/"data/live/results"/f"{prefix}_results.csv")
        for name, x in (("frozen", frozen), ("model",model), ("results",results)):
            _validate_match_keys(x, f"GW{gw} {name}", 10)
            if not x["Gameweek"].eq(gw).all():
                raise ValueError(f"GW{gw} {name}: incorrect week")
        if not frozen["SnapshotType"].eq("OFFICIAL").all():
            raise ValueError(f"GW{gw} snapshot not officially frozen")
        if not results["Status"].eq("FINISHED").all():
            raise ValueError(f"GW{gw} results not verified final")
        if not results["ActualResultCode"].isin(OUTCOME).all():
            raise ValueError(f"GW{gw} invalid outcome code")
        g_h = results["ActualHomeGoals"].to_numpy(dtype=int)
        g_a = results["ActualAwayGoals"].to_numpy(dtype=int)
        expected = np.where(g_h>g_a,"H",np.where(g_h<g_a,"A","D"))
        if not np.array_equal(expected, results["ActualResultCode"].to_numpy()):
            raise ValueError(f"GW{gw} score/outcome mismatch")
        merged = frozen[KEY+BASE].merge(
            model[KEY+["ExpectedHomeGoals","ExpectedAwayGoals"]], on=KEY,
            how="outer", validate="one_to_one", indicator=True)
        if not merged["_merge"].eq("both").all() or len(merged)!=10:
            raise ValueError(f"GW{gw} model/frozen fixture identities mismatch")
        merged = merged.drop(columns="_merge").merge(
            results[KEY+["ActualResultCode"]], on=KEY,
            how="outer", validate="one_to_one", indicator=True)
        if not merged["_merge"].eq("both").all() or len(merged)!=10:
            raise ValueError(f"GW{gw} final results fixture identities mismatch")
        merged = merged.drop(columns="_merge").rename(columns={"ActualResultCode":"Actual"})
        _probabilities(merged[BASE].to_numpy(), tolerance=1e-6)
        merged["Provenance"] = "official_frozen_prematch"
        frames.append(merged)
    all_rows = pd.concat(frames, ignore_index=True)
    _validate_match_keys(all_rows, "holdout", 60)
    return all_rows


def score(data: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_match_keys(data, label, len(data))
    if not data["Actual"].isin(OUTCOME).all():
        raise ValueError("Invalid observed outcome")
    actual = data["Actual"].map(OUTCOME).to_numpy(dtype=int)
    base = _probabilities(data[BASE].to_numpy(), tolerance=1e-6)
    h03, signal = adjust_draws(base, data["ExpectedHomeGoals"].to_numpy(),
                              data["ExpectedAwayGoals"].to_numpy())
    scored = []
    summaries = []
    for name, p in (("Fixed_40_60",base),(VERSION,h03)):
        pick = p.argmax(axis=1)
        hits = (pick==actual).astype(int)
        onehot = np.eye(3)[actual]
        ll = -np.log(np.maximum(p[np.arange(len(p)), actual], 1e-15))
        brier = ((p-onehot)**2).sum(axis=1)
        draw_truth = (actual==1).astype(int)
        draw_pred = (pick==1).astype(int)
        x = data[KEY+["Actual","Provenance"]].copy()
        x["Approach"] = name
        x["Predicted"] = LABELS[pick]
        x["Correct"] = hits
        x["DrawProbability"] = p[:,1]
        x["DrawSignal"] = signal
        x["LogLoss"] = ll
        x["Brier"] = brier
        scored.append(x)
        tp = int((draw_pred & draw_truth).sum())
        predicted = int(draw_pred.sum())
        actual_draws = int(draw_truth.sum())
        summaries.append(dict(
            Split=label, Approach=name, Matches=len(data),
            Correct=int(hits.sum()), Accuracy=float(hits.mean()),
            ActualDraws=actual_draws, PredictedDraws=predicted,
            DrawRecall=(tp/actual_draws if actual_draws else np.nan),
            DrawPrecision=(tp/predicted if predicted else np.nan),
            MeanPredictedDrawProbability=float(p[:,1].mean()),
            DrawCalibrationGap=float(p[:,1].mean()-draw_truth.mean()),
            LogLoss=float(ll.mean()), Brier=float(brier.mean())))
    return pd.concat(scored, ignore_index=True), pd.DataFrame(summaries)


def run(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    development = load_development(root)
    holdout = load_holdout(root)
    dev_f, dev_s = score(development,"GW1-5 retrospective development")
    test_f, test_s = score(holdout,"GW6-11 official held-out prospective")
    return pd.concat((dev_f,test_f),ignore_index=True),pd.concat((dev_s,test_s),ignore_index=True)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=ROOT/"artifacts/h03")
    args=parser.parse_args()
    fixtures,summary=run(ROOT)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fixtures.to_csv(args.output_dir/"h03_fixture_scores.csv",index=False)
    summary.to_csv(args.output_dir/"h03_summary.csv",index=False)
    print(summary.to_string(index=False),flush=True)
    print("H03 EXPERIMENTAL only; 50 retrospective development + 60 frozen hold-out.")
    print("H03 does not verify the 1100-match +2pp historical hurdle or promote a model.")


if __name__ == "__main__":
    main()
