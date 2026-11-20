"""Freeze the already-registered adaptive market-confidence challenger.

No fitting, tuning or use of outcomes. Depends on the write-once Friday
Model 2 and market snapshot produced by scripts/run_gameweek_once.py.
This is always a shadow challenger, never an official champion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VERSION = "AdaptiveMarketConfidence_v1.0"
RULE = "market_weight=clip(0.60-0.60*(max_market_probability-0.50),0.20,0.90)"
KEYS = ["FixtureDate", "FixtureTime", "HomeTeam", "AwayTeam"]
MODEL_P = ["HomeWinProbability", "DrawProbability", "AwayWinProbability"]
MARKET_P = ["MarketHomeProbability", "MarketDrawProbability", "MarketAwayProbability"]


def adaptive_probabilities(model: np.ndarray, market: np.ndarray):
    """The registered GW8 rule, unchanged. Both inputs are decimals."""
    model = np.asarray(model, dtype=float)
    market = np.asarray(market, dtype=float)
    if model.ndim != 2 or model.shape[1] != 3 or model.shape != market.shape:
        raise ValueError("Expected matched 3-way probability arrays")
    if not (np.isfinite(model).all() and np.isfinite(market).all()):
        raise ValueError("Nonfinite probability")
    if (model < 0).any() or (market < 0).any():
        raise ValueError("Negative probability")
    if not (np.allclose(model.sum(axis=1), 1, atol=0.003)
            and np.allclose(market.sum(axis=1), 1, atol=1e-6)):
        raise ValueError("Input probabilities must each sum to one")
    # Model 2 percentages have 1-decimal rounding, hence normalise first.
    m = model / model.sum(axis=1, keepdims=True)
    k = market / market.sum(axis=1, keepdims=True)
    market_weight = np.clip(0.60 - 0.60 * (k.max(axis=1) - 0.50), 0.20, 0.90)
    model_weight = 1 - market_weight
    z = model_weight[:, None] * m + market_weight[:, None] * k
    if not np.allclose(z.sum(axis=1), 1, atol=1e-8):
        raise RuntimeError("Adaptive probabilities do not sum to one")
    return z, model_weight, market_weight


def build(model: pd.DataFrame, market: pd.DataFrame, gameweek: int):
    if len(model) != 10 or len(market) != 10:
        raise ValueError("GW freeze needs exactly 10 fixtures per input")
    for df, extra in ((model, MODEL_P), (market, MARKET_P)):
        if not set(KEYS + extra).issubset(df.columns):
            raise ValueError("Missing fixture/probability columns")
        if df.duplicated(KEYS).any():
            raise ValueError("Duplicate fixtures")
    if "SnapshotType" in market and not market["SnapshotType"].eq("OFFICIAL").all():
        raise ValueError("Only official market snapshot allowed")
    if "PredictionStatus" in model and not model["PredictionStatus"].eq("Predicted").all():
        raise ValueError("Only pre-match Model2 predictions allowed")
    merged = model.merge(market, on=KEYS, how="outer", validate="one_to_one",
                         indicator=True, suffixes=("", "_Market"))
    if len(merged) != 10 or not merged["_merge"].eq("both").all():
        raise ValueError("Model 2 and market fixtures do not match 10/10")
    if "Gameweek" in model and not model["Gameweek"].eq(gameweek).all():
        raise ValueError("Wrong gameweek in model freeze")
    model_probs = merged[MODEL_P].to_numpy(dtype=float) / 100
    market_probs = merged[MARKET_P].to_numpy(dtype=float)
    z, mw, kw = adaptive_probabilities(model_probs, market_probs)
    if "ShadowHomeProbability" in merged:
        fixed = .4 * (model_probs/model_probs.sum(axis=1,keepdims=True)) + .6 * market_probs
        recorded = merged[["ShadowHomeProbability","ShadowDrawProbability","ShadowAwayProbability"]].to_numpy(dtype=float)
        if not np.allclose(recorded, fixed, atol=1e-8):
            raise ValueError("Official fixed 40/60 probabilities do not reconcile")
    idx = z.argmax(axis=1)
    stamp = str(model["PredictionTimestamp"].iloc[0]) if "PredictionTimestamp" in model else ""
    out = merged[KEYS].copy()
    out.insert(0, "Gameweek", gameweek)
    out.insert(1, "ExperimentVersion", VERSION)
    out.insert(2, "ForecastRole", "ProspectiveChallenger")
    out["PredictionTimestamp"] = stamp
    out["ModelWeight"] = mw
    out["MarketWeight"] = kw
    out["HomeProbability"], out["DrawProbability"], out["AwayProbability"] = z.T
    out["PredictedResult"] = np.array(["H", "D", "A"])[idx]
    return out


def freeze(gameweek: int):
    if gameweek != 11:
        raise ValueError("This one-time release step is strictly GW11")
    prefix = f"2026_27_gw{gameweek:02d}"
    model_path = ROOT / "data/live/predictions" / f"{prefix}_predictions.csv"
    market_dir = ROOT / "data/live/market"
    market_paths = sorted(market_dir.glob(f"{prefix}_opening_market_official_*.csv"))
    output = ROOT / "data/live/challenger_predictions" / f"{prefix}_adaptive_confidence_v1.csv"
    manifest = output.with_suffix(".json")
    if output.exists() or manifest.exists():
        raise FileExistsError("GW11 adaptive challenger already frozen; no overwrite")
    if not model_path.is_file() or len(market_paths) != 1:
        raise RuntimeError("Need one official Model2 file and exactly one official market snapshot")
    market_path = market_paths[0]
    model = pd.read_csv(model_path)
    market = pd.read_csv(market_path)
    if "SnapshotTimestamp" not in market:
        raise ValueError("Market snapshot requires capture timestamp")
    out = build(model, market, gameweek)
    mtime = pd.Timestamp(model["PredictionTimestamp"].iloc[0])
    captures = pd.to_datetime(market["SnapshotTimestamp"], utc=True)
    kickoffs = pd.to_datetime(model["KickoffUTC"], utc=True)
    if mtime.tzinfo is None:
        raise ValueError("Model capture timestamp must have timezone")
    if not ((mtime < kickoffs).all() and
            (captures.to_numpy() < kickoffs.to_numpy()).all()):
        raise ValueError("Capture must precede every fixture kickoff")
    meta = {
        "version": VERSION, "gameweek": gameweek,
        "status": "ProspectiveChallenger",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_captured_utc": str(mtime),
        "original_registration": "2026-10-23T09:43:21.823146+00:00",
        "rule": RULE,
        "model_source": str(model_path.relative_to(ROOT)),
        "market_source": str(market_path.relative_to(ROOT)),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "market_sha256": hashlib.sha256(market_path.read_bytes()).hexdigest(),
        "fixtures": len(out),
        "note": "Registered adaptive formula unchanged; not promoted."
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        out.to_csv(f, index=False)
    with manifest.open("x", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"GW11 ADAPTIVE FROZEN: 10/10 fixtures, rule unchanged, {output.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameweek", type=int, default=11)
    args = parser.parse_args()
    freeze(args.gameweek)
