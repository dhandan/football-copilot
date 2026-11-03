"""H08 research only: regularised adjustments to opening market probabilities.

Reuses the existing market and feature builders.  Never reads live GW snapshots.
A market probability is an *offset*, not an input to a standalone 1X2 model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
MARKET = ROOT / "data/processed/market_probabilities.csv"
FEATURES = ROOT / "data/processed/prediction_features_v3.csv"
OUTPUT = ROOT / "experiments/h08_market_residual/h08_out_of_time_predictions.csv"

# Deliberately reuse existing pre-match, strictly prior-date football features.
# EloDifference is EXCLUDED: build_features_v3 updates Elo sequentially for
# same-date fixtures and can therefore leak earlier results from that date.
FEATURE_COLS = [
    "RecentPPGDifference",
    "TenMatchPPGDifference",
    "SeasonPPGDifference",
    "WeightedAttackDifference",
    "WeightedDefenceDifference",
    "HomeHistoryGames10",
    "AwayHistoryGames10",
]
PROB_COLS = ["MarketHomeProbability", "MarketDrawProbability", "MarketAwayProbability"]
KEY = ["Season", "Date", "HomeTeam", "AwayTeam"]
LABELS = {"H": 0, "D": 1, "A": 2}
TRAIN = ("2021/22", "2022/23")
TUNE = "2023/24"
TEST = ("2024/25", "2025/26")
REGULARISATION = (0.1, 1.0, 10.0, 100.0)


def load_data() -> pd.DataFrame:
    for path, build_command in [
        (MARKET, "python prediction/build_market_data.py && python prediction/calculate_market_probabilities.py"),
        (FEATURES, "python prediction/build_features_v3.py"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Existing pipeline: {build_command}")

    market = pd.read_csv(MARKET)
    features = pd.read_csv(FEATURES)
    for name, frame in [("market", market), ("features", features)]:
        missing = set(KEY) - set(frame)
        if missing:
            raise ValueError(f"{name} missing key columns: {sorted(missing)}")
        if frame.duplicated(KEY).any():
            raise ValueError(f"{name} has duplicate match identities")
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise").dt.normalize()

    necessary = PROB_COLS + ["FTR", "OddsSource"]
    if not set(necessary).issubset(market.columns):
        raise ValueError(f"market missing {sorted(set(necessary) - set(market.columns))}")
    if not set(FEATURE_COLS).issubset(features.columns):
        raise ValueError(f"features missing {sorted(set(FEATURE_COLS) - set(features.columns))}")

    # OddsSource must be average *pre-match* odds. Do not use closing 'C' columns.
    if not market["OddsSource"].eq("Average market odds").all():
        raise ValueError("This version requires the existing Average market odds source")
    data = market[KEY + necessary].merge(
        features[KEY + FEATURE_COLS], on=KEY, how="inner", validate="one_to_one"
    )
    if data.empty:
        raise ValueError("No matches joined: inspect date/team/season keys")
    if not data["FTR"].isin(LABELS).all():
        raise ValueError("Unknown 1X2 result code")
    if data[FEATURE_COLS].isna().any().any():
        raise ValueError("Null pre-match features; do not impute from outcomes")
    p = data[PROB_COLS].to_numpy(dtype=float)
    if not (np.isfinite(p).all() and (p > 0).all()
            and np.allclose(p.sum(axis=1), 1.0, atol=1e-6)):
        raise ValueError("Invalid market probabilities")
    if not np.isfinite(data[FEATURE_COLS].to_numpy(dtype=float)).all():
        raise ValueError("Invalid numeric football features")
    data = data.sort_values(KEY[:2] + KEY[2:]).reset_index(drop=True)
    return data


def design(train: pd.DataFrame, test: pd.DataFrame):
    scaler = StandardScaler()
    a = scaler.fit_transform(train[FEATURE_COLS].to_numpy(dtype=float))
    b = scaler.transform(test[FEATURE_COLS].to_numpy(dtype=float))
    # An intercept adjusts systematic market calibration; football features
    # adjust match-specific residuals. Both shrink toward the untouched market.
    return (
        np.column_stack([np.ones(len(train)), a]),
        np.column_stack([np.ones(len(test)), b]),
    )


def adjusted_probabilities(market: np.ndarray, x: np.ndarray, beta: np.ndarray):
    logits = np.log(market) + x @ beta
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def fit(train: pd.DataFrame, alpha: float):
    x, _ = design(train, train)
    p = train[PROB_COLS].to_numpy(dtype=float)
    y = train["FTR"].map(LABELS).to_numpy(dtype=int)
    target = np.eye(3)[y]
    n = len(train)

    def objective(flat):
        beta = flat.reshape(x.shape[1], 3)
        logits = np.log(p) + x @ beta
        pred = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        loss = np.mean(logsumexp(logits, axis=1) - logits[np.arange(n), y])
        loss += 0.5 * alpha * np.sum(beta * beta) / n
        gradient = (x.T @ (pred - target) + alpha * beta) / n
        return float(loss), gradient.ravel()

    optim = minimize(
        objective, np.zeros(x.shape[1] * 3), jac=True, method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-11},
    )
    if not optim.success:
        raise RuntimeError(f"Residual fit failed: {optim.message}")
    return optim.x.reshape(x.shape[1], 3)


def forecast(train: pd.DataFrame, test: pd.DataFrame, alpha: float):
    beta = fit(train, alpha)
    _, x_test = design(train, test)
    return adjusted_probabilities(test[PROB_COLS].to_numpy(dtype=float), x_test, beta)


def scores(y: np.ndarray, prob: np.ndarray):
    actual = np.array([LABELS[v] for v in y], dtype=int)
    target = np.eye(3)[actual]
    return {
        "matches": len(y),
        "accuracy": float(np.mean(np.argmax(prob, axis=1) == actual)),
        "log_loss": float(-np.mean(np.log(prob[np.arange(len(y)), actual]))),
        "brier": float(np.mean(np.sum((prob - target) ** 2, axis=1))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--write", action="store_true",
                        help="Save scored holdout predictions; not official GW forecasts")
    args = parser.parse_args()
    df = load_data()
    counts = df.groupby("Season", sort=True).size()
    print("H08 Market residual | historical data audit")
    print(counts.to_string())
    print(f"Joined fixtures: {len(df)} | feature columns: {len(FEATURE_COLS)}")
    required = set(TRAIN + (TUNE,) + TEST)
    if required - set(df["Season"]):
        raise ValueError(f"Missing required seasons: {sorted(required - set(df['Season']))}")
    if any(counts[s] < 200 for s in required):
        raise ValueError("Inadequate seasonal overlap (<200): check market/feature join")
    if args.audit_only:
        return

    tr = df[df["Season"].isin(TRAIN)]
    val = df[df["Season"] == TUNE]
    print("\nDevelopment-only 2023/24 regularisation selection")
    candidates = []
    for alpha in REGULARISATION:
        model = scores(val["FTR"].to_numpy(), forecast(tr, val, alpha))
        candidates.append((model["log_loss"], alpha))
        print(f"alpha={alpha:g} | LL={model['log_loss']:.5f} | Brier={model['brier']:.5f}")
    _, selected = min(candidates)
    print(f"Selected alpha={selected:g} based ONLY on 2023/24 validation")

    results = []
    for season in TEST:
        history = df[df["Season"] < season]
        holdout = df[df["Season"] == season]
        model_p = forecast(history, holdout, selected)
        market_p = holdout[PROB_COLS].to_numpy(dtype=float)
        baseline = scores(holdout["FTR"].to_numpy(), market_p)
        model = scores(holdout["FTR"].to_numpy(), model_p)
        print(f"\nCHRONOLOGICAL TEST {season} | matches={len(holdout)}")
        print(f"Market:   {baseline}")
        print(f"Residual: {model}")
        for k in range(len(holdout)):
            row = dict(zip(KEY + ["FTR"], holdout.iloc[k][KEY + ["FTR"]]))
            row["SelectedAlpha"] = selected
            for j, name in enumerate(("H", "D", "A")):
                row[f"Market_{name}"] = market_p[k, j]
                row[f"Residual_{name}"] = model_p[k, j]
            results.append(row)
    pooled = pd.DataFrame(results)
    base_p = pooled[[f"Market_{v}" for v in ("H", "D", "A")]].to_numpy()
    residual_p = pooled[[f"Residual_{v}" for v in ("H", "D", "A")]].to_numpy()
    baseline = scores(pooled["FTR"].to_numpy(), base_p)
    model = scores(pooled["FTR"].to_numpy(), residual_p)
    print("\nPOOLED CHRONOLOGICAL 2024/25 + 2025/26 (NOT pristine holdouts)")
    print("Market  ", baseline)
    print("Residual", model)
    improvement = (baseline["log_loss"] - model["log_loss"]) / baseline["log_loss"]
    print(f"Relative Log Loss improvement: {improvement:.2%}")
    print("NOTE: 40/60 comparison requires matched historical Model 2 forecasts;"
          " never substitute prospective GW6-GW9 results.")
    print("NO promotion: earlier project experiments examined these seasons; this is exploratory chronological evidence, not untouched confirmation.")
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        pooled.to_csv(OUTPUT, index=False)
        print(f"Research output: {OUTPUT}")


if __name__ == "__main__":
    main()
