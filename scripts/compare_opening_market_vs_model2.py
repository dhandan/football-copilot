from pathlib import Path
import glob
import numpy as np
import pandas as pd

MARKET_FILE = Path("data/raw/E0_2627.csv")
PRED_PATTERN = "data/live/predictions/2026_27_gw0[1-5]_predictions.csv"
OUTPUT_DIR = Path("data/live/challengers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Football-Data -> Football Copilot naming
TEAM_MAP = {
    "Coventry": "Coventry City",
    "Hull": "Hull City",
}

# ------------------------------------------------------------
# Load frozen Model 2 predictions
# ------------------------------------------------------------
pred_files = sorted(glob.glob(PRED_PATTERN))

if len(pred_files) != 5:
    raise ValueError(f"Expected 5 prediction files, found {len(pred_files)}")

pred = pd.concat(
    [pd.read_csv(f) for f in pred_files],
    ignore_index=True
)

if len(pred) != 50:
    raise ValueError(f"Expected 50 frozen predictions, found {len(pred)}")

if not (pred["ModelVersion"] == "Model2_v1.0").all():
    raise ValueError(
        "Unexpected model version found:\n"
        + pred["ModelVersion"].value_counts().to_string()
    )

pred["FixtureDate"] = pd.to_datetime(pred["FixtureDate"]).dt.normalize()

# Model probabilities are stored as percentages
pred["Model_H"] = pred["HomeWinProbability"] / 100.0
pred["Model_D"] = pred["DrawProbability"] / 100.0
pred["Model_A"] = pred["AwayWinProbability"] / 100.0

# Tiny rounding correction because source percentages are 1dp
model_total = pred[["Model_H", "Model_D", "Model_A"]].sum(axis=1)
pred["Model_H"] /= model_total
pred["Model_D"] /= model_total
pred["Model_A"] /= model_total

# ------------------------------------------------------------
# Load Football-Data Opening Market
# ------------------------------------------------------------
market = pd.read_csv(MARKET_FILE)

if len(market) != 50:
    raise ValueError(f"Expected 50 market matches, found {len(market)}")

required = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "AvgH", "AvgD", "AvgA"
]

missing = [c for c in required if c not in market.columns]
if missing:
    raise ValueError(f"Missing market columns: {missing}")

market = market[required].copy()

market["FixtureDate"] = pd.to_datetime(
    market["Date"],
    format="%d/%m/%Y"
).dt.normalize()

market["HomeTeam"] = market["HomeTeam"].replace(TEAM_MAP)
market["AwayTeam"] = market["AwayTeam"].replace(TEAM_MAP)

# ------------------------------------------------------------
# Convert opening decimal odds to no-vig probabilities
# ------------------------------------------------------------
market["Raw_H"] = 1.0 / market["AvgH"]
market["Raw_D"] = 1.0 / market["AvgD"]
market["Raw_A"] = 1.0 / market["AvgA"]

market["MarketOverround"] = (
    market["Raw_H"] +
    market["Raw_D"] +
    market["Raw_A"]
)

market["Market_H"] = market["Raw_H"] / market["MarketOverround"]
market["Market_D"] = market["Raw_D"] / market["MarketOverround"]
market["Market_A"] = market["Raw_A"] / market["MarketOverround"]

# ------------------------------------------------------------
# Join by date + home + away
# ------------------------------------------------------------
compare = pred.merge(
    market,
    on=["FixtureDate", "HomeTeam", "AwayTeam"],
    how="left",
    validate="one_to_one",
    indicator=True
)

unmatched = compare[compare["_merge"] != "both"]

print("\n=== JOIN CHECK ===")
print("Frozen Model 2 predictions:", len(pred))
print("Opening Market matches:", len(market))
print("Matched:", (compare["_merge"] == "both").sum())
print("Unmatched:", len(unmatched))

if len(unmatched):
    print("\nUNMATCHED FIXTURES:")
    print(
        unmatched[
            ["Gameweek", "FixtureDate", "HomeTeam", "AwayTeam"]
        ].to_string(index=False)
    )
    raise ValueError("Not all fixtures matched. Scoring stopped.")

compare.drop(columns="_merge", inplace=True)

# ------------------------------------------------------------
# Actual result
# ------------------------------------------------------------
result_index = {"H": 0, "D": 1, "A": 2}
labels = np.array(["H", "D", "A"])

compare["Actual"] = compare["FTR"]

model_probs = compare[["Model_H", "Model_D", "Model_A"]].to_numpy()
market_probs = compare[["Market_H", "Market_D", "Market_A"]].to_numpy()

compare["ModelPick"] = labels[np.argmax(model_probs, axis=1)]
compare["MarketPick"] = labels[np.argmax(market_probs, axis=1)]

compare["ModelCorrect"] = compare["ModelPick"] == compare["Actual"]
compare["MarketCorrect"] = compare["MarketPick"] == compare["Actual"]

# ------------------------------------------------------------
# Match-level Log Loss and multiclass Brier
# Brier = sum of squared probability error across H/D/A
# ------------------------------------------------------------
def match_scores(row, prefix):
    actual_idx = result_index[row["Actual"]]

    p = np.array([
        row[f"{prefix}_H"],
        row[f"{prefix}_D"],
        row[f"{prefix}_A"]
    ], dtype=float)

    p = np.clip(p, 1e-15, 1.0)

    log_loss = -np.log(p[actual_idx])

    y = np.zeros(3)
    y[actual_idx] = 1.0

    brier = np.sum((p - y) ** 2)

    return pd.Series({
        f"{prefix}LogLoss": log_loss,
        f"{prefix}Brier": brier
    })

compare = pd.concat(
    [
        compare,
        compare.apply(lambda r: match_scores(r, "Model"), axis=1),
        compare.apply(lambda r: match_scores(r, "Market"), axis=1)
    ],
    axis=1
)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
def summarise(df):
    return pd.Series({
        "Matches": len(df),

        "Model2_Accuracy":
            df["ModelCorrect"].mean(),

        "OpeningMarket_Accuracy":
            df["MarketCorrect"].mean(),

        "Model2_LogLoss":
            df["ModelLogLoss"].mean(),

        "OpeningMarket_LogLoss":
            df["MarketLogLoss"].mean(),

        "Model2_Brier":
            df["ModelBrier"].mean(),

        "OpeningMarket_Brier":
            df["MarketBrier"].mean(),
    })

by_gw = compare.groupby("Gameweek").apply(
    summarise,
    include_groups=False
).reset_index()

overall = summarise(compare).to_frame().T
overall.insert(0, "Gameweek", "GW1-GW5")

# ------------------------------------------------------------
# Save auditable outputs
# ------------------------------------------------------------
match_output = OUTPUT_DIR / "opening_market_vs_model2_gw01_gw05_matches.csv"
summary_output = OUTPUT_DIR / "opening_market_vs_model2_gw01_gw05_summary.csv"

compare.to_csv(match_output, index=False)

summary = pd.concat([by_gw, overall], ignore_index=True)
summary.to_csv(summary_output, index=False)

# ------------------------------------------------------------
# Display
# ------------------------------------------------------------
display = summary.copy()

for c in ["Model2_Accuracy", "OpeningMarket_Accuracy"]:
    display[c] = (display[c] * 100).round(1).astype(str) + "%"

for c in [
    "Model2_LogLoss", "OpeningMarket_LogLoss",
    "Model2_Brier", "OpeningMarket_Brier"
]:
    display[c] = display[c].astype(float).round(4)

print("\n=== MODEL 2 vs OPENING MARKET ===")
print(display.to_string(index=False))

print("\n=== PICK DISAGREEMENTS ===")

disagreements = compare[
    compare["ModelPick"] != compare["MarketPick"]
][[
    "Gameweek",
    "FixtureDate",
    "HomeTeam",
    "AwayTeam",
    "Actual",
    "ModelPick",
    "MarketPick",
    "Model_H",
    "Model_D",
    "Model_A",
    "Market_H",
    "Market_D",
    "Market_A"
]]

print("Disagreements:", len(disagreements))

if len(disagreements):
    print(disagreements.to_string(index=False))

print("\nSaved:")
print(match_output)
print(summary_output)
