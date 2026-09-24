from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path(
    "data/live/challengers/"
    "opening_market_vs_model2_gw01_gw05_matches.csv"
)

OUTPUT = Path(
    "data/live/challengers/"
    "model2_opening_market_blend_sweep_gw01_gw05.csv"
)

df = pd.read_csv(INPUT)

if len(df) != 50:
    raise ValueError(f"Expected 50 matches, found {len(df)}")

required = [
    "Actual",
    "Model_H", "Model_D", "Model_A",
    "Market_H", "Market_D", "Market_A"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

labels = np.array(["H", "D", "A"])
result_index = {"H": 0, "D": 1, "A": 2}

model = df[["Model_H", "Model_D", "Model_A"]].to_numpy(dtype=float)
market = df[["Market_H", "Market_D", "Market_A"]].to_numpy(dtype=float)
actual = df["Actual"].to_numpy()

actual_idx = np.array([result_index[x] for x in actual])

y = np.zeros((len(df), 3))
y[np.arange(len(df)), actual_idx] = 1.0

results = []

# ------------------------------------------------------------
# Sweep Model 2 weight from 100% to 0% in 5% increments
# ------------------------------------------------------------
for model_weight in np.arange(1.0, -0.001, -0.05):

    market_weight = 1.0 - model_weight

    probs = (
        model_weight * model +
        market_weight * market
    )

    # Defensive normalisation
    probs = probs / probs.sum(axis=1, keepdims=True)

    picks = labels[np.argmax(probs, axis=1)]

    accuracy = np.mean(picks == actual)

    actual_probs = probs[np.arange(len(df)), actual_idx]
    actual_probs = np.clip(actual_probs, 1e-15, 1.0)

    log_loss = -np.mean(np.log(actual_probs))

    # Same multiclass Brier definition as previous experiment
    brier = np.mean(np.sum((probs - y) ** 2, axis=1))

    results.append({
        "Model2Weight": round(model_weight, 2),
        "MarketWeight": round(market_weight, 2),
        "Accuracy": accuracy,
        "LogLoss": log_loss,
        "Brier": brier
    })

results = pd.DataFrame(results)

# ------------------------------------------------------------
# Identify best weights independently by metric
# ------------------------------------------------------------
best_log = results.loc[results["LogLoss"].idxmin()]
best_brier = results.loc[results["Brier"].idxmin()]
best_accuracy = results.loc[results["Accuracy"].idxmax()]

results.to_csv(OUTPUT, index=False)

display = results.copy()

display["Model2Weight"] = (
    display["Model2Weight"] * 100
).round().astype(int).astype(str) + "%"

display["MarketWeight"] = (
    display["MarketWeight"] * 100
).round().astype(int).astype(str) + "%"

display["Accuracy"] = (
    display["Accuracy"] * 100
).round(1).astype(str) + "%"

display["LogLoss"] = display["LogLoss"].round(4)
display["Brier"] = display["Brier"].round(4)

print("\n=== MODEL 2 + OPENING MARKET BLEND SWEEP ===")
print(display.to_string(index=False))

print("\n=== BEST BY LOG LOSS ===")
print(
    f"Model 2: {best_log['Model2Weight']:.0%} | "
    f"Market: {best_log['MarketWeight']:.0%} | "
    f"Accuracy: {best_log['Accuracy']:.1%} | "
    f"Log Loss: {best_log['LogLoss']:.4f} | "
    f"Brier: {best_log['Brier']:.4f}"
)

print("\n=== BEST BY BRIER ===")
print(
    f"Model 2: {best_brier['Model2Weight']:.0%} | "
    f"Market: {best_brier['MarketWeight']:.0%} | "
    f"Accuracy: {best_brier['Accuracy']:.1%} | "
    f"Log Loss: {best_brier['LogLoss']:.4f} | "
    f"Brier: {best_brier['Brier']:.4f}"
)

print("\n=== BEST OBSERVED ACCURACY ===")
print(
    f"Model 2: {best_accuracy['Model2Weight']:.0%} | "
    f"Market: {best_accuracy['MarketWeight']:.0%} | "
    f"Accuracy: {best_accuracy['Accuracy']:.1%} | "
    f"Log Loss: {best_accuracy['LogLoss']:.4f} | "
    f"Brier: {best_accuracy['Brier']:.4f}"
)

print("\n=== PURE BASELINES ===")

pure_model = results.loc[results["Model2Weight"].idxmax()]
pure_market = results.loc[results["Model2Weight"].idxmin()]

print(
    f"Model 2 only   | Accuracy {pure_model['Accuracy']:.1%} | "
    f"Log Loss {pure_model['LogLoss']:.4f} | "
    f"Brier {pure_model['Brier']:.4f}"
)

print(
    f"Market only    | Accuracy {pure_market['Accuracy']:.1%} | "
    f"Log Loss {pure_market['LogLoss']:.4f} | "
    f"Brier {pure_market['Brier']:.4f}"
)

print("\nSaved:")
print(OUTPUT)
