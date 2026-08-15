from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss


# --------------------------------------------------
# Project setup
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


from prediction.poisson_predictor import (
    predict_match
)


# --------------------------------------------------
# Files
# --------------------------------------------------

INPUT_FILE = "data/processed/prediction_features.csv"
OUTPUT_FILE = "data/processed/poisson_backtest_results.csv"


print("\nPOISSON MODEL BACKTEST")
print("======================")


# --------------------------------------------------
# Load historical feature data
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df = df.sort_values(
    "Date"
).reset_index(drop=True)


# --------------------------------------------------
# Helper: actual result
# --------------------------------------------------

def get_actual_result(
    home_goals,
    away_goals
):
    if home_goals > away_goals:
        return "H"

    if home_goals == away_goals:
        return "D"

    return "A"


# --------------------------------------------------
# Helper: predicted result
# --------------------------------------------------

def get_predicted_result(
    home_probability,
    draw_probability,
    away_probability
):
    probabilities = {
        "H": home_probability,
        "D": draw_probability,
        "A": away_probability,
    }

    return max(
        probabilities,
        key=probabilities.get
    )


# --------------------------------------------------
# Back-test every match
# --------------------------------------------------

results = []


for index, row in df.iterrows():

    feature_values = {
        "HomeRecentGoalsFor":
            row["HomeRecentGoalsFor"],

        "HomeRecentGoalsAgainst":
            row["HomeRecentGoalsAgainst"],

        "HomeRecentPPG":
            row["HomeRecentPPG"],

        "AwayRecentGoalsFor":
            row["AwayRecentGoalsFor"],

        "AwayRecentGoalsAgainst":
            row["AwayRecentGoalsAgainst"],

        "AwayRecentPPG":
            row["AwayRecentPPG"],
    }


    prediction = predict_match(
        feature_values
    )


    home_probability = (
        prediction[
            "home_win_probability"
        ] / 100
    )

    draw_probability = (
        prediction[
            "draw_probability"
        ] / 100
    )

    away_probability = (
        prediction[
            "away_win_probability"
        ] / 100
    )


    actual_result = get_actual_result(
        row["HomeGoals"],
        row["AwayGoals"]
    )


    predicted_result = get_predicted_result(
        home_probability,
        draw_probability,
        away_probability
    )


    correct = (
        actual_result == predicted_result
    )


    results.append(
        {
            "Season": row["Season"],
            "Date": row["Date"],
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"],

            "HomeGoals": row["HomeGoals"],
            "AwayGoals": row["AwayGoals"],

            "ActualResult": actual_result,
            "PredictedResult": predicted_result,

            "HomeProbability": home_probability,
            "DrawProbability": draw_probability,
            "AwayProbability": away_probability,

            "ExpectedHomeGoals":
                prediction[
                    "expected_home_goals"
                ],

            "ExpectedAwayGoals":
                prediction[
                    "expected_away_goals"
                ],

            "Correct": correct,
        }
    )


backtest = pd.DataFrame(
    results
)


# --------------------------------------------------
# Overall accuracy
# --------------------------------------------------

accuracy = backtest[
    "Correct"
].mean()


print(
    f"\nMatches tested: {len(backtest)}"
)

print(
    f"1X2 accuracy: {accuracy:.3f}"
)

print(
    f"1X2 accuracy %: {accuracy * 100:.1f}%"
)


# --------------------------------------------------
# Log loss
# --------------------------------------------------

result_to_number = {
    "H": 0,
    "D": 1,
    "A": 2,
}


y_true = backtest[
    "ActualResult"
].map(
    result_to_number
)


y_prob = backtest[
    [
        "HomeProbability",
        "DrawProbability",
        "AwayProbability",
    ]
].to_numpy()


model_log_loss = log_loss(
    y_true,
    y_prob,
    labels=[0, 1, 2],
)


print(
    f"\nLog loss: {model_log_loss:.4f}"
)


# --------------------------------------------------
# Multiclass Brier score
# --------------------------------------------------

actual_matrix = np.zeros(
    (
        len(backtest),
        3
    )
)


for i, result in enumerate(
    backtest["ActualResult"]
):

    if result == "H":
        actual_matrix[i, 0] = 1

    elif result == "D":
        actual_matrix[i, 1] = 1

    else:
        actual_matrix[i, 2] = 1


brier_score = np.mean(
    np.sum(
        (
            y_prob
            -
            actual_matrix
        ) ** 2,
        axis=1
    )
)


print(
    f"Brier score: {brier_score:.4f}"
)


# --------------------------------------------------
# Simple baseline
# --------------------------------------------------

home_rate = (
    backtest["ActualResult"] == "H"
).mean()

draw_rate = (
    backtest["ActualResult"] == "D"
).mean()

away_rate = (
    backtest["ActualResult"] == "A"
).mean()


baseline_probabilities = np.tile(
    [
        home_rate,
        draw_rate,
        away_rate
    ],
    (
        len(backtest),
        1
    )
)


baseline_log_loss = log_loss(
    y_true,
    baseline_probabilities,
    labels=[0, 1, 2],
)


baseline_brier = np.mean(
    np.sum(
        (
            baseline_probabilities
            -
            actual_matrix
        ) ** 2,
        axis=1
    )
)


baseline_prediction = max(
    {
        "H": home_rate,
        "D": draw_rate,
        "A": away_rate,
    },
    key={
        "H": home_rate,
        "D": draw_rate,
        "A": away_rate,
    }.get
)


baseline_accuracy = (
    backtest["ActualResult"]
    ==
    baseline_prediction
).mean()


print("\nBASELINE")
print("========")

print(
    f"Home rate: {home_rate:.3f}"
)

print(
    f"Draw rate: {draw_rate:.3f}"
)

print(
    f"Away rate: {away_rate:.3f}"
)

print(
    f"Baseline prediction: {baseline_prediction}"
)

print(
    f"Baseline accuracy: "
    f"{baseline_accuracy * 100:.1f}%"
)

print(
    f"Baseline log loss: "
    f"{baseline_log_loss:.4f}"
)

print(
    f"Baseline Brier score: "
    f"{baseline_brier:.4f}"
)


# --------------------------------------------------
# Performance by season
# --------------------------------------------------

print("\nPERFORMANCE BY SEASON")
print("=====================")


season_results = (
    backtest
    .groupby("Season")
    .agg(
        Matches=("Correct", "count"),
        Accuracy=("Correct", "mean"),
    )
    .reset_index()
)


season_results[
    "AccuracyPct"
] = (
    season_results[
        "Accuracy"
    ] * 100
)


print(
    season_results[
        [
            "Season",
            "Matches",
            "AccuracyPct"
        ]
    ]
)


# --------------------------------------------------
# Performance by predicted result
# --------------------------------------------------

print("\nPREDICTION DISTRIBUTION")
print("=======================")


prediction_distribution = (
    backtest[
        "PredictedResult"
    ]
    .value_counts()
    .rename_axis(
        "PredictedResult"
    )
    .reset_index(
        name="Count"
    )
)


print(
    prediction_distribution
)


# --------------------------------------------------
# Accuracy by predicted result
# --------------------------------------------------

print("\nACCURACY BY PREDICTED RESULT")
print("============================")


accuracy_by_prediction = (
    backtest
    .groupby(
        "PredictedResult"
    )
    .agg(
        Predictions=("Correct", "count"),
        Accuracy=("Correct", "mean"),
    )
    .reset_index()
)


accuracy_by_prediction[
    "AccuracyPct"
] = (
    accuracy_by_prediction[
        "Accuracy"
    ] * 100
)


print(
    accuracy_by_prediction[
        [
            "PredictedResult",
            "Predictions",
            "AccuracyPct",
        ]
    ]
)


# --------------------------------------------------
# Probability calibration buckets
# --------------------------------------------------

print("\nHOME WIN CALIBRATION")
print("====================")


backtest[
    "HomeProbabilityBucket"
] = pd.cut(
    backtest[
        "HomeProbability"
    ],
    bins=[
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ],
    include_lowest=True,
)


home_calibration = (
    backtest
    .groupby(
        "HomeProbabilityBucket",
        observed=False
    )
    .agg(
        Matches=("ActualResult", "count"),
        AveragePredictedProbability=(
            "HomeProbability",
            "mean"
        ),
        ActualHomeWinRate=(
            "ActualResult",
            lambda values: (
                values == "H"
            ).mean()
        ),
    )
    .reset_index()
)


home_calibration[
    "AveragePredictedPct"
] = (
    home_calibration[
        "AveragePredictedProbability"
    ] * 100
)


home_calibration[
    "ActualHomeWinPct"
] = (
    home_calibration[
        "ActualHomeWinRate"
    ] * 100
)


print(
    home_calibration[
        [
            "HomeProbabilityBucket",
            "Matches",
            "AveragePredictedPct",
            "ActualHomeWinPct",
        ]
    ]
)


# --------------------------------------------------
# Save detailed match predictions
# --------------------------------------------------

Path(
    "data/processed"
).mkdir(
    parents=True,
    exist_ok=True
)


backtest.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nDetailed backtest saved to: "
    f"{OUTPUT_FILE}"
)


# --------------------------------------------------
# Final model comparison
# --------------------------------------------------

print("\nMODEL VS BASELINE")
print("=================")


print(
    f"Model accuracy:    "
    f"{accuracy * 100:.1f}%"
)

print(
    f"Baseline accuracy: "
    f"{baseline_accuracy * 100:.1f}%"
)


print(
    f"\nModel log loss:    "
    f"{model_log_loss:.4f}"
)

print(
    f"Baseline log loss: "
    f"{baseline_log_loss:.4f}"
)


print(
    f"\nModel Brier:       "
    f"{brier_score:.4f}"
)

print(
    f"Baseline Brier:    "
    f"{baseline_brier:.4f}"
)


if (
    model_log_loss
    <
    baseline_log_loss
):

    print(
        "\nGOOD: Model log loss "
        "beats baseline."
    )

else:

    print(
        "\nWARNING: Model log loss "
        "does not beat baseline."
    )