from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# FILES
# ============================================================

INPUT_FILE = Path(
    "reports/model5/model5_2025_26_out_of_time.csv"
)

OUTPUT_FILE = Path(
    "reports/model5/model5_2025_26_uplift_diagnostics.csv"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="raise",
)


# ============================================================
# LABELS
# ============================================================

CLASS_LABELS = {
    0: "Home",
    1: "Draw",
    2: "Away",
}


df["Actual"] = (
    df["ActualClass"]
    .map(CLASS_LABELS)
)

df["Model2Predicted"] = (
    df["Model2Prediction"]
    .map(CLASS_LABELS)
)

df["Model5Predicted"] = (
    df["Model5Prediction"]
    .map(CLASS_LABELS)
)


# ============================================================
# CORRECTNESS
# ============================================================

df["Model2Correct"] = (
    df["Model2Prediction"]
    ==
    df["ActualClass"]
)

df["Model5Correct"] = (
    df["Model5Prediction"]
    ==
    df["ActualClass"]
)


# ============================================================
# PREDICTION SWITCHES
# ============================================================

df["PredictionChanged"] = (
    df["Model2Prediction"]
    !=
    df["Model5Prediction"]
)


changed = df[
    df["PredictionChanged"]
].copy()


changed["SwitchOutcome"] = np.select(
    [
        (
            ~changed["Model2Correct"]
            &
            changed["Model5Correct"]
        ),
        (
            changed["Model2Correct"]
            &
            ~changed["Model5Correct"]
        ),
    ],
    [
        "Improved",
        "Worsened",
    ],
    default="ChangedStillWrong",
)


# ============================================================
# CONFIDENCE
# ============================================================

M2_PROBABILITY_COLUMNS = [
    "Model2HomeProbability",
    "Model2DrawProbability",
    "Model2AwayProbability",
]

M5_PROBABILITY_COLUMNS = [
    "Model5HomeProbability",
    "Model5DrawProbability",
    "Model5AwayProbability",
]


df["Model2Confidence"] = (
    df[M2_PROBABILITY_COLUMNS]
    .max(axis=1)
)

df["Model5Confidence"] = (
    df[M5_PROBABILITY_COLUMNS]
    .max(axis=1)
)


changed["Model2Confidence"] = (
    changed[M2_PROBABILITY_COLUMNS]
    .max(axis=1)
)

changed["Model5Confidence"] = (
    changed[M5_PROBABILITY_COLUMNS]
    .max(axis=1)
)


# ============================================================
# MATCH-LEVEL LOG LOSS
# ============================================================

epsilon = 1e-15


def actual_probability(
    row,
    prefix,
):
    actual = int(
        row["ActualClass"]
    )

    columns = [
        f"{prefix}HomeProbability",
        f"{prefix}DrawProbability",
        f"{prefix}AwayProbability",
    ]

    return float(
        row[
            columns[actual]
        ]
    )


df["Model2ActualProbability"] = df.apply(
    lambda row:
        actual_probability(
            row,
            "Model2",
        ),
    axis=1,
)

df["Model5ActualProbability"] = df.apply(
    lambda row:
        actual_probability(
            row,
            "Model5",
        ),
    axis=1,
)


df["Model2MatchLogLoss"] = (
    -np.log(
        np.clip(
            df["Model2ActualProbability"],
            epsilon,
            1.0,
        )
    )
)

df["Model5MatchLogLoss"] = (
    -np.log(
        np.clip(
            df["Model5ActualProbability"],
            epsilon,
            1.0,
        )
    )
)


df["LogLossImprovement"] = (
    df["Model2MatchLogLoss"]
    -
    df["Model5MatchLogLoss"]
)


# ============================================================
# MATCH-LEVEL BRIER
# ============================================================

actual_matrix = np.zeros(
    (
        len(df),
        3,
    )
)

actual_matrix[
    np.arange(
        len(df)
    ),
    df["ActualClass"].astype(int),
] = 1.0


m2_probabilities = (
    df[M2_PROBABILITY_COLUMNS]
    .to_numpy()
)

m5_probabilities = (
    df[M5_PROBABILITY_COLUMNS]
    .to_numpy()
)


df["Model2MatchBrier"] = np.sum(
    (
        m2_probabilities
        -
        actual_matrix
    )
    ** 2,
    axis=1,
)


df["Model5MatchBrier"] = np.sum(
    (
        m5_probabilities
        -
        actual_matrix
    )
    ** 2,
    axis=1,
)


df["BrierImprovement"] = (
    df["Model2MatchBrier"]
    -
    df["Model5MatchBrier"]
)


# ============================================================
# REBUILD CHANGED DATA AFTER DIAGNOSTICS
# ============================================================

changed = df[
    df["PredictionChanged"]
].copy()


changed["SwitchOutcome"] = np.select(
    [
        (
            ~changed["Model2Correct"]
            &
            changed["Model5Correct"]
        ),
        (
            changed["Model2Correct"]
            &
            ~changed["Model5Correct"]
        ),
    ],
    [
        "Improved",
        "Worsened",
    ],
    default="ChangedStillWrong",
)


# ============================================================
# SUMMARY
# ============================================================

total_matches = len(df)

total_changed = len(changed)

improved_switches = int(
    (
        changed["SwitchOutcome"]
        ==
        "Improved"
    ).sum()
)

worsened_switches = int(
    (
        changed["SwitchOutcome"]
        ==
        "Worsened"
    ).sum()
)

still_wrong_switches = int(
    (
        changed["SwitchOutcome"]
        ==
        "ChangedStillWrong"
    ).sum()
)


m2_correct = int(
    df["Model2Correct"].sum()
)

m5_correct = int(
    df["Model5Correct"].sum()
)


better_logloss = int(
    (
        df["LogLossImprovement"]
        >
        0
    ).sum()
)

better_brier = int(
    (
        df["BrierImprovement"]
        >
        0
    ).sum()
)


# ============================================================
# PRINT CORE DIAGNOSTICS
# ============================================================

print()
print("FOOTBALL COPILOT")
print("MODEL 5 2025/26 UPLIFT DIAGNOSTICS")
print("===================================")

print()
print("CLASSIFICATION")
print("==============")

print(
    f"Matches: {total_matches}"
)

print(
    f"Model 2 correct: "
    f"{m2_correct}/{total_matches} "
    f"({m2_correct / total_matches:.4%})"
)

print(
    f"Model 5 correct: "
    f"{m5_correct}/{total_matches} "
    f"({m5_correct / total_matches:.4%})"
)

print(
    f"Net correct-match change: "
    f"{m5_correct - m2_correct:+d}"
)


print()
print("PREDICTION SWITCHES")
print("===================")

print(
    f"Predictions changed: "
    f"{total_changed}/{total_matches} "
    f"({total_changed / total_matches:.4%})"
)

print(
    "Improved switches:",
    improved_switches,
)

print(
    "Worsened switches:",
    worsened_switches,
)

print(
    "Changed but still wrong:",
    still_wrong_switches,
)

print(
    "Net switch effect:",
    improved_switches
    -
    worsened_switches,
)


# ============================================================
# SWITCH DIRECTIONS
# ============================================================

print()
print("SWITCH DIRECTIONS")
print("=================")

if total_changed:

    switch_table = pd.crosstab(
        changed["Model2Predicted"],
        changed["Model5Predicted"],
    )

    print(
        switch_table.to_string()
    )


# ============================================================
# SWITCH OUTCOME BY DIRECTION
# ============================================================

print()
print("SWITCH OUTCOME BY DIRECTION")
print("===========================")

if total_changed:

    switch_outcome_table = (
        changed
        .groupby(
            [
                "Model2Predicted",
                "Model5Predicted",
                "SwitchOutcome",
            ]
        )
        .size()
        .reset_index(
            name="Matches"
        )
        .sort_values(
            [
                "Matches",
                "Model2Predicted",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    print(
        switch_outcome_table.to_string(
            index=False
        )
    )


# ============================================================
# CONFIDENCE OF SWITCHED FIXTURES
# ============================================================

print()
print("SWITCH CONFIDENCE")
print("=================")

if total_changed:

    print(
        f"Mean Model 2 confidence: "
        f"{changed['Model2Confidence'].mean():.4%}"
    )

    print(
        f"Mean Model 5 confidence: "
        f"{changed['Model5Confidence'].mean():.4%}"
    )

    print(
        "Model 2 switched fixtures <50% confidence:",
        int(
            (
                changed[
                    "Model2Confidence"
                ]
                <
                0.50
            ).sum()
        ),
        "/",
        total_changed,
    )

    print(
        "Model 2 switched fixtures <45% confidence:",
        int(
            (
                changed[
                    "Model2Confidence"
                ]
                <
                0.45
            ).sum()
        ),
        "/",
        total_changed,
    )


# ============================================================
# PROBABILITY QUALITY
# ============================================================

print()
print("PROBABILITY QUALITY")
print("===================")

print(
    f"Model 5 better match-level Log Loss: "
    f"{better_logloss}/{total_matches} "
    f"({better_logloss / total_matches:.4%})"
)

print(
    f"Model 5 better match-level Brier: "
    f"{better_brier}/{total_matches} "
    f"({better_brier / total_matches:.4%})"
)

print(
    f"Mean match-level Log Loss improvement: "
    f"{df['LogLossImprovement'].mean():+.6f}"
)

print(
    f"Mean match-level Brier improvement: "
    f"{df['BrierImprovement'].mean():+.6f}"
)


# ============================================================
# PROBABILITY QUALITY BY ACTUAL RESULT
# ============================================================

print()
print("PROBABILITY QUALITY BY ACTUAL RESULT")
print("====================================")

by_actual = (
    df
    .groupby(
        "Actual"
    )
    .agg(
        Matches=(
            "Actual",
            "size",
        ),
        MeanLogLossImprovement=(
            "LogLossImprovement",
            "mean",
        ),
        MeanBrierImprovement=(
            "BrierImprovement",
            "mean",
        ),
    )
    .reset_index()
)


print(
    by_actual.to_string(
        index=False,
        formatters={
            "MeanLogLossImprovement":
                "{:+.6f}".format,

            "MeanBrierImprovement":
                "{:+.6f}".format,
        },
    )
)


# ============================================================
# BIGGEST IMPROVEMENTS / DETERIORATIONS
# ============================================================

display_columns = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "HomeGoals",
    "AwayGoals",
    "Actual",
    "Model2Predicted",
    "Model5Predicted",
    "Model2ActualProbability",
    "Model5ActualProbability",
    "LogLossImprovement",
]


print()
print("TOP 10 LOG LOSS IMPROVEMENTS")
print("============================")

print(
    df
    .sort_values(
        "LogLossImprovement",
        ascending=False,
    )[
        display_columns
    ]
    .head(10)
    .to_string(
        index=False,
        formatters={
            "Model2ActualProbability":
                "{:.4%}".format,

            "Model5ActualProbability":
                "{:.4%}".format,

            "LogLossImprovement":
                "{:+.6f}".format,
        },
    )
)


print()
print("TOP 10 LOG LOSS DETERIORATIONS")
print("==============================")

print(
    df
    .sort_values(
        "LogLossImprovement",
        ascending=True,
    )[
        display_columns
    ]
    .head(10)
    .to_string(
        index=False,
        formatters={
            "Model2ActualProbability":
                "{:.4%}".format,

            "Model5ActualProbability":
                "{:.4%}".format,

            "LogLossImprovement":
                "{:+.6f}".format,
        },
    )
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print(
    "Saved:",
    OUTPUT_FILE
)

print()
print("DIAGNOSTIC COMPLETE")
print("===================")