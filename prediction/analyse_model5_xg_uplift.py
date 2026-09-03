from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# FILES
# ==================================================

INPUT_FILE = Path(
    "reports/model5/"
    "model5_xg_enriched_backtest.csv"
)

OUTPUT_TEAM = Path(
    "reports/model5/"
    "model5_uplift_by_team.csv"
)

OUTPUT_CONFIDENCE = Path(
    "reports/model5/"
    "model5_uplift_by_confidence.csv"
)

OUTPUT_SWITCHES = Path(
    "reports/model5/"
    "model5_prediction_switches.csv"
)


# ==================================================
# LOAD
# ==================================================

print()
print("FOOTBALL COPILOT")
print("MODEL 5 XG UPLIFT ANALYSIS")
print("==========================")
print()

df = pd.read_csv(
    INPUT_FILE
)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce",
)


# ==================================================
# HELPERS
# ==================================================

RESULT_TO_NUMBER = {
    "H": 0,
    "D": 1,
    "A": 2,
}


def multiclass_brier_row(
    actual_result,
    home_probability,
    draw_probability,
    away_probability,
):

    actual_numeric = RESULT_TO_NUMBER[
        actual_result
    ]

    probabilities = np.array(
        [
            home_probability,
            draw_probability,
            away_probability,
        ],
        dtype=float,
    )

    target = np.zeros(
        3,
        dtype=float,
    )

    target[
        actual_numeric
    ] = 1.0

    return float(
        np.sum(
            (
                probabilities
                -
                target
            )
            ** 2
        )
    )


def log_loss_row(
    actual_result,
    home_probability,
    draw_probability,
    away_probability,
):

    actual_numeric = RESULT_TO_NUMBER[
        actual_result
    ]

    probabilities = np.array(
        [
            home_probability,
            draw_probability,
            away_probability,
        ],
        dtype=float,
    )

    probability = probabilities[
        actual_numeric
    ]

    probability = np.clip(
        probability,
        1e-15,
        1.0,
    )

    return float(
        -np.log(
            probability
        )
    )


# ==================================================
# MATCH-LEVEL QUALITY
# ==================================================

df["Model2Correct"] = (
    df["Model2Prediction"]
    ==
    df["ActualResult"]
)

df["Model5Correct"] = (
    df["Model5Prediction"]
    ==
    df["ActualResult"]
)


df["Model2LogLossRow"] = df.apply(
    lambda row: log_loss_row(
        row["ActualResult"],
        row["Model2HomeProbability"],
        row["Model2DrawProbability"],
        row["Model2AwayProbability"],
    ),
    axis=1,
)

df["Model5LogLossRow"] = df.apply(
    lambda row: log_loss_row(
        row["ActualResult"],
        row["Model5HomeProbability"],
        row["Model5DrawProbability"],
        row["Model5AwayProbability"],
    ),
    axis=1,
)


df["Model2BrierRow"] = df.apply(
    lambda row: multiclass_brier_row(
        row["ActualResult"],
        row["Model2HomeProbability"],
        row["Model2DrawProbability"],
        row["Model2AwayProbability"],
    ),
    axis=1,
)

df["Model5BrierRow"] = df.apply(
    lambda row: multiclass_brier_row(
        row["ActualResult"],
        row["Model5HomeProbability"],
        row["Model5DrawProbability"],
        row["Model5AwayProbability"],
    ),
    axis=1,
)


df["LogLossImprovement"] = (
    df["Model2LogLossRow"]
    -
    df["Model5LogLossRow"]
)

df["BrierImprovement"] = (
    df["Model2BrierRow"]
    -
    df["Model5BrierRow"]
)


# ==================================================
# PREDICTION SWITCHES
# ==================================================

df["PredictionChanged"] = (
    df["Model2Prediction"]
    !=
    df["Model5Prediction"]
)


df["SwitchCategory"] = np.select(
    [
        (
            df["PredictionChanged"]
            &
            ~df["Model2Correct"]
            &
            df["Model5Correct"]
        ),

        (
            df["PredictionChanged"]
            &
            df["Model2Correct"]
            &
            ~df["Model5Correct"]
        ),

        (
            df["PredictionChanged"]
            &
            ~df["Model2Correct"]
            &
            ~df["Model5Correct"]
        ),
    ],
    [
        "Improved",
        "Worsened",
        "Changed but still wrong",
    ],
    default="No change",
)


print("PREDICTION SWITCHES")
print("===================")
print()

print(
    df[
        "SwitchCategory"
    ]
    .value_counts()
    .to_string()
)


# ==================================================
# SWITCH DETAIL
# ==================================================

switches = df[
    df["PredictionChanged"]
].copy()

switches = switches[
    [
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "HomeGoals",
        "AwayGoals",
        "ActualResult",
        "Model2Prediction",
        "Model5Prediction",
        "SwitchCategory",
        "Model2HomeProbability",
        "Model2DrawProbability",
        "Model2AwayProbability",
        "Model5HomeProbability",
        "Model5DrawProbability",
        "Model5AwayProbability",
        "LogLossImprovement",
        "BrierImprovement",
    ]
].sort_values(
    [
        "Season",
        "Date",
    ]
)


# ==================================================
# TEAM-PERSPECTIVE ANALYSIS
# ==================================================

home_team_rows = pd.DataFrame(
    {
        "Season":
            df["Season"],

        "Team":
            df["HomeTeam"],

        "Matches":
            1,

        "Model2Correct":
            df["Model2Correct"].astype(int),

        "Model5Correct":
            df["Model5Correct"].astype(int),

        "LogLossImprovement":
            df["LogLossImprovement"],

        "BrierImprovement":
            df["BrierImprovement"],
    }
)


away_team_rows = pd.DataFrame(
    {
        "Season":
            df["Season"],

        "Team":
            df["AwayTeam"],

        "Matches":
            1,

        "Model2Correct":
            df["Model2Correct"].astype(int),

        "Model5Correct":
            df["Model5Correct"].astype(int),

        "LogLossImprovement":
            df["LogLossImprovement"],

        "BrierImprovement":
            df["BrierImprovement"],
    }
)


team_rows = pd.concat(
    [
        home_team_rows,
        away_team_rows,
    ],
    ignore_index=True,
)


team_summary = (
    team_rows
    .groupby(
        "Team",
        as_index=False,
    )
    .agg(
        Matches=(
            "Matches",
            "sum",
        ),

        Model2Correct=(
            "Model2Correct",
            "sum",
        ),

        Model5Correct=(
            "Model5Correct",
            "sum",
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
)


team_summary[
    "Model2Accuracy"
] = (
    team_summary[
        "Model2Correct"
    ]
    /
    team_summary[
        "Matches"
    ]
    *
    100.0
)


team_summary[
    "Model5Accuracy"
] = (
    team_summary[
        "Model5Correct"
    ]
    /
    team_summary[
        "Matches"
    ]
    *
    100.0
)


team_summary[
    "AccuracyUpliftPP"
] = (
    team_summary[
        "Model5Accuracy"
    ]
    -
    team_summary[
        "Model2Accuracy"
    ]
)


team_summary = team_summary.sort_values(
    [
        "MeanLogLossImprovement",
        "AccuracyUpliftPP",
    ],
    ascending=[
        False,
        False,
    ],
).reset_index(
    drop=True
)


# ==================================================
# CONFIDENCE ANALYSIS
#
# Uses Model 2 maximum probability before the
# challenger is applied.
# ==================================================

df["Model2Confidence"] = df[
    [
        "Model2HomeProbability",
        "Model2DrawProbability",
        "Model2AwayProbability",
    ]
].max(
    axis=1
)


df["ConfidenceBand"] = pd.cut(
    df["Model2Confidence"],
    bins=[
        0.0,
        0.40,
        0.50,
        0.60,
        0.70,
        1.0,
    ],
    labels=[
        "<=40%",
        "40-50%",
        "50-60%",
        "60-70%",
        ">70%",
    ],
    include_lowest=True,
)


confidence_summary = (
    df
    .groupby(
        "ConfidenceBand",
        observed=False,
    )
    .agg(
        Matches=(
            "ActualResult",
            "size",
        ),

        Model2Accuracy=(
            "Model2Correct",
            "mean",
        ),

        Model5Accuracy=(
            "Model5Correct",
            "mean",
        ),

        MeanLogLossImprovement=(
            "LogLossImprovement",
            "mean",
        ),

        MeanBrierImprovement=(
            "BrierImprovement",
            "mean",
        ),

        PredictionChanges=(
            "PredictionChanged",
            "sum",
        ),
    )
    .reset_index()
)


confidence_summary[
    "Model2Accuracy"
] = (
    confidence_summary[
        "Model2Accuracy"
    ]
    *
    100.0
)


confidence_summary[
    "Model5Accuracy"
] = (
    confidence_summary[
        "Model5Accuracy"
    ]
    *
    100.0
)


confidence_summary[
    "AccuracyUpliftPP"
] = (
    confidence_summary[
        "Model5Accuracy"
    ]
    -
    confidence_summary[
        "Model2Accuracy"
    ]
)


# ==================================================
# PROBABILITY MOVEMENT
# ==================================================

df["HomeProbabilityChange"] = (
    df["Model5HomeProbability"]
    -
    df["Model2HomeProbability"]
)

df["DrawProbabilityChange"] = (
    df["Model5DrawProbability"]
    -
    df["Model2DrawProbability"]
)

df["AwayProbabilityChange"] = (
    df["Model5AwayProbability"]
    -
    df["Model2AwayProbability"]
)


print()
print("AVERAGE PROBABILITY MOVEMENT")
print("============================")
print()

print(
    "Home:",
    f"{df['HomeProbabilityChange'].mean():+.6f}",
)

print(
    "Draw:",
    f"{df['DrawProbabilityChange'].mean():+.6f}",
)

print(
    "Away:",
    f"{df['AwayProbabilityChange'].mean():+.6f}",
)


# ==================================================
# MATCH-LEVEL IMPROVEMENT COUNTS
# ==================================================

logloss_better = int(
    (
        df["LogLossImprovement"]
        >
        0
    ).sum()
)

brier_better = int(
    (
        df["BrierImprovement"]
        >
        0
    ).sum()
)


print()
print("MATCH-LEVEL PROBABILITY QUALITY")
print("===============================")
print()

print(
    "Matches with better Log Loss:",
    f"{logloss_better}/{len(df)}",
    f"({logloss_better / len(df) * 100:.2f}%)",
)

print(
    "Matches with better Brier:",
    f"{brier_better}/{len(df)}",
    f"({brier_better / len(df) * 100:.2f}%)",
)


# ==================================================
# TEAM RESULTS
# ==================================================

print()
print("TOP 10 TEAMS BY LOG LOSS IMPROVEMENT")
print("====================================")

print(
    team_summary
    .head(
        10
    )
    .to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)


print()
print("BOTTOM 10 TEAMS BY LOG LOSS IMPROVEMENT")
print("=======================================")

print(
    team_summary
    .tail(
        10
    )
    .sort_values(
        "MeanLogLossImprovement"
    )
    .to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)


print()
print("UPLIFT BY MODEL 2 CONFIDENCE")
print("============================")

print(
    confidence_summary.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)


# ==================================================
# SAVE
# ==================================================

OUTPUT_TEAM.parent.mkdir(
    parents=True,
    exist_ok=True,
)


team_summary.to_csv(
    OUTPUT_TEAM,
    index=False,
)

confidence_summary.to_csv(
    OUTPUT_CONFIDENCE,
    index=False,
)

switches.to_csv(
    OUTPUT_SWITCHES,
    index=False,
)


print()
print("FILES SAVED")
print("===========")

print(
    OUTPUT_TEAM
)

print(
    OUTPUT_CONFIDENCE
)

print(
    OUTPUT_SWITCHES
)


print()
print("MODEL 5 UPLIFT ANALYSIS COMPLETE")
print("================================")