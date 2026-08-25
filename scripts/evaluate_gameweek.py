from pathlib import Path
import argparse
import math

import pandas as pd


# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "predictions"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "results"
)

EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "evaluations"
)


# ==================================================
# ARGUMENTS
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--gameweek",
    type=int,
    required=True,
    help="Premier League gameweek number.",
)

args = parser.parse_args()

gameweek = args.gameweek


# ==================================================
# INPUT FILES
# ==================================================

prediction_file = (
    PREDICTION_DIR
    /
    f"2026_27_gw{gameweek:02d}_predictions.csv"
)

results_file = (
    RESULTS_DIR
    /
    f"2026_27_gw{gameweek:02d}_results.csv"
)


if not prediction_file.exists():

    raise FileNotFoundError(
        f"Prediction file not found: "
        f"{prediction_file}"
    )


if not results_file.exists():

    raise FileNotFoundError(
        f"Results file not found: "
        f"{results_file}"
    )


predictions = pd.read_csv(
    prediction_file
)

results = pd.read_csv(
    results_file
)


if predictions.empty:

    raise ValueError(
        "Prediction file is empty."
    )


if results.empty:

    raise ValueError(
        "Results file is empty."
    )


# ==================================================
# REMOVE EMPTY RESULT PLACEHOLDERS
#
# The frozen prediction snapshot deliberately
# contained blank result columns.
#
# Remove them from the working copy before joining
# the actual result snapshot so Pandas does not
# create ActualResult_x / ActualResult_y columns.
#
# The original prediction CSV is NOT modified.
# ==================================================

prediction_result_columns = [
    "ActualHomeGoals",
    "ActualAwayGoals",
    "ActualResult",
    "ActualScore",
    "ActualResultCode",
    "Evaluated",
]


predictions_for_evaluation = (
    predictions.drop(
        columns=[
            column
            for column
            in prediction_result_columns
            if column in predictions.columns
        ],
        errors="ignore",
    )
)


# ==================================================
# RESULT COLUMNS
# ==================================================

required_result_columns = [
    "FixtureId",
    "ActualHomeGoals",
    "ActualAwayGoals",
    "ActualScore",
    "ActualResult",
    "ActualResultCode",
]


missing_result_columns = [
    column
    for column
    in required_result_columns
    if column not in results.columns
]


if missing_result_columns:

    raise ValueError(
        "Results file is missing columns: "
        f"{missing_result_columns}"
    )


# ==================================================
# JOIN PREDICTIONS TO ACTUAL RESULTS
# ==================================================

evaluation = (
    predictions_for_evaluation.merge(
        results[
            required_result_columns
        ],
        on="FixtureId",
        how="left",
        validate="one_to_one",
    )
)


# ==================================================
# VALIDATE JOIN
# ==================================================

if len(evaluation) != len(predictions):

    raise RuntimeError(
        "Prediction/result join did not "
        "produce all prediction rows."
    )


missing_results = (
    evaluation[
        "ActualResult"
    ]
    .isna()
    .sum()
)


if missing_results > 0:

    raise RuntimeError(
        f"{missing_results} prediction(s) "
        "could not be matched to a result."
    )


# ==================================================
# NORMALISE TYPES
# ==================================================

for column in [
    "HomeWinProbability",
    "DrawProbability",
    "AwayWinProbability",
    "ExpectedHomeGoals",
    "ExpectedAwayGoals",
    "ActualHomeGoals",
    "ActualAwayGoals",
]:

    evaluation[
        column
    ] = pd.to_numeric(
        evaluation[
            column
        ],
        errors="coerce",
    )


if evaluation[
    [
        "HomeWinProbability",
        "DrawProbability",
        "AwayWinProbability",
    ]
].isna().any().any():

    raise ValueError(
        "Missing prediction probabilities found."
    )


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def actual_probability(
    row,
):

    if (
        row[
            "ActualResultCode"
        ]
        ==
        "H"
    ):

        return (
            row[
                "HomeWinProbability"
            ]
            /
            100.0
        )


    if (
        row[
            "ActualResultCode"
        ]
        ==
        "D"
    ):

        return (
            row[
                "DrawProbability"
            ]
            /
            100.0
        )


    if (
        row[
            "ActualResultCode"
        ]
        ==
        "A"
    ):

        return (
            row[
                "AwayWinProbability"
            ]
            /
            100.0
        )


    raise ValueError(
        "Unknown ActualResultCode: "
        f"{row['ActualResultCode']}"
    )


def calculate_brier(
    row,
):

    probabilities = [
        row[
            "HomeWinProbability"
        ]
        /
        100.0,

        row[
            "DrawProbability"
        ]
        /
        100.0,

        row[
            "AwayWinProbability"
        ]
        /
        100.0,
    ]


    if (
        row[
            "ActualResultCode"
        ]
        ==
        "H"
    ):

        actual = [
            1.0,
            0.0,
            0.0,
        ]


    elif (
        row[
            "ActualResultCode"
        ]
        ==
        "D"
    ):

        actual = [
            0.0,
            1.0,
            0.0,
        ]


    elif (
        row[
            "ActualResultCode"
        ]
        ==
        "A"
    ):

        actual = [
            0.0,
            0.0,
            1.0,
        ]


    else:

        raise ValueError(
            "Unknown ActualResultCode: "
            f"{row['ActualResultCode']}"
        )


    return sum(
        (
            probability
            -
            target
        )
        ** 2

        for probability, target
        in zip(
            probabilities,
            actual,
        )
    )


# ==================================================
# OUTCOME ACCURACY
# ==================================================

evaluation[
    "OutcomeCorrect"
] = (
    evaluation[
        "PredictedResult"
    ]
    ==
    evaluation[
        "ActualResult"
    ]
)


# ==================================================
# EXACT SCORE
# ==================================================

evaluation[
    "MostLikelyScore"
] = (
    evaluation[
        "MostLikelyScore"
    ]
    .astype(
        str
    )
)


evaluation[
    "ActualScore"
] = (
    evaluation[
        "ActualScore"
    ]
    .astype(
        str
    )
)


evaluation[
    "ExactScoreCorrect"
] = (
    evaluation[
        "MostLikelyScore"
    ]
    ==
    evaluation[
        "ActualScore"
    ]
)


# ==================================================
# PROBABILITY METRICS
# ==================================================

evaluation[
    "ActualOutcomeProbability"
] = evaluation.apply(
    actual_probability,
    axis=1,
)


evaluation[
    "LogLoss"
] = evaluation[
    "ActualOutcomeProbability"
].apply(
    lambda probability:
        -math.log(
            max(
                probability,
                1e-15,
            )
        )
)


evaluation[
    "Brier"
] = evaluation.apply(
    calculate_brier,
    axis=1,
)


# ==================================================
# GOAL ERROR METRICS
# ==================================================

evaluation[
    "HomeGoalError"
] = (
    evaluation[
        "ExpectedHomeGoals"
    ]
    -
    evaluation[
        "ActualHomeGoals"
    ]
)


evaluation[
    "AwayGoalError"
] = (
    evaluation[
        "ExpectedAwayGoals"
    ]
    -
    evaluation[
        "ActualAwayGoals"
    ]
)


evaluation[
    "HomeGoalAbsoluteError"
] = (
    evaluation[
        "HomeGoalError"
    ]
    .abs()
)


evaluation[
    "AwayGoalAbsoluteError"
] = (
    evaluation[
        "AwayGoalError"
    ]
    .abs()
)


evaluation[
    "PredictedTotalGoals"
] = (
    evaluation[
        "ExpectedHomeGoals"
    ]
    +
    evaluation[
        "ExpectedAwayGoals"
    ]
)


evaluation[
    "ActualTotalGoals"
] = (
    evaluation[
        "ActualHomeGoals"
    ]
    +
    evaluation[
        "ActualAwayGoals"
    ]
)


evaluation[
    "TotalGoalsAbsoluteError"
] = (
    evaluation[
        "PredictedTotalGoals"
    ]
    -
    evaluation[
        "ActualTotalGoals"
    ]
).abs()


# ==================================================
# OVERALL METRICS
# ==================================================

matches = len(
    evaluation
)


correct = int(
    evaluation[
        "OutcomeCorrect"
    ]
    .sum()
)


accuracy = (
    correct
    /
    matches
    *
    100.0
)


exact_scores = int(
    evaluation[
        "ExactScoreCorrect"
    ]
    .sum()
)


log_loss = float(
    evaluation[
        "LogLoss"
    ]
    .mean()
)


brier = float(
    evaluation[
        "Brier"
    ]
    .mean()
)


home_goal_mae = float(
    evaluation[
        "HomeGoalAbsoluteError"
    ]
    .mean()
)


away_goal_mae = float(
    evaluation[
        "AwayGoalAbsoluteError"
    ]
    .mean()
)


total_goal_mae = float(
    evaluation[
        "TotalGoalsAbsoluteError"
    ]
    .mean()
)


# ==================================================
# PREDICTED / ACTUAL DRAW DIAGNOSTICS
# ==================================================

predicted_draws = int(
    (
        evaluation[
            "PredictedResult"
        ]
        ==
        "Draw"
    )
    .sum()
)


actual_draws = int(
    (
        evaluation[
            "ActualResult"
        ]
        ==
        "Draw"
    )
    .sum()
)


modal_one_one = int(
    (
        evaluation[
            "MostLikelyScore"
        ]
        ==
        "1-1"
    )
    .sum()
)


actual_one_one = int(
    (
        evaluation[
            "ActualScore"
        ]
        ==
        "1-1"
    )
    .sum()
)


# ==================================================
# COLD START PERFORMANCE
# ==================================================

if (
    "ColdStartUsed"
    in evaluation.columns
):

    cold_start_flag = (
        evaluation[
            "ColdStartUsed"
        ]
        .astype(
            str
        )
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    )


    cold_start = (
        evaluation[
            cold_start_flag
        ]
        .copy()
    )


else:

    cold_start = (
        evaluation.iloc[
            0:0
        ]
        .copy()
    )


# ==================================================
# SAVE EVALUATION
# ==================================================

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


output_file = (
    EVALUATION_DIR
    /
    f"2026_27_gw{gameweek:02d}_evaluation.csv"
)


evaluation.to_csv(
    output_file,
    index=False,
)


# ==================================================
# DISPLAY FIXTURE EVALUATION
# ==================================================

print()
print("FOOTBALL COPILOT")

print(
    f"2026/27 GAMEWEEK {gameweek} "
    "MODEL EVALUATION"
)

print(
    "=" * 65
)


for _, row in (
    evaluation.iterrows()
):

    outcome_symbol = (
        "✓"
        if row[
            "OutcomeCorrect"
        ]
        else
        "✗"
    )


    exact_symbol = (
        "✓"
        if row[
            "ExactScoreCorrect"
        ]
        else
        "✗"
    )


    print()

    print(
        f"{row['HomeTeam']} "
        f"vs "
        f"{row['AwayTeam']}"
    )


    print(
        f"Predicted outcome: "
        f"{row['PredictedResult']}"
    )


    print(
        f"Actual outcome:    "
        f"{row['ActualResult']} "
        f"{outcome_symbol}"
    )


    print(
        f"Predicted score:   "
        f"{row['MostLikelyScore']}"
    )


    print(
        f"Actual score:      "
        f"{row['ActualScore']} "
        f"{exact_symbol}"
    )


    print(
        "Actual outcome probability: "
        f"{row['ActualOutcomeProbability'] * 100:.1f}%"
    )


    print(
        f"Fixture Log Loss:  "
        f"{row['LogLoss']:.4f}"
    )


# ==================================================
# OVERALL SUMMARY
# ==================================================

print()
print(
    "=" * 65
)

print(
    "GAMEWEEK PERFORMANCE"
)

print(
    "=" * 65
)


print(
    f"Matches evaluated:     "
    f"{matches}"
)

print(
    f"Correct outcomes:      "
    f"{correct}"
)

print(
    f"1X2 accuracy:          "
    f"{accuracy:.1f}%"
)

print(
    f"Exact scores:          "
    f"{exact_scores}"
)

print(
    f"Log Loss:              "
    f"{log_loss:.4f}"
)

print(
    f"Brier Score:           "
    f"{brier:.4f}"
)


print()

print(
    f"Home goals MAE:        "
    f"{home_goal_mae:.3f}"
)

print(
    f"Away goals MAE:        "
    f"{away_goal_mae:.3f}"
)

print(
    f"Total goals MAE:       "
    f"{total_goal_mae:.3f}"
)


# ==================================================
# HISTORICAL COMPARISON
# ==================================================

print()
print(
    "=" * 65
)

print(
    "VS HISTORICAL MODEL 2"
)

print(
    "=" * 65
)


print(
    "Historical accuracy:   "
    "52.69%"
)

print(
    f"GW{gameweek} accuracy:          "
    f"{accuracy:.1f}%"
)


print()

print(
    "Historical Log Loss:   "
    "0.9927"
)

print(
    f"GW{gameweek} Log Loss:          "
    f"{log_loss:.4f}"
)


print()

print(
    "Historical Brier:      "
    "0.5927"
)

print(
    f"GW{gameweek} Brier:             "
    f"{brier:.4f}"
)


# ==================================================
# DRAW / SCORELINE DIAGNOSTICS
# ==================================================

print()
print(
    "=" * 65
)

print(
    "DRAW / SCORELINE DIAGNOSTICS"
)

print(
    "=" * 65
)


print(
    f"Predicted 1X2 draws:    "
    f"{predicted_draws}"
)

print(
    f"Actual draws:           "
    f"{actual_draws}"
)

print(
    f"1-1 modal scorelines:   "
    f"{modal_one_one}"
)

print(
    f"Actual 1-1 results:     "
    f"{actual_one_one}"
)


# ==================================================
# COLD START
# ==================================================

if not cold_start.empty:

    cold_matches = len(
        cold_start
    )


    cold_correct = int(
        cold_start[
            "OutcomeCorrect"
        ]
        .sum()
    )


    cold_accuracy = (
        cold_correct
        /
        cold_matches
        *
        100.0
    )


    cold_log_loss = float(
        cold_start[
            "LogLoss"
        ]
        .mean()
    )


    cold_brier = float(
        cold_start[
            "Brier"
        ]
        .mean()
    )


    print()
    print(
        "=" * 65
    )

    print(
        "PROMOTED TEAM COLD-START"
    )

    print(
        "=" * 65
    )


    print(
        f"Fixtures:              "
        f"{cold_matches}"
    )

    print(
        f"Correct outcomes:      "
        f"{cold_correct}"
    )

    print(
        f"Accuracy:              "
        f"{cold_accuracy:.1f}%"
    )

    print(
        f"Log Loss:              "
        f"{cold_log_loss:.4f}"
    )

    print(
        f"Brier:                 "
        f"{cold_brier:.4f}"
    )


# ==================================================
# BEST / WORST PROBABILITY CALLS
# ==================================================

worst = (
    evaluation
    .sort_values(
        "ActualOutcomeProbability",
        ascending=True,
    )
    .iloc[
        0
    ]
)


best = (
    evaluation
    .sort_values(
        "ActualOutcomeProbability",
        ascending=False,
    )
    .iloc[
        0
    ]
)


print()
print(
    "=" * 65
)

print(
    "DIAGNOSTICS"
)

print(
    "=" * 65
)


print()

print(
    "BIGGEST MODEL SURPRISE"
)

print(
    f"{worst['HomeTeam']} "
    f"vs "
    f"{worst['AwayTeam']}"
)

print(
    f"Actual outcome: "
    f"{worst['ActualResult']}"
)

print(
    "Probability assigned to actual outcome: "
    f"{worst['ActualOutcomeProbability'] * 100:.1f}%"
)


print()

print(
    "HIGHEST-CONFIDENCE SUCCESS"
)

print(
    f"{best['HomeTeam']} "
    f"vs "
    f"{best['AwayTeam']}"
)

print(
    f"Actual outcome: "
    f"{best['ActualResult']}"
)

print(
    "Probability assigned to actual outcome: "
    f"{best['ActualOutcomeProbability'] * 100:.1f}%"
)


print()

print(
    f"Evaluation saved to: "
    f"{output_file}"
)