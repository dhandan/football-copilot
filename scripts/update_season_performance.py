from pathlib import Path
import argparse

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

EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "evaluations"
)

SEASON_FILE = (
    EVALUATION_DIR
    / "2026_27_season_performance.csv"
)


# ==================================================
# ARGUMENTS
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--gameweek",
    type=int,
    required=True,
)

args = parser.parse_args()

gameweek = args.gameweek


# ==================================================
# INPUT FILES
# ==================================================

evaluation_file = (
    EVALUATION_DIR
    / f"2026_27_gw{gameweek:02d}_evaluation.csv"
)

baseline_file = (
    EVALUATION_DIR
    / f"2026_27_gw{gameweek:02d}_baselines.csv"
)


if not evaluation_file.exists():
    raise FileNotFoundError(
        f"Evaluation file not found: {evaluation_file}"
    )


if not baseline_file.exists():
    raise FileNotFoundError(
        f"Baseline file not found: {baseline_file}"
    )


# ==================================================
# LOAD DATA
# ==================================================

evaluation = pd.read_csv(
    evaluation_file
)

baselines = pd.read_csv(
    baseline_file
)


# ==================================================
# BOOLEAN HELPER
# ==================================================

def to_bool(value):

    if isinstance(value, bool):
        return value

    return (
        str(value)
        .strip()
        .lower()
        in [
            "true",
            "1",
            "yes",
        ]
    )


# ==================================================
# GAMEWEEK METRICS
# ==================================================

matches = len(
    evaluation
)


correct = int(
    evaluation[
        "OutcomeCorrect"
    ]
    .apply(to_bool)
    .sum()
)


exact_scores = int(
    evaluation[
        "ExactScoreCorrect"
    ]
    .apply(to_bool)
    .sum()
)


accuracy = (
    correct
    /
    matches
    *
    100
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
# DIAGNOSTIC COUNTS
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
        .astype(str)
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
        .astype(str)
        ==
        "1-1"
    )
    .sum()
)


# ==================================================
# COLD START
# ==================================================

if "ColdStartUsed" in evaluation.columns:

    cold_start = evaluation[
        evaluation[
            "ColdStartUsed"
        ]
        .apply(to_bool)
    ]

else:

    cold_start = pd.DataFrame()


cold_start_matches = len(
    cold_start
)


if cold_start_matches > 0:

    cold_start_correct = int(
        cold_start[
            "OutcomeCorrect"
        ]
        .apply(to_bool)
        .sum()
    )

    cold_start_accuracy = (
        cold_start_correct
        /
        cold_start_matches
        *
        100
    )

else:

    cold_start_correct = 0
    cold_start_accuracy = None


# ==================================================
# BASELINE LOOKUP
# ==================================================

def get_baseline(name):

    rows = baselines[
        baselines[
            "Baseline"
        ]
        ==
        name
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


always_home = get_baseline(
    "Always Home"
)

historical_majority = get_baseline(
    "Historical Majority Class"
)

historical_frequency = get_baseline(
    "Historical Outcome Frequency"
)


# ==================================================
# BUILD GAMEWEEK ROW
# ==================================================

row = {

    "Season":
        "2026/27",

    "Gameweek":
        gameweek,

    "Matches":
        matches,

    "ModelCorrect":
        correct,

    "ModelAccuracyPct":
        accuracy,

    "ModelLogLoss":
        log_loss,

    "ModelBrier":
        brier,

    "ExactScores":
        exact_scores,

    "HomeGoalMAE":
        home_goal_mae,

    "AwayGoalMAE":
        away_goal_mae,

    "TotalGoalMAE":
        total_goal_mae,

    "AlwaysHomeAccuracyPct":
        (
            float(
                always_home[
                    "AccuracyPct"
                ]
            )
            if always_home is not None
            else None
        ),

    "HistoricalMajorityAccuracyPct":
        (
            float(
                historical_majority[
                    "AccuracyPct"
                ]
            )
            if historical_majority is not None
            else None
        ),

    "HistoricalFrequencyAccuracyPct":
        (
            float(
                historical_frequency[
                    "AccuracyPct"
                ]
            )
            if historical_frequency is not None
            else None
        ),

    "HistoricalFrequencyLogLoss":
        (
            float(
                historical_frequency[
                    "LogLoss"
                ]
            )
            if historical_frequency is not None
            else None
        ),

    "HistoricalFrequencyBrier":
        (
            float(
                historical_frequency[
                    "Brier"
                ]
            )
            if historical_frequency is not None
            else None
        ),

    "PredictedDraws":
        predicted_draws,

    "ActualDraws":
        actual_draws,

    "ModalOneOnePredictions":
        modal_one_one,

    "ActualOneOneResults":
        actual_one_one,

    "ColdStartMatches":
        cold_start_matches,

    "ColdStartCorrect":
        cold_start_correct,

    "ColdStartAccuracyPct":
        cold_start_accuracy,
}


new_row = pd.DataFrame(
    [row]
)


# ==================================================
# UPDATE SEASON FILE
# ==================================================

if SEASON_FILE.exists():

    season = pd.read_csv(
        SEASON_FILE
    )

    # Remove an existing row for this GW.
    # This makes the script safe to rerun.

    season = season[
        season[
            "Gameweek"
        ]
        !=
        gameweek
    ]

    season = pd.concat(
        [
            season,
            new_row,
        ],
        ignore_index=True,
    )

else:

    season = new_row


season = season.sort_values(
    "Gameweek"
).reset_index(
    drop=True
)


season.to_csv(
    SEASON_FILE,
    index=False,
)


# ==================================================
# CUMULATIVE PERFORMANCE
# ==================================================

total_matches = int(
    season[
        "Matches"
    ]
    .sum()
)


total_correct = int(
    season[
        "ModelCorrect"
    ]
    .sum()
)


cumulative_accuracy = (
    total_correct
    /
    total_matches
    *
    100
)


weighted_log_loss = (
    (
        season[
            "ModelLogLoss"
        ]
        *
        season[
            "Matches"
        ]
    )
    .sum()
    /
    total_matches
)


weighted_brier = (
    (
        season[
            "ModelBrier"
        ]
        *
        season[
            "Matches"
        ]
    )
    .sum()
    /
    total_matches
)


# ==================================================
# OUTPUT
# ==================================================

print()
print("FOOTBALL COPILOT")
print("2026/27 SEASON PERFORMANCE TRACKER")
print("=" * 65)

print()
print(
    season.to_string(
        index=False
    )
)

print()
print("=" * 65)
print("CUMULATIVE MODEL 2 PERFORMANCE")
print("=" * 65)

print()

print(
    f"Gameweeks evaluated: "
    f"{len(season)}"
)

print(
    f"Matches evaluated:    "
    f"{total_matches}"
)

print(
    f"Correct outcomes:     "
    f"{total_correct}"
)

print(
    f"1X2 accuracy:         "
    f"{cumulative_accuracy:.1f}%"
)

print(
    f"Log Loss:             "
    f"{weighted_log_loss:.4f}"
)

print(
    f"Brier Score:          "
    f"{weighted_brier:.4f}"
)

print()

print(
    f"Saved to: {SEASON_FILE}"
)