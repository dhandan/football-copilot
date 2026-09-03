from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    accuracy_score,
    log_loss,
)


# ==================================================
# PROJECT SETUP
# ==================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.append(
    str(PROJECT_ROOT)
)

from prediction.dixon_coles import (
    independent_poisson_probabilities,
)


# ==================================================
# FILES
# ==================================================

INPUT_FILE = (
    "data/processed/"
    "prediction_features_v2.csv"
)

OUTPUT_FILE = (
    "reports/model5/"
    "model5_overlap_baseline.csv"
)


# ==================================================
# MODEL 2 FEATURES
#
# Intentionally identical to the controlled
# Model 2 recreation used in the Model 4 backtest.
# ==================================================

MODEL_2_FEATURES = [

    "HomeRecentGoalsFor",
    "HomeRecentGoalsAgainst",
    "HomeRecentPPG",

    "AwayRecentGoalsFor",
    "AwayRecentGoalsAgainst",
    "AwayRecentPPG",

    "Home10GoalsFor",
    "Home10GoalsAgainst",
    "Home10PPG",

    "Away10GoalsFor",
    "Away10GoalsAgainst",
    "Away10PPG",

    "HomeVenuePPG",
    "HomeVenueGoalsFor",

    "AwayVenuePPG",
    "AwayVenueGoalsFor",

    "HomeSeasonPPG",
    "HomeSeasonGoalDifferencePG",

    "AwaySeasonPPG",
    "AwaySeasonGoalDifferencePG",

    "RecentPPGDifference",
    "TenMatchPPGDifference",
    "SeasonPPGDifference",

    "AttackVsDefenceHome",
    "AttackVsDefenceAway",
]


# ==================================================
# LOCKED MODEL 5 OVERLAP TEST SEASONS
# ==================================================

TEST_SEASONS = [
    "2023/24",
    "2024/25",
]


# ==================================================
# HELPERS
# ==================================================

def actual_result(
    home_goals: int,
    away_goals: int,
) -> str:

    if home_goals > away_goals:
        return "H"

    if home_goals < away_goals:
        return "A"

    return "D"


def multiclass_brier_score(
    actual_numeric,
    probabilities,
):

    actual_numeric = np.asarray(
        actual_numeric
    )

    probabilities = np.asarray(
        probabilities
    )

    targets = np.zeros_like(
        probabilities
    )

    targets[
        np.arange(
            len(actual_numeric)
        ),
        actual_numeric,
    ] = 1.0

    return float(
        np.mean(
            np.sum(
                (
                    probabilities
                    -
                    targets
                )
                ** 2,
                axis=1,
            )
        )
    )


# ==================================================
# LOAD DATA
# ==================================================

print()
print("FOOTBALL COPILOT")
print("MODEL 5 OVERLAP BASELINE")
print("========================")

df = pd.read_csv(
    INPUT_FILE
)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce",
)

df = df.sort_values(
    [
        "Date",
        "HomeTeam",
        "AwayTeam",
    ]
).reset_index(
    drop=True
)


# ==================================================
# TARGET
# ==================================================

df["ActualResult"] = df.apply(
    lambda row: actual_result(
        int(row["HomeGoals"]),
        int(row["AwayGoals"]),
    ),
    axis=1,
)

RESULT_TO_NUMBER = {
    "H": 0,
    "D": 1,
    "A": 2,
}

df["Target"] = (
    df["ActualResult"]
    .map(
        RESULT_TO_NUMBER
    )
)


# ==================================================
# VALIDATE FEATURES
# ==================================================

missing_features = [
    feature
    for feature
    in MODEL_2_FEATURES
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing Model 2 features: "
        +
        ", ".join(
            missing_features
        )
    )


df[
    MODEL_2_FEATURES
] = (
    df[
        MODEL_2_FEATURES
    ]
    .replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )
    .fillna(
        0.0
    )
)


# ==================================================
# WALK-FORWARD TEST
# ==================================================

season_rows = []

all_actual = []
all_probabilities = []
all_predictions = []

for test_season in TEST_SEASONS:

    season_position = (
        sorted(
            df["Season"]
            .dropna()
            .unique()
        )
        .index(
            test_season
        )
    )

    all_seasons = sorted(
        df["Season"]
        .dropna()
        .unique()
    )

    training_seasons = (
        all_seasons[
            :season_position
        ]
    )

    train = df[
        df["Season"].isin(
            training_seasons
        )
    ].copy()

    test = df[
        df["Season"]
        ==
        test_season
    ].copy()

    train = train.reset_index(
        drop=True
    )

    test = test.reset_index(
        drop=True
    )

    print()
    print(
        "Training on:",
        ", ".join(
            training_seasons
        ),
    )

    print(
        "Testing on:",
        test_season,
    )

    print(
        "Training matches:",
        len(train),
    )

    print(
        "Test matches:",
        len(test),
    )


    # ==============================================
    # MODEL 2
    # ==============================================

    home_poisson = PoissonRegressor(
        alpha=0.1,
        max_iter=1000,
    )

    away_poisson = PoissonRegressor(
        alpha=0.1,
        max_iter=1000,
    )

    home_poisson.fit(
        train[
            MODEL_2_FEATURES
        ],
        train[
            "HomeGoals"
        ],
    )

    away_poisson.fit(
        train[
            MODEL_2_FEATURES
        ],
        train[
            "AwayGoals"
        ],
    )

    home_xg = (
        home_poisson
        .predict(
            test[
                MODEL_2_FEATURES
            ]
        )
    )

    away_xg = (
        away_poisson
        .predict(
            test[
                MODEL_2_FEATURES
            ]
        )
    )

    probabilities = []

    predictions = []

    for i in range(
        len(test)
    ):

        probs = (
            independent_poisson_probabilities(
                home_xg=float(
                    home_xg[i]
                ),
                away_xg=float(
                    away_xg[i]
                ),
                max_goals=8,
            )
        )

        h = float(
            probs[
                "home_probability"
            ]
        )

        d = float(
            probs[
                "draw_probability"
            ]
        )

        a = float(
            probs[
                "away_probability"
            ]
        )

        probabilities.append(
            [
                h,
                d,
                a,
            ]
        )

        predictions.append(
            int(
                np.argmax(
                    [
                        h,
                        d,
                        a,
                    ]
                )
            )
        )


    # ==============================================
    # METRICS
    # ==============================================

    actual_numeric = (
        test[
            "Target"
        ]
        .astype(int)
        .to_numpy()
    )

    probabilities_array = np.asarray(
        probabilities
    )

    predictions_array = np.asarray(
        predictions
    )

    accuracy = (
        accuracy_score(
            actual_numeric,
            predictions_array,
        )
        *
        100.0
    )

    season_logloss = log_loss(
        actual_numeric,
        probabilities_array,
        labels=[
            0,
            1,
            2,
        ],
    )

    season_brier = (
        multiclass_brier_score(
            actual_numeric,
            probabilities_array,
        )
    )

    actual_draws = int(
        np.sum(
            actual_numeric == 1
        )
    )

    predicted_draws = int(
        np.sum(
            predictions_array == 1
        )
    )

    print(
        f"Accuracy: {accuracy:.4f}%"
    )

    print(
        f"Log Loss: {season_logloss:.4f}"
    )

    print(
        f"Brier: {season_brier:.4f}"
    )

    print(
        f"Actual draws: {actual_draws}"
    )

    print(
        f"Predicted draws: {predicted_draws}"
    )

    season_rows.append(
        {
            "Season":
                test_season,

            "Matches":
                len(test),

            "Accuracy":
                accuracy,

            "LogLoss":
                season_logloss,

            "Brier":
                season_brier,

            "ActualDraws":
                actual_draws,

            "PredictedDraws":
                predicted_draws,
        }
    )

    all_actual.extend(
        actual_numeric.tolist()
    )

    all_probabilities.extend(
        probabilities
    )

    all_predictions.extend(
        predictions
    )


# ==================================================
# OVERALL 760-MATCH RESULT
# ==================================================

all_actual = np.asarray(
    all_actual
)

all_probabilities = np.asarray(
    all_probabilities
)

all_predictions = np.asarray(
    all_predictions
)

overall_accuracy = (
    accuracy_score(
        all_actual,
        all_predictions,
    )
    *
    100.0
)

overall_logloss = log_loss(
    all_actual,
    all_probabilities,
    labels=[
        0,
        1,
        2,
    ],
)

overall_brier = (
    multiclass_brier_score(
        all_actual,
        all_probabilities,
    )
)

overall_actual_draws = int(
    np.sum(
        all_actual == 1
    )
)

overall_predicted_draws = int(
    np.sum(
        all_predictions == 1
    )
)


# ==================================================
# RESULTS
# ==================================================

results = pd.DataFrame(
    season_rows
)

print()
print("BY SEASON")
print("=========")

print(
    results.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)

print()
print("OVERALL XG-COVERED BASELINE")
print("===========================")

print(
    f"Matches: {len(all_actual)}"
)

print(
    f"Accuracy: "
    f"{overall_accuracy:.4f}%"
)

print(
    f"Log Loss: "
    f"{overall_logloss:.4f}"
)

print(
    f"Brier: "
    f"{overall_brier:.4f}"
)

print(
    f"Actual draws: "
    f"{overall_actual_draws}"
)

print(
    f"Predicted draws: "
    f"{overall_predicted_draws}"
)


# ==================================================
# SAVE
# ==================================================

output = pd.DataFrame(
    [
        {
            "Matches":
                len(all_actual),

            "Accuracy":
                overall_accuracy,

            "LogLoss":
                overall_logloss,

            "Brier":
                overall_brier,

            "ActualDraws":
                overall_actual_draws,

            "PredictedDraws":
                overall_predicted_draws,
        }
    ]
)

Path(
    "reports/model5"
).mkdir(
    parents=True,
    exist_ok=True,
)

output.to_csv(
    OUTPUT_FILE,
    index=False,
)

print()
print(
    "Saved:",
    OUTPUT_FILE,
)

print()
print("MODEL 5 BASELINE COMPLETE")
print("=========================")