from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.linear_model import (
    PoissonRegressor
)

from sklearn.metrics import (
    log_loss
)

from scipy.stats import poisson


# --------------------------------------------------
# Setup
# --------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.append(
    str(PROJECT_ROOT)
)


INPUT_FILE = (
    "data/processed/"
    "prediction_features.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "walk_forward_results.csv"
)


FEATURE_COLUMNS = [
    "HomeRecentGoalsFor",
    "HomeRecentGoalsAgainst",
    "HomeRecentPPG",
    "AwayRecentGoalsFor",
    "AwayRecentGoalsAgainst",
    "AwayRecentPPG",
]


# --------------------------------------------------
# Functions
# --------------------------------------------------

def match_probabilities(
    home_xg,
    away_xg,
    max_goals=8,
):

    home_win = 0.0
    draw = 0.0
    away_win = 0.0


    for home_goals in range(
        max_goals + 1
    ):

        for away_goals in range(
            max_goals + 1
        ):

            probability = (
                poisson.pmf(
                    home_goals,
                    home_xg,
                )
                *
                poisson.pmf(
                    away_goals,
                    away_xg,
                )
            )

            if home_goals > away_goals:
                home_win += probability

            elif home_goals == away_goals:
                draw += probability

            else:
                away_win += probability


    total = (
        home_win
        +
        draw
        +
        away_win
    )


    return (
        home_win / total,
        draw / total,
        away_win / total,
    )


def actual_result(
    home_goals,
    away_goals,
):

    if home_goals > away_goals:
        return "H"

    if home_goals == away_goals:
        return "D"

    return "A"


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(
    INPUT_FILE
)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)


SEASONS = sorted(
    df["Season"].unique()
)


print("\nWALK-FORWARD BACKTEST")
print("=====================")

print(
    "Seasons:",
    SEASONS
)


all_predictions = []


# --------------------------------------------------
# Walk forward through seasons
# --------------------------------------------------

for test_index in range(
    2,
    len(SEASONS)
):

    test_season = (
        SEASONS[
            test_index
        ]
    )

    training_seasons = (
        SEASONS[
            :test_index
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


    print()
    print(
        f"Training on: "
        f"{', '.join(training_seasons)}"
    )

    print(
        f"Testing on: {test_season}"
    )


    X_train = train[
        FEATURE_COLUMNS
    ]

    y_home_train = train[
        "HomeGoals"
    ]

    y_away_train = train[
        "AwayGoals"
    ]


    home_model = PoissonRegressor(
        alpha=0.1,
        max_iter=1000,
    )

    away_model = PoissonRegressor(
        alpha=0.1,
        max_iter=1000,
    )


    home_model.fit(
        X_train,
        y_home_train,
    )

    away_model.fit(
        X_train,
        y_away_train,
    )


    X_test = test[
        FEATURE_COLUMNS
    ]


    predicted_home_goals = (
        home_model.predict(
            X_test
        )
    )

    predicted_away_goals = (
        away_model.predict(
            X_test
        )
    )


    test = test.reset_index(
        drop=True
    )


    for i in range(
        len(test)
    ):

        home_probability, \
        draw_probability, \
        away_probability = (
            match_probabilities(
                predicted_home_goals[i],
                predicted_away_goals[i],
            )
        )


        actual = actual_result(
            test.loc[
                i,
                "HomeGoals"
            ],
            test.loc[
                i,
                "AwayGoals"
            ],
        )


        probabilities = {
            "H": home_probability,
            "D": draw_probability,
            "A": away_probability,
        }


        predicted = max(
            probabilities,
            key=probabilities.get
        )


        all_predictions.append(
            {
                "Season":
                    test_season,

                "Date":
                    test.loc[
                        i,
                        "Date"
                    ],

                "HomeTeam":
                    test.loc[
                        i,
                        "HomeTeam"
                    ],

                "AwayTeam":
                    test.loc[
                        i,
                        "AwayTeam"
                    ],

                "ActualResult":
                    actual,

                "PredictedResult":
                    predicted,

                "HomeProbability":
                    home_probability,

                "DrawProbability":
                    draw_probability,

                "AwayProbability":
                    away_probability,

                "ExpectedHomeGoals":
                    predicted_home_goals[i],

                "ExpectedAwayGoals":
                    predicted_away_goals[i],
            }
        )


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

results = pd.DataFrame(
    all_predictions
)


results[
    "Correct"
] = (
    results[
        "ActualResult"
    ]
    ==
    results[
        "PredictedResult"
    ]
)


accuracy = (
    results[
        "Correct"
    ].mean()
)


label_map = {
    "H": 0,
    "D": 1,
    "A": 2,
}


y_true = (
    results[
        "ActualResult"
    ].map(
        label_map
    )
)


y_prob = results[
    [
        "HomeProbability",
        "DrawProbability",
        "AwayProbability",
    ]
].to_numpy()


walk_forward_log_loss = (
    log_loss(
        y_true,
        y_prob,
        labels=[0, 1, 2],
    )
)


actual_matrix = np.zeros(
    (
        len(results),
        3
    )
)


for i, result in enumerate(
    results[
        "ActualResult"
    ]
):

    actual_matrix[
        i,
        label_map[result]
    ] = 1


walk_forward_brier = (
    np.mean(
        np.sum(
            (
                y_prob
                -
                actual_matrix
            )
            ** 2,
            axis=1,
        )
    )
)


# --------------------------------------------------
# Output
# --------------------------------------------------

print("\nOVERALL RESULTS")
print("===============")

print(
    f"Matches tested: "
    f"{len(results)}"
)

print(
    f"Accuracy: "
    f"{accuracy * 100:.1f}%"
)

print(
    f"Log loss: "
    f"{walk_forward_log_loss:.4f}"
)

print(
    f"Brier score: "
    f"{walk_forward_brier:.4f}"
)


print("\nBY SEASON")
print("=========")


season_summary = (
    results
    .groupby("Season")
    .agg(
        Matches=("Correct", "count"),
        Accuracy=("Correct", "mean"),
    )
    .reset_index()
)


season_summary[
    "AccuracyPct"
] = (
    season_summary[
        "Accuracy"
    ]
    * 100
)


print(
    season_summary[
        [
            "Season",
            "Matches",
            "AccuracyPct",
        ]
    ]
)


# --------------------------------------------------
# Save
# --------------------------------------------------

results.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nResults saved to: "
    f"{OUTPUT_FILE}"
)