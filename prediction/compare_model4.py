from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import log_loss
from scipy.stats import poisson


# --------------------------------------------------
# Files
# --------------------------------------------------

FEATURE_FILE = (
    "data/processed/"
    "prediction_features_v3.csv"
)

MARKET_FILE = (
    "data/processed/"
    "market_probabilities.csv"
)

DETAIL_OUTPUT = (
    "data/processed/"
    "model4_comparison_results.csv"
)

SUMMARY_OUTPUT = (
    "data/processed/"
    "model4_comparison_summary.csv"
)

SEASON_OUTPUT = (
    "data/processed/"
    "model4_comparison_by_season.csv"
)


# --------------------------------------------------
# Model 2 features
#
# Model 2 remains our current champion.
# --------------------------------------------------

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


# --------------------------------------------------
# Model 4 features
#
# Use the full Model 3 feature set, but allow
# boosted trees to learn nonlinear relationships.
# --------------------------------------------------

MODEL_4_FEATURES = (

    MODEL_2_FEATURES
    +
    [

        "HomeElo",
        "AwayElo",
        "EloDifference",

        "HomeWeightedGoalsFor",
        "HomeWeightedGoalsAgainst",
        "HomeWeightedPPG",

        "AwayWeightedGoalsFor",
        "AwayWeightedGoalsAgainst",
        "AwayWeightedPPG",

        "HomeWeightedVenueGoalsFor",
        "HomeWeightedVenueGoalsAgainst",
        "HomeWeightedVenuePPG",

        "AwayWeightedVenueGoalsFor",
        "AwayWeightedVenueGoalsAgainst",
        "AwayWeightedVenuePPG",

        "WeightedPPGDifference",
        "WeightedAttackDifference",
        "WeightedDefenceDifference",

        "WeightedHomeAttackVsAwayDefence",
        "WeightedAwayAttackVsHomeDefence",
    ]
)


# --------------------------------------------------
# Labels
# --------------------------------------------------

LABEL_MAP = {
    "H": 0,
    "D": 1,
    "A": 2,
}

REVERSE_LABEL_MAP = {
    0: "H",
    1: "D",
    2: "A",
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def actual_result(
    home_goals,
    away_goals,
):

    if home_goals > away_goals:
        return "H"

    if home_goals == away_goals:
        return "D"

    return "A"


def poisson_probabilities(
    home_xg,
    away_xg,
    max_goals=8,
):

    home = 0.0
    draw = 0.0
    away = 0.0


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

                home += probability


            elif home_goals == away_goals:

                draw += probability


            else:

                away += probability


    total = (
        home
        +
        draw
        +
        away
    )


    return np.array(
        [
            home / total,
            draw / total,
            away / total,
        ]
    )


def multiclass_brier(
    actual_results,
    probabilities,
):

    actual_matrix = np.zeros(
        (
            len(actual_results),
            3,
        )
    )


    for row_number, result in enumerate(
        actual_results
    ):

        actual_matrix[
            row_number,
            LABEL_MAP[result]
        ] = 1


    return np.mean(
        np.sum(
            (
                probabilities
                -
                actual_matrix
            )
            ** 2,
            axis=1,
        )
    )


def calculate_metrics(
    actual_results,
    probabilities,
):

    predictions_numeric = np.argmax(
        probabilities,
        axis=1,
    )


    predictions = np.array(
        [
            REVERSE_LABEL_MAP[
                value
            ]
            for value in predictions_numeric
        ]
    )


    actual_array = np.array(
        actual_results
    )


    accuracy = np.mean(
        predictions
        ==
        actual_array
    )


    numeric_actual = np.array(
        [
            LABEL_MAP[
                value
            ]
            for value in actual_results
        ]
    )


    loss = log_loss(
        numeric_actual,
        probabilities,
        labels=[
            0,
            1,
            2,
        ],
    )


    brier = multiclass_brier(
        actual_results,
        probabilities,
    )


    return {
        "accuracy":
            accuracy,

        "log_loss":
            loss,

        "brier":
            brier,

        "predictions":
            predictions,
    }


# --------------------------------------------------
# Load features
# --------------------------------------------------

df = pd.read_csv(
    FEATURE_FILE
)


df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce",
)


# --------------------------------------------------
# Load market data
# --------------------------------------------------

market = pd.read_csv(
    MARKET_FILE
)


market["Date"] = pd.to_datetime(
    market["Date"],
    errors="coerce",
)


# --------------------------------------------------
# Join market probabilities
# --------------------------------------------------

df = df.merge(
    market[
        [
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",
            "MarketHomeProbability",
            "MarketDrawProbability",
            "MarketAwayProbability",
        ]
    ],

    on=[
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
    ],

    how="inner",
)


df = df.sort_values(
    "Date"
).reset_index(
    drop=True
)


SEASONS = sorted(
    df[
        "Season"
    ].unique()
)


print()
print("MODEL 4 WALK-FORWARD TEST")
print("=========================")

print(
    "Seasons:",
    SEASONS
)


# --------------------------------------------------
# Walk-forward predictions
# --------------------------------------------------

all_results = []


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
        df[
            "Season"
        ].isin(
            training_seasons
        )
    ].copy()


    test = df[
        df[
            "Season"
        ]
        ==
        test_season
    ].copy()


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


    # ==================================================
    # MODEL 2
    #
    # Rebuild Model 2 on the SAME SAMPLE.
    # ==================================================

    model2_home = PoissonRegressor(
        alpha=0.1,
        max_iter=1000,
    )


    model2_away = PoissonRegressor(
        alpha=0.1,
        max_iter=1000,
    )


    model2_home.fit(
        train[
            MODEL_2_FEATURES
        ],
        train[
            "HomeGoals"
        ],
    )


    model2_away.fit(
        train[
            MODEL_2_FEATURES
        ],
        train[
            "AwayGoals"
        ],
    )


    model2_home_xg = (
        model2_home.predict(
            test[
                MODEL_2_FEATURES
            ]
        )
    )


    model2_away_xg = (
        model2_away.predict(
            test[
                MODEL_2_FEATURES
            ]
        )
    )


    model2_probabilities = []


    for (
        home_xg,
        away_xg,
    ) in zip(
        model2_home_xg,
        model2_away_xg,
    ):

        model2_probabilities.append(
            poisson_probabilities(
                home_xg,
                away_xg,
            )
        )


    model2_probabilities = np.array(
        model2_probabilities
    )


    # ==================================================
    # MODEL 4
    #
    # Direct multiclass boosted-tree model.
    # ==================================================

    y_train = (
        train.apply(
            lambda row:
                LABEL_MAP[
                    actual_result(
                        row[
                            "HomeGoals"
                        ],
                        row[
                            "AwayGoals"
                        ],
                    )
                ],
            axis=1,
        )
    )


    model4 = (
        HistGradientBoostingClassifier(

            loss="log_loss",

            learning_rate=0.05,

            max_iter=150,

            max_leaf_nodes=15,

            min_samples_leaf=20,

            l2_regularization=1.0,

            random_state=42,
        )
    )


    model4.fit(
        train[
            MODEL_4_FEATURES
        ],
        y_train,
    )


    raw_model4_probabilities = (
        model4.predict_proba(
            test[
                MODEL_4_FEATURES
            ]
        )
    )


    # --------------------------------------------------
    # Ensure probability columns are always:
    # H, D, A = class 0,1,2
    # --------------------------------------------------

    model4_probabilities = np.zeros(
        (
            len(test),
            3,
        )
    )


    for class_position, class_value in enumerate(
        model4.classes_
    ):

        model4_probabilities[
            :,
            int(
                class_value
            )
        ] = (
            raw_model4_probabilities[
                :,
                class_position
            ]
        )


    # ==================================================
    # MARKET
    # ==================================================

    market_probabilities = (
        test[
            [
                "MarketHomeProbability",
                "MarketDrawProbability",
                "MarketAwayProbability",
            ]
        ]
        .to_numpy()
    )


    # ==================================================
    # Store match-level results
    # ==================================================

    test = test.reset_index(
        drop=True
    )


    for i in range(
        len(test)
    ):

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


        all_results.append(
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


                # Model 2

                "M2HomeProbability":
                    model2_probabilities[
                        i,
                        0
                    ],

                "M2DrawProbability":
                    model2_probabilities[
                        i,
                        1
                    ],

                "M2AwayProbability":
                    model2_probabilities[
                        i,
                        2
                    ],


                # Model 4

                "M4HomeProbability":
                    model4_probabilities[
                        i,
                        0
                    ],

                "M4DrawProbability":
                    model4_probabilities[
                        i,
                        1
                    ],

                "M4AwayProbability":
                    model4_probabilities[
                        i,
                        2
                    ],


                # Market

                "MarketHomeProbability":
                    market_probabilities[
                        i,
                        0
                    ],

                "MarketDrawProbability":
                    market_probabilities[
                        i,
                        1
                    ],

                "MarketAwayProbability":
                    market_probabilities[
                        i,
                        2
                    ],
            }
        )


# --------------------------------------------------
# Final dataframe
# --------------------------------------------------

results = pd.DataFrame(
    all_results
)


actual_results = results[
    "ActualResult"
].tolist()


model2_probs = results[
    [
        "M2HomeProbability",
        "M2DrawProbability",
        "M2AwayProbability",
    ]
].to_numpy()


model4_probs = results[
    [
        "M4HomeProbability",
        "M4DrawProbability",
        "M4AwayProbability",
    ]
].to_numpy()


market_probs = results[
    [
        "MarketHomeProbability",
        "MarketDrawProbability",
        "MarketAwayProbability",
    ]
].to_numpy()


# --------------------------------------------------
# Metrics
# --------------------------------------------------

model2_metrics = calculate_metrics(
    actual_results,
    model2_probs,
)


model4_metrics = calculate_metrics(
    actual_results,
    model4_probs,
)


market_metrics = calculate_metrics(
    actual_results,
    market_probs,
)


results[
    "M2PredictedResult"
] = (
    model2_metrics[
        "predictions"
    ]
)


results[
    "M4PredictedResult"
] = (
    model4_metrics[
        "predictions"
    ]
)


results[
    "MarketPredictedResult"
] = (
    market_metrics[
        "predictions"
    ]
)


# --------------------------------------------------
# Overall comparison
# --------------------------------------------------

comparison = pd.DataFrame(
    [
        {
            "Model":
                "Model 2",

            "AccuracyPct":
                model2_metrics[
                    "accuracy"
                ]
                * 100,

            "LogLoss":
                model2_metrics[
                    "log_loss"
                ],

            "Brier":
                model2_metrics[
                    "brier"
                ],
        },

        {
            "Model":
                "Model 4",

            "AccuracyPct":
                model4_metrics[
                    "accuracy"
                ]
                * 100,

            "LogLoss":
                model4_metrics[
                    "log_loss"
                ],

            "Brier":
                model4_metrics[
                    "brier"
                ],
        },

        {
            "Model":
                "Market",

            "AccuracyPct":
                market_metrics[
                    "accuracy"
                ]
                * 100,

            "LogLoss":
                market_metrics[
                    "log_loss"
                ],

            "Brier":
                market_metrics[
                    "brier"
                ],
        },
    ]
)


print()
print("OVERALL COMPARISON")
print("==================")


print(
    comparison.to_string(
        index=False,
        formatters={
            "AccuracyPct":
                "{:.2f}".format,

            "LogLoss":
                "{:.4f}".format,

            "Brier":
                "{:.4f}".format,
        },
    )
)


# --------------------------------------------------
# Seasonal results
# --------------------------------------------------

season_rows = []


for season in sorted(
    results[
        "Season"
    ].unique()
):

    subset = results[
        results[
            "Season"
        ]
        ==
        season
    ].copy()


    actual = subset[
        "ActualResult"
    ].tolist()


    for (
        model_name,
        columns,
    ) in [

        (
            "Model 2",
            [
                "M2HomeProbability",
                "M2DrawProbability",
                "M2AwayProbability",
            ],
        ),

        (
            "Model 4",
            [
                "M4HomeProbability",
                "M4DrawProbability",
                "M4AwayProbability",
            ],
        ),

        (
            "Market",
            [
                "MarketHomeProbability",
                "MarketDrawProbability",
                "MarketAwayProbability",
            ],
        ),
    ]:

        probabilities = (
            subset[
                columns
            ]
            .to_numpy()
        )


        metrics = calculate_metrics(
            actual,
            probabilities,
        )


        season_rows.append(
            {
                "Season":
                    season,

                "Model":
                    model_name,

                "Matches":
                    len(
                        subset
                    ),

                "AccuracyPct":
                    metrics[
                        "accuracy"
                    ]
                    *
                    100,

                "LogLoss":
                    metrics[
                        "log_loss"
                    ],

                "Brier":
                    metrics[
                        "brier"
                    ],
            }
        )


season_results = pd.DataFrame(
    season_rows
)


print()
print("COMPARISON BY SEASON")
print("====================")


print(
    season_results.to_string(
        index=False,
        formatters={
            "AccuracyPct":
                "{:.2f}".format,

            "LogLoss":
                "{:.4f}".format,

            "Brier":
                "{:.4f}".format,
        },
    )
)


# --------------------------------------------------
# Model 4 change vs Model 2
# --------------------------------------------------

print()
print("MODEL 4 CHANGE VS MODEL 2")
print("=========================")


print(
    "Accuracy change: "
    f"{(
        model4_metrics[
            'accuracy'
        ]
        -
        model2_metrics[
            'accuracy'
        ]
    ) * 100:+.2f} percentage points"
)


print(
    "Log loss change: "
    f"{(
        model4_metrics[
            'log_loss'
        ]
        -
        model2_metrics[
            'log_loss'
        ]
    ):+.4f}"
)


print(
    "Brier change: "
    f"{(
        model4_metrics[
            'brier'
        ]
        -
        model2_metrics[
            'brier'
        ]
    ):+.4f}"
)


# --------------------------------------------------
# Model 4 gap to market
# --------------------------------------------------

print()
print("MODEL 4 GAP TO MARKET")
print("=====================")


print(
    "Accuracy gap: "
    f"{(
        market_metrics[
            'accuracy'
        ]
        -
        model4_metrics[
            'accuracy'
        ]
    ) * 100:+.2f} percentage points"
)


print(
    "Log loss gap: "
    f"{(
        model4_metrics[
            'log_loss'
        ]
        -
        market_metrics[
            'log_loss'
        ]
    ):+.4f}"
)


print(
    "Brier gap: "
    f"{(
        model4_metrics[
            'brier'
        ]
        -
        market_metrics[
            'brier'
        ]
    ):+.4f}"
)


# --------------------------------------------------
# Draw behaviour
# --------------------------------------------------

print()
print("PREDICTION DISTRIBUTION")
print("=======================")


for column in [
    "M2PredictedResult",
    "M4PredictedResult",
    "MarketPredictedResult",
]:

    print()
    print(
        column
    )

    print(
        results[
            column
        ].value_counts()
    )


# --------------------------------------------------
# Save
# --------------------------------------------------

Path(
    "data/processed"
).mkdir(
    parents=True,
    exist_ok=True,
)


results.to_csv(
    DETAIL_OUTPUT,
    index=False,
)


comparison.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


season_results.to_csv(
    SEASON_OUTPUT,
    index=False,
)


print()
print("FILES SAVED")
print("===========")

print(
    DETAIL_OUTPUT
)

print(
    SUMMARY_OUTPUT
)

print(
    SEASON_OUTPUT
)