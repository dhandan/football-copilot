from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import poisson

from sklearn.linear_model import (
    PoissonRegressor
)

from sklearn.metrics import (
    log_loss
)


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
    "all_model_comparison_results.csv"
)

SUMMARY_OUTPUT = (
    "data/processed/"
    "all_model_comparison_summary.csv"
)

SEASON_OUTPUT = (
    "data/processed/"
    "all_model_comparison_by_season.csv"
)


# --------------------------------------------------
# Model 1
# --------------------------------------------------

MODEL_1_FEATURES = [

    "HomeRecentGoalsFor",
    "HomeRecentGoalsAgainst",
    "HomeRecentPPG",

    "AwayRecentGoalsFor",
    "AwayRecentGoalsAgainst",
    "AwayRecentPPG",
]


# --------------------------------------------------
# Model 2
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
# Model 3
# --------------------------------------------------

MODEL_3_FEATURES = (

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
# Helpers
# --------------------------------------------------

LABEL_MAP = {
    "H": 0,
    "D": 1,
    "A": 2,
}


def actual_result(
    home_goals,
    away_goals,
):

    if home_goals > away_goals:
        return "H"

    if home_goals == away_goals:
        return "D"

    return "A"


def match_probabilities(
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


    return (
        home / total,
        draw / total,
        away / total,
    )


def predict_result(
    home_probability,
    draw_probability,
    away_probability,
):

    probabilities = {
        "H":
            home_probability,

        "D":
            draw_probability,

        "A":
            away_probability,
    }


    return max(
        probabilities,
        key=probabilities.get,
    )


def multiclass_brier(
    actual,
    probabilities,
):

    actual_matrix = np.zeros(
        (
            len(actual),
            3,
        )
    )


    for i, result in enumerate(
        actual
    ):

        actual_matrix[
            i,
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
    results,
    prefix,
):

    actual = results[
        "ActualResult"
    ]


    predicted = results[
        f"{prefix}PredictedResult"
    ]


    probability_columns = [
        f"{prefix}HomeProbability",
        f"{prefix}DrawProbability",
        f"{prefix}AwayProbability",
    ]


    probabilities = (
        results[
            probability_columns
        ]
        .to_numpy()
    )


    numeric_actual = actual.map(
        LABEL_MAP
    )


    return {

        "Accuracy":
            (
                actual
                ==
                predicted
            ).mean(),

        "LogLoss":
            log_loss(
                numeric_actual,
                probabilities,
                labels=[0, 1, 2],
            ),

        "Brier":
            multiclass_brier(
                actual,
                probabilities,
            ),
    }


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(
    FEATURE_FILE
)


df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce",
)


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


SEASONS = sorted(
    df[
        "Season"
    ].unique()
)


print()
print("BASELINE VS MODEL 1 VS MODEL 2 VS MODEL 3 VS MARKET")
print("===================================================")

print(
    "Seasons:",
    SEASONS
)


all_results = []


# --------------------------------------------------
# Walk-forward
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
        "Training on:",
        ", ".join(
            training_seasons
        ),
    )

    print(
        "Testing on:",
        test_season,
    )


    # ----------------------------------------------
    # Models
    # ----------------------------------------------

    feature_sets = {

        "M1":
            MODEL_1_FEATURES,

        "M2":
            MODEL_2_FEATURES,

        "M3":
            MODEL_3_FEATURES,
    }


    trained_models = {}


    for (
        model_name,
        feature_columns,
    ) in feature_sets.items():

        home_model = (
            PoissonRegressor(
                alpha=0.1,
                max_iter=1000,
            )
        )

        away_model = (
            PoissonRegressor(
                alpha=0.1,
                max_iter=1000,
            )
        )


        home_model.fit(
            train[
                feature_columns
            ],
            train[
                "HomeGoals"
            ],
        )


        away_model.fit(
            train[
                feature_columns
            ],
            train[
                "AwayGoals"
            ],
        )


        trained_models[
            model_name
        ] = {
            "home":
                home_model,

            "away":
                away_model,

            "features":
                feature_columns,
        }


    # ----------------------------------------------
    # Baseline from training seasons only
    # ----------------------------------------------

    training_results = train.apply(
        lambda row:
            actual_result(
                row[
                    "HomeGoals"
                ],
                row[
                    "AwayGoals"
                ],
            ),
        axis=1,
    )


    baseline_home = (
        training_results
        ==
        "H"
    ).mean()


    baseline_draw = (
        training_results
        ==
        "D"
    ).mean()


    baseline_away = (
        training_results
        ==
        "A"
    ).mean()


    baseline_prediction = (
        predict_result(
            baseline_home,
            baseline_draw,
            baseline_away,
        )
    )


    # ----------------------------------------------
    # Generate predictions
    # ----------------------------------------------

    model_predictions = {}


    for (
        model_name,
        model_info,
    ) in trained_models.items():

        features = (
            model_info[
                "features"
            ]
        )


        home_xg = (
            model_info[
                "home"
            ]
            .predict(
                test[
                    features
                ]
            )
        )


        away_xg = (
            model_info[
                "away"
            ]
            .predict(
                test[
                    features
                ]
            )
        )


        model_predictions[
            model_name
        ] = (
            home_xg,
            away_xg,
        )


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


        row_result = {

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


            # Baseline

            "BaselineHomeProbability":
                baseline_home,

            "BaselineDrawProbability":
                baseline_draw,

            "BaselineAwayProbability":
                baseline_away,

            "BaselinePredictedResult":
                baseline_prediction,


            # Market

            "MarketHomeProbability":
                test.loc[
                    i,
                    "MarketHomeProbability"
                ],

            "MarketDrawProbability":
                test.loc[
                    i,
                    "MarketDrawProbability"
                ],

            "MarketAwayProbability":
                test.loc[
                    i,
                    "MarketAwayProbability"
                ],
        }


        row_result[
            "MarketPredictedResult"
        ] = predict_result(
            row_result[
                "MarketHomeProbability"
            ],
            row_result[
                "MarketDrawProbability"
            ],
            row_result[
                "MarketAwayProbability"
            ],
        )


        for model_name in [
            "M1",
            "M2",
            "M3",
        ]:

            (
                home_xg,
                away_xg,
            ) = model_predictions[
                model_name
            ]


            (
                home_probability,
                draw_probability,
                away_probability,
            ) = match_probabilities(
                home_xg[i],
                away_xg[i],
            )


            row_result[
                f"{model_name}HomeProbability"
            ] = home_probability


            row_result[
                f"{model_name}DrawProbability"
            ] = draw_probability


            row_result[
                f"{model_name}AwayProbability"
            ] = away_probability


            row_result[
                f"{model_name}PredictedResult"
            ] = predict_result(
                home_probability,
                draw_probability,
                away_probability,
            )


        all_results.append(
            row_result
        )


# --------------------------------------------------
# Results
# --------------------------------------------------

results = pd.DataFrame(
    all_results
)


# --------------------------------------------------
# Overall comparison
# --------------------------------------------------

comparison_rows = []


for (
    name,
    prefix,
) in [

    (
        "Baseline",
        "Baseline",
    ),

    (
        "Model 1",
        "M1",
    ),

    (
        "Model 2",
        "M2",
    ),

    (
        "Model 3",
        "M3",
    ),

    (
        "Market",
        "Market",
    ),
]:

    metrics = calculate_metrics(
        results,
        prefix,
    )


    comparison_rows.append(
        {
            "Model":
                name,

            "AccuracyPct":
                metrics[
                    "Accuracy"
                ]
                *
                100,

            "LogLoss":
                metrics[
                    "LogLoss"
                ],

            "Brier":
                metrics[
                    "Brier"
                ],
        }
    )


comparison = pd.DataFrame(
    comparison_rows
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
# Seasonal comparison
# --------------------------------------------------

season_rows = []


for season in sorted(
    results[
        "Season"
    ].unique()
):

    season_data = results[
        results[
            "Season"
        ]
        ==
        season
    ].copy()


    for (
        name,
        prefix,
    ) in [

        (
            "Baseline",
            "Baseline",
        ),

        (
            "Model 1",
            "M1",
        ),

        (
            "Model 2",
            "M2",
        ),

        (
            "Model 3",
            "M3",
        ),

        (
            "Market",
            "Market",
        ),
    ]:

        metrics = (
            calculate_metrics(
                season_data,
                prefix,
            )
        )


        season_rows.append(
            {
                "Season":
                    season,

                "Model":
                    name,

                "Matches":
                    len(
                        season_data
                    ),

                "AccuracyPct":
                    metrics[
                        "Accuracy"
                    ]
                    *
                    100,

                "LogLoss":
                    metrics[
                        "LogLoss"
                    ],

                "Brier":
                    metrics[
                        "Brier"
                    ],
            }
        )


season_comparison = pd.DataFrame(
    season_rows
)


print()
print("COMPARISON BY SEASON")
print("====================")


print(
    season_comparison.to_string(
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
# Model 3 changes
# --------------------------------------------------

model2_row = comparison[
    comparison[
        "Model"
    ]
    ==
    "Model 2"
].iloc[0]


model3_row = comparison[
    comparison[
        "Model"
    ]
    ==
    "Model 3"
].iloc[0]


market_row = comparison[
    comparison[
        "Model"
    ]
    ==
    "Market"
].iloc[0]


print()
print("MODEL 3 CHANGE VS MODEL 2")
print("=========================")


print(
    "Accuracy change: "
    f"{(
        model3_row[
            'AccuracyPct'
        ]
        -
        model2_row[
            'AccuracyPct'
        ]
    ):+.2f} percentage points"
)


print(
    "Log loss change: "
    f"{(
        model3_row[
            'LogLoss'
        ]
        -
        model2_row[
            'LogLoss'
        ]
    ):+.4f}"
)


print(
    "Brier change: "
    f"{(
        model3_row[
            'Brier'
        ]
        -
        model2_row[
            'Brier'
        ]
    ):+.4f}"
)


print()
print("MODEL 3 GAP TO MARKET")
print("=====================")


print(
    "Accuracy gap: "
    f"{(
        market_row[
            'AccuracyPct'
        ]
        -
        model3_row[
            'AccuracyPct'
        ]
    ):+.2f} percentage points"
)


print(
    "Log loss gap: "
    f"{(
        model3_row[
            'LogLoss'
        ]
        -
        market_row[
            'LogLoss'
        ]
    ):+.4f}"
)


print(
    "Brier gap: "
    f"{(
        model3_row[
            'Brier'
        ]
        -
        market_row[
            'Brier'
        ]
    ):+.4f}"
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


season_comparison.to_csv(
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