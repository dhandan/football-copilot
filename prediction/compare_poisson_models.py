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

INPUT_FILE = (
    "data/processed/"
    "prediction_features_v2.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "model_comparison_results.csv"
)


# --------------------------------------------------
# Model 1 features
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
# Model 2 features
# --------------------------------------------------

MODEL_2_FEATURES = [

    # Original Model 1

    "HomeRecentGoalsFor",
    "HomeRecentGoalsAgainst",
    "HomeRecentPPG",

    "AwayRecentGoalsFor",
    "AwayRecentGoalsAgainst",
    "AwayRecentPPG",


    # 10-match form

    "Home10GoalsFor",
    "Home10GoalsAgainst",
    "Home10PPG",

    "Away10GoalsFor",
    "Away10GoalsAgainst",
    "Away10PPG",


    # Venue form

    "HomeVenuePPG",
    "HomeVenueGoalsFor",

    "AwayVenuePPG",
    "AwayVenueGoalsFor",


    # Season-to-date

    "HomeSeasonPPG",
    "HomeSeasonGoalDifferencePG",

    "AwaySeasonPPG",
    "AwaySeasonGoalDifferencePG",


    # Relative strength

    "RecentPPGDifference",
    "TenMatchPPGDifference",
    "SeasonPPGDifference",

    "AttackVsDefenceHome",
    "AttackVsDefenceAway",
]


# --------------------------------------------------
# Utility functions
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


def match_probabilities(
    home_expected_goals,
    away_expected_goals,
    max_goals=8,
):

    home_win = 0.0
    draw = 0.0
    away_win = 0.0


    for home_goals in range(
        max_goals + 1
    ):

        home_probability = (
            poisson.pmf(
                home_goals,
                home_expected_goals,
            )
        )


        for away_goals in range(
            max_goals + 1
        ):

            probability = (
                home_probability
                *
                poisson.pmf(
                    away_goals,
                    away_expected_goals,
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


def calculate_metrics(
    results,
    prefix,
):
    """
    Calculate accuracy, log loss and
    multiclass Brier score.
    """

    actual = results[
        "ActualResult"
    ]


    predicted = results[
        f"{prefix}PredictedResult"
    ]


    accuracy = (
        actual == predicted
    ).mean()


    label_map = {
        "H": 0,
        "D": 1,
        "A": 2,
    }


    y_true = actual.map(
        label_map
    )


    y_prob = results[
        [
            f"{prefix}HomeProbability",
            f"{prefix}DrawProbability",
            f"{prefix}AwayProbability",
        ]
    ].to_numpy()


    loss = log_loss(
        y_true,
        y_prob,
        labels=[
            0,
            1,
            2,
        ],
    )


    actual_matrix = np.zeros(
        (
            len(results),
            3,
        )
    )


    for row_number, result in enumerate(
        actual
    ):

        actual_matrix[
            row_number,
            label_map[result]
        ] = 1


    brier = np.mean(
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


    return {
        "accuracy":
            accuracy,

        "log_loss":
            loss,

        "brier":
            brier,
    }


# --------------------------------------------------
# Load feature data
# --------------------------------------------------

df = pd.read_csv(
    INPUT_FILE
)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce",
)

df = df.sort_values(
    "Date"
).reset_index(
    drop=True
)


SEASONS = sorted(
    df["Season"].unique()
)


print()
print("MODEL 1 VS MODEL 2")
print("==================")

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
    # MODEL 1
    # ----------------------------------------------

    model1_home = (
        PoissonRegressor(
            alpha=0.1,
            max_iter=1000,
        )
    )

    model1_away = (
        PoissonRegressor(
            alpha=0.1,
            max_iter=1000,
        )
    )


    model1_home.fit(
        train[
            MODEL_1_FEATURES
        ],
        train[
            "HomeGoals"
        ],
    )


    model1_away.fit(
        train[
            MODEL_1_FEATURES
        ],
        train[
            "AwayGoals"
        ],
    )


    model1_home_xg = (
        model1_home.predict(
            test[
                MODEL_1_FEATURES
            ]
        )
    )


    model1_away_xg = (
        model1_away.predict(
            test[
                MODEL_1_FEATURES
            ]
        )
    )


    # ----------------------------------------------
    # MODEL 2
    # ----------------------------------------------

    model2_home = (
        PoissonRegressor(
            alpha=0.1,
            max_iter=1000,
        )
    )

    model2_away = (
        PoissonRegressor(
            alpha=0.1,
            max_iter=1000,
        )
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


    # ----------------------------------------------
    # Walk-forward baseline
    #
    # IMPORTANT:
    # Frequencies come only from training seasons,
    # not from the test season.
    # ----------------------------------------------

    train_actual_results = []


    for _, train_row in train.iterrows():

        train_actual_results.append(
            actual_result(
                train_row[
                    "HomeGoals"
                ],
                train_row[
                    "AwayGoals"
                ],
            )
        )


    train_actual_results = (
        pd.Series(
            train_actual_results
        )
    )


    baseline_home = (
        train_actual_results
        ==
        "H"
    ).mean()


    baseline_draw = (
        train_actual_results
        ==
        "D"
    ).mean()


    baseline_away = (
        train_actual_results
        ==
        "A"
    ).mean()


    baseline_probabilities = {
        "H":
            baseline_home,

        "D":
            baseline_draw,

        "A":
            baseline_away,
    }


    baseline_predicted = max(
        baseline_probabilities,
        key=baseline_probabilities.get,
    )


    # ----------------------------------------------
    # Match-by-match predictions
    # ----------------------------------------------

    test = test.reset_index(
        drop=True
    )


    for i in range(
        len(test)
    ):

        # Model 1 probabilities

        (
            m1_home_probability,
            m1_draw_probability,
            m1_away_probability,
        ) = match_probabilities(
            model1_home_xg[i],
            model1_away_xg[i],
        )


        m1_probabilities = {
            "H":
                m1_home_probability,

            "D":
                m1_draw_probability,

            "A":
                m1_away_probability,
        }


        m1_prediction = max(
            m1_probabilities,
            key=m1_probabilities.get,
        )


        # Model 2 probabilities

        (
            m2_home_probability,
            m2_draw_probability,
            m2_away_probability,
        ) = match_probabilities(
            model2_home_xg[i],
            model2_away_xg[i],
        )


        m2_probabilities = {
            "H":
                m2_home_probability,

            "D":
                m2_draw_probability,

            "A":
                m2_away_probability,
        }


        m2_prediction = max(
            m2_probabilities,
            key=m2_probabilities.get,
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


                # Model 1

                "M1HomeProbability":
                    m1_home_probability,

                "M1DrawProbability":
                    m1_draw_probability,

                "M1AwayProbability":
                    m1_away_probability,

                "M1PredictedResult":
                    m1_prediction,

                "M1ExpectedHomeGoals":
                    model1_home_xg[i],

                "M1ExpectedAwayGoals":
                    model1_away_xg[i],


                # Model 2

                "M2HomeProbability":
                    m2_home_probability,

                "M2DrawProbability":
                    m2_draw_probability,

                "M2AwayProbability":
                    m2_away_probability,

                "M2PredictedResult":
                    m2_prediction,

                "M2ExpectedHomeGoals":
                    model2_home_xg[i],

                "M2ExpectedAwayGoals":
                    model2_away_xg[i],


                # Baseline

                "BaselineHomeProbability":
                    baseline_home,

                "BaselineDrawProbability":
                    baseline_draw,

                "BaselineAwayProbability":
                    baseline_away,

                "BaselinePredictedResult":
                    baseline_predicted,
            }
        )


# --------------------------------------------------
# Results dataframe
# --------------------------------------------------

results = pd.DataFrame(
    all_results
)


# --------------------------------------------------
# Overall metrics
# --------------------------------------------------

model1_metrics = (
    calculate_metrics(
        results,
        "M1",
    )
)


model2_metrics = (
    calculate_metrics(
        results,
        "M2",
    )
)


baseline_metrics = (
    calculate_metrics(
        results,
        "Baseline",
    )
)


# --------------------------------------------------
# Print overall comparison
# --------------------------------------------------

print()
print("OVERALL COMPARISON")
print("==================")


comparison = pd.DataFrame(
    [
        {
            "Model":
                "Baseline",

            "AccuracyPct":
                baseline_metrics[
                    "accuracy"
                ]
                * 100,

            "LogLoss":
                baseline_metrics[
                    "log_loss"
                ],

            "Brier":
                baseline_metrics[
                    "brier"
                ],
        },

        {
            "Model":
                "Model 1",

            "AccuracyPct":
                model1_metrics[
                    "accuracy"
                ]
                * 100,

            "LogLoss":
                model1_metrics[
                    "log_loss"
                ],

            "Brier":
                model1_metrics[
                    "brier"
                ],
        },

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
    ]
)


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
# Seasonal comparisons
# --------------------------------------------------

print()
print("COMPARISON BY SEASON")
print("====================")


season_rows = []


for season in sorted(
    results[
        "Season"
    ].unique()
):

    season_data = results[
        results["Season"] == season
    ].copy()


    m1 = calculate_metrics(
        season_data,
        "M1",
    )


    m2 = calculate_metrics(
        season_data,
        "M2",
    )


    baseline = calculate_metrics(
        season_data,
        "Baseline",
    )


    season_rows.append(
        {
            "Season":
                season,

            "Matches":
                len(
                    season_data
                ),

            "BaselineAccuracy":
                baseline[
                    "accuracy"
                ]
                * 100,

            "Model1Accuracy":
                m1[
                    "accuracy"
                ]
                * 100,

            "Model2Accuracy":
                m2[
                    "accuracy"
                ]
                * 100,

            "Model1LogLoss":
                m1[
                    "log_loss"
                ],

            "Model2LogLoss":
                m2[
                    "log_loss"
                ],

            "Model1Brier":
                m1[
                    "brier"
                ],

            "Model2Brier":
                m2[
                    "brier"
                ],
        }
    )


season_comparison = pd.DataFrame(
    season_rows
)


print(
    season_comparison.to_string(
        index=False,
        formatters={

            "BaselineAccuracy":
                "{:.2f}".format,

            "Model1Accuracy":
                "{:.2f}".format,

            "Model2Accuracy":
                "{:.2f}".format,

            "Model1LogLoss":
                "{:.4f}".format,

            "Model2LogLoss":
                "{:.4f}".format,

            "Model1Brier":
                "{:.4f}".format,

            "Model2Brier":
                "{:.4f}".format,
        },
    )
)


# --------------------------------------------------
# Improvement calculations
# --------------------------------------------------

accuracy_change = (
    (
        model2_metrics[
            "accuracy"
        ]
        -
        model1_metrics[
            "accuracy"
        ]
    )
    * 100
)


log_loss_change = (
    model2_metrics[
        "log_loss"
    ]
    -
    model1_metrics[
        "log_loss"
    ]
)


brier_change = (
    model2_metrics[
        "brier"
    ]
    -
    model1_metrics[
        "brier"
    ]
)


print()
print("MODEL 2 CHANGE VS MODEL 1")
print("=========================")

print(
    f"Accuracy change: "
    f"{accuracy_change:+.2f} percentage points"
)

print(
    f"Log loss change: "
    f"{log_loss_change:+.4f}"
)

print(
    f"Brier change: "
    f"{brier_change:+.4f}"
)


# --------------------------------------------------
# Automatic verdict
# --------------------------------------------------

model2_better_log_loss = (
    model2_metrics[
        "log_loss"
    ]
    <
    model1_metrics[
        "log_loss"
    ]
)


model2_better_brier = (
    model2_metrics[
        "brier"
    ]
    <
    model1_metrics[
        "brier"
    ]
)


print()
print("VERDICT")
print("=======")


if (
    model2_better_log_loss
    and
    model2_better_brier
):

    print(
        "Model 2 improves both probability metrics."
    )

    print(
        "Model 2 is the stronger candidate "
        "for the next stage."
    )


elif (
    model2_better_log_loss
    or
    model2_better_brier
):

    print(
        "Model 2 gives mixed results."
    )

    print(
        "Review the seasonal comparison before "
        "selecting a production model."
    )


else:

    print(
        "Model 2 does not improve the probability "
        "metrics."
    )

    print(
        "Keep Model 1 as the stronger baseline."
    )


# --------------------------------------------------
# Save match-level results
# --------------------------------------------------

Path(
    "data/processed"
).mkdir(
    parents=True,
    exist_ok=True,
)


results.to_csv(
    OUTPUT_FILE,
    index=False,
)


comparison.to_csv(
    "data/processed/"
    "model_comparison_summary.csv",
    index=False,
)


season_comparison.to_csv(
    "data/processed/"
    "model_comparison_by_season.csv",
    index=False,
)


print()
print("FILES SAVED")
print("===========")

print(
    OUTPUT_FILE
)

print(
    "data/processed/"
    "model_comparison_summary.csv"
)

print(
    "data/processed/"
    "model_comparison_by_season.csv"
)