from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar

from sklearn.linear_model import (
    PoissonRegressor
)

from sklearn.metrics import (
    log_loss,
)


# --------------------------------------------------
# Project setup
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


from prediction.dixon_coles import (
    independent_poisson_probabilities,
    match_probabilities,
)


# --------------------------------------------------
# Files
# --------------------------------------------------

INPUT_FILE = (
    "data/processed/"
    "prediction_features_v2.csv"
)

RESULTS_FILE = (
    "data/processed/"
    "model3_dixon_coles_backtest.csv"
)

SUMMARY_FILE = (
    "data/processed/"
    "model3_dixon_coles_summary.csv"
)

SEASON_FILE = (
    "data/processed/"
    "model3_dixon_coles_by_season.csv"
)


# --------------------------------------------------
# Model 2 features
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
# Result helpers
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


def predicted_result(
    home_probability,
    draw_probability,
    away_probability,
):

    probabilities = {
        "H": home_probability,
        "D": draw_probability,
        "A": away_probability,
    }

    return max(
        probabilities,
        key=probabilities.get,
    )


# --------------------------------------------------
# Fit rho on training data only
# --------------------------------------------------

def fit_rho(
    home_xg,
    away_xg,
    home_goals,
    away_goals,
):
    """
    Estimate Dixon-Coles rho from training matches.

    The xG estimates supplied here come from
    Model 2's Poisson regressors.

    rho is fitted using training data only.
    """

    home_xg = np.asarray(
        home_xg,
        dtype=float,
    )

    away_xg = np.asarray(
        away_xg,
        dtype=float,
    )

    home_goals = np.asarray(
        home_goals,
        dtype=int,
    )

    away_goals = np.asarray(
        away_goals,
        dtype=int,
    )

    def negative_log_likelihood(
        rho,
    ):

        total_log_probability = 0.0

        for (
            hxg,
            axg,
            hg,
            ag,
        ) in zip(
            home_xg,
            away_xg,
            home_goals,
            away_goals,
        ):

            probabilities = (
                match_probabilities(
                    home_xg=hxg,
                    away_xg=axg,
                    rho=rho,
                    max_goals=8,
                )
            )

            matrix = probabilities[
                "score_matrix"
            ]

            if (
                hg < matrix.shape[0]
                and
                ag < matrix.shape[1]
            ):

                probability = matrix[
                    hg,
                    ag,
                ]

            else:
                probability = 1e-12

            probability = max(
                float(probability),
                1e-12,
            )

            total_log_probability += (
                np.log(
                    probability
                )
            )

        return -total_log_probability

    optimisation = minimize_scalar(
        negative_log_likelihood,
        bounds=(-0.25, 0.25),
        method="bounded",
        options={
            "xatol": 1e-5,
        },
    )

    if not optimisation.success:
        raise RuntimeError(
            "Dixon-Coles rho optimisation failed."
        )

    return float(
        optimisation.x
    )


# --------------------------------------------------
# Metrics
# --------------------------------------------------

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

    probabilities = results[
        [
            f"{prefix}HomeProbability",
            f"{prefix}DrawProbability",
            f"{prefix}AwayProbability",
        ]
    ].to_numpy()

    loss = log_loss(
        y_true,
        probabilities,
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

    for index, result in enumerate(
        actual
    ):

        actual_matrix[
            index,
            label_map[result],
        ] = 1.0

    brier = np.mean(
        np.sum(
            (
                probabilities
                - actual_matrix
            )
            ** 2,
            axis=1,
        )
    )

    actual_draw = (
        actual == "D"
    )

    predicted_draw = (
        predicted == "D"
    )

    actual_draw_count = int(
        actual_draw.sum()
    )

    predicted_draw_count = int(
        predicted_draw.sum()
    )

    if actual_draw_count > 0:

        draw_recall = (
            (
                actual_draw
                &
                predicted_draw
            ).sum()
            /
            actual_draw_count
        )

    else:

        draw_recall = np.nan

    draw_probabilities = results[
        f"{prefix}DrawProbability"
    ]

    mean_draw_probability = (
        draw_probabilities.mean()
    )

    if actual_draw.any():

        mean_draw_probability_actual_draws = (
            draw_probabilities[
                actual_draw
            ].mean()
        )

    else:

        mean_draw_probability_actual_draws = (
            np.nan
        )

    if (~actual_draw).any():

        mean_draw_probability_non_draws = (
            draw_probabilities[
                ~actual_draw
            ].mean()
        )

    else:

        mean_draw_probability_non_draws = (
            np.nan
        )

    modal_one_one_count = int(
        results[
            f"{prefix}ModalScore"
        ]
        .astype(str)
        .eq("1-1")
        .sum()
    )

    return {
        "AccuracyPct":
            accuracy * 100,

        "LogLoss":
            loss,

        "Brier":
            brier,

        "PredictedDraws":
            predicted_draw_count,

        "PredictedDrawPct":
            (
                predicted_draw_count
                /
                len(results)
                * 100
            ),

        "ActualDraws":
            actual_draw_count,

        "ActualDrawPct":
            (
                actual_draw_count
                /
                len(results)
                * 100
            ),

        "DrawRecallPct":
            draw_recall * 100,

        "MeanDrawProbabilityPct":
            mean_draw_probability * 100,

        "MeanDrawProbabilityActualDrawsPct":
            (
                mean_draw_probability_actual_draws
                * 100
            ),

        "MeanDrawProbabilityNonDrawsPct":
            (
                mean_draw_probability_non_draws
                * 100
            ),

        "ModalOneOneCount":
            modal_one_one_count,

        "ModalOneOnePct":
            (
                modal_one_one_count
                /
                len(results)
                * 100
            ),
    }


# --------------------------------------------------
# Load data
# --------------------------------------------------

print()
print("FOOTBALL COPILOT")
print("MODEL 3A: DIXON-COLES SHADOW CHALLENGER")
print("========================================")

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
print(
    "Seasons:",
    SEASONS,
)


# --------------------------------------------------
# Walk-forward backtest
# --------------------------------------------------

all_results = []
season_rhos = []


for test_index in range(
    2,
    len(SEASONS),
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
    # Fit the exact Model 2 goal regressors
    # ----------------------------------------------

    home_model = PoissonRegressor(
        alpha=0.1,
        max_iter=1000,
    )

    away_model = PoissonRegressor(
        alpha=0.1,
        max_iter=1000,
    )

    home_model.fit(
        train[
            MODEL_2_FEATURES
        ],
        train[
            "HomeGoals"
        ],
    )

    away_model.fit(
        train[
            MODEL_2_FEATURES
        ],
        train[
            "AwayGoals"
        ],
    )

    # ----------------------------------------------
    # Training xG used ONLY to estimate rho
    # ----------------------------------------------

    train_home_xg = (
        home_model.predict(
            train[
                MODEL_2_FEATURES
            ]
        )
    )

    train_away_xg = (
        away_model.predict(
            train[
                MODEL_2_FEATURES
            ]
        )
    )

    rho = fit_rho(
        home_xg=train_home_xg,
        away_xg=train_away_xg,
        home_goals=train[
            "HomeGoals"
        ].to_numpy(),
        away_goals=train[
            "AwayGoals"
        ].to_numpy(),
    )

    season_rhos.append(
        {
            "TestSeason":
                test_season,

            "Rho":
                rho,
        }
    )

    print(
        f"Fitted rho: {rho:.5f}"
    )

    # ----------------------------------------------
    # Test xG
    # ----------------------------------------------

    test = test.reset_index(
        drop=True
    )

    test_home_xg = (
        home_model.predict(
            test[
                MODEL_2_FEATURES
            ]
        )
    )

    test_away_xg = (
        away_model.predict(
            test[
                MODEL_2_FEATURES
            ]
        )
    )

    # ----------------------------------------------
    # Match-by-match comparison
    # ----------------------------------------------

    for index in range(
        len(test)
    ):

        home_xg = float(
            test_home_xg[index]
        )

        away_xg = float(
            test_away_xg[index]
        )

        model2 = (
            independent_poisson_probabilities(
                home_xg=home_xg,
                away_xg=away_xg,
            )
        )

        model3 = (
            match_probabilities(
                home_xg=home_xg,
                away_xg=away_xg,
                rho=rho,
            )
        )

        actual = actual_result(
            test.loc[
                index,
                "HomeGoals",
            ],
            test.loc[
                index,
                "AwayGoals",
            ],
        )

        model2_prediction = (
            predicted_result(
                model2[
                    "home_probability"
                ],
                model2[
                    "draw_probability"
                ],
                model2[
                    "away_probability"
                ],
            )
        )

        model3_prediction = (
            predicted_result(
                model3[
                    "home_probability"
                ],
                model3[
                    "draw_probability"
                ],
                model3[
                    "away_probability"
                ],
            )
        )

        actual_score = (
            f"{int(test.loc[index, 'HomeGoals'])}-"
            f"{int(test.loc[index, 'AwayGoals'])}"
        )

        all_results.append(
            {
                "Season":
                    test_season,

                "Date":
                    test.loc[
                        index,
                        "Date",
                    ],

                "HomeTeam":
                    test.loc[
                        index,
                        "HomeTeam",
                    ],

                "AwayTeam":
                    test.loc[
                        index,
                        "AwayTeam",
                    ],

                "ActualHomeGoals":
                    test.loc[
                        index,
                        "HomeGoals",
                    ],

                "ActualAwayGoals":
                    test.loc[
                        index,
                        "AwayGoals",
                    ],

                "ActualScore":
                    actual_score,

                "ActualResult":
                    actual,

                "ExpectedHomeGoals":
                    home_xg,

                "ExpectedAwayGoals":
                    away_xg,

                "Rho":
                    rho,

                # Model 2

                "M2HomeProbability":
                    model2[
                        "home_probability"
                    ],

                "M2DrawProbability":
                    model2[
                        "draw_probability"
                    ],

                "M2AwayProbability":
                    model2[
                        "away_probability"
                    ],

                "M2PredictedResult":
                    model2_prediction,

                "M2ModalScore":
                    model2[
                        "modal_score"
                    ],

                # Model 3

                "M3HomeProbability":
                    model3[
                        "home_probability"
                    ],

                "M3DrawProbability":
                    model3[
                        "draw_probability"
                    ],

                "M3AwayProbability":
                    model3[
                        "away_probability"
                    ],

                "M3PredictedResult":
                    model3_prediction,

                "M3ModalScore":
                    model3[
                        "modal_score"
                    ],
            }
        )


# --------------------------------------------------
# Results
# --------------------------------------------------

results = pd.DataFrame(
    all_results
)

model2_metrics = calculate_metrics(
    results,
    "M2",
)

model3_metrics = calculate_metrics(
    results,
    "M3",
)


# --------------------------------------------------
# Actual 1-1 diagnostics
# --------------------------------------------------

actual_one_one_count = int(
    results[
        "ActualScore"
    ]
    .astype(str)
    .eq("1-1")
    .sum()
)

actual_one_one_pct = (
    actual_one_one_count
    /
    len(results)
    * 100
)


# --------------------------------------------------
# Goal MAE
#
# Both models intentionally share Model 2 xG.
# Therefore goal MAE is identical.
# --------------------------------------------------

home_goal_mae = np.mean(
    np.abs(
        results[
            "ExpectedHomeGoals"
        ]
        -
        results[
            "ActualHomeGoals"
        ]
    )
)

away_goal_mae = np.mean(
    np.abs(
        results[
            "ExpectedAwayGoals"
        ]
        -
        results[
            "ActualAwayGoals"
        ]
    )
)

total_goal_mae = np.mean(
    np.abs(
        (
            results[
                "ExpectedHomeGoals"
            ]
            +
            results[
                "ExpectedAwayGoals"
            ]
        )
        -
        (
            results[
                "ActualHomeGoals"
            ]
            +
            results[
                "ActualAwayGoals"
            ]
        )
    )
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

summary = pd.DataFrame(
    [
        {
            "Model":
                "Model 2",

            **model2_metrics,
        },

        {
            "Model":
                "Model 3A Dixon-Coles",

            **model3_metrics,
        },
    ]
)

summary[
    "ActualOneOneCount"
] = actual_one_one_count

summary[
    "ActualOneOnePct"
] = actual_one_one_pct

summary[
    "HomeGoalMAE"
] = home_goal_mae

summary[
    "AwayGoalMAE"
] = away_goal_mae

summary[
    "TotalGoalMAE"
] = total_goal_mae


# --------------------------------------------------
# By-season comparison
# --------------------------------------------------

season_rows = []

for season in sorted(
    results[
        "Season"
    ].unique()
):

    season_data = results[
        results["Season"]
        ==
        season
    ].copy()

    m2 = calculate_metrics(
        season_data,
        "M2",
    )

    m3 = calculate_metrics(
        season_data,
        "M3",
    )

    season_rows.append(
        {
            "Season":
                season,

            "Matches":
                len(
                    season_data
                ),

            "Rho":
                season_data[
                    "Rho"
                ].iloc[0],

            "M2AccuracyPct":
                m2[
                    "AccuracyPct"
                ],

            "M3AccuracyPct":
                m3[
                    "AccuracyPct"
                ],

            "M2LogLoss":
                m2[
                    "LogLoss"
                ],

            "M3LogLoss":
                m3[
                    "LogLoss"
                ],

            "M2Brier":
                m2[
                    "Brier"
                ],

            "M3Brier":
                m3[
                    "Brier"
                ],

            "M2PredictedDrawPct":
                m2[
                    "PredictedDrawPct"
                ],

            "M3PredictedDrawPct":
                m3[
                    "PredictedDrawPct"
                ],

            "ActualDrawPct":
                m3[
                    "ActualDrawPct"
                ],

            "M2ModalOneOnePct":
                m2[
                    "ModalOneOnePct"
                ],

            "M3ModalOneOnePct":
                m3[
                    "ModalOneOnePct"
                ],
        }
    )


season_summary = pd.DataFrame(
    season_rows
)


# --------------------------------------------------
# Console output
# --------------------------------------------------

print()
print("OVERALL COMPARISON")
print("==================")

display_columns = [
    "Model",
    "AccuracyPct",
    "LogLoss",
    "Brier",
    "PredictedDraws",
    "PredictedDrawPct",
    "ActualDraws",
    "ActualDrawPct",
    "DrawRecallPct",
    "MeanDrawProbabilityPct",
    "MeanDrawProbabilityActualDrawsPct",
    "MeanDrawProbabilityNonDrawsPct",
    "ModalOneOneCount",
    "ModalOneOnePct",
]

print(
    summary[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)


print()
print("SCORELINE DIAGNOSTICS")
print("=====================")

print(
    f"Actual 1-1 results: "
    f"{actual_one_one_count} "
    f"({actual_one_one_pct:.2f}%)"
)

print(
    f"Model 2 modal 1-1: "
    f"{model2_metrics['ModalOneOneCount']} "
    f"({model2_metrics['ModalOneOnePct']:.2f}%)"
)

print(
    f"Model 3 modal 1-1: "
    f"{model3_metrics['ModalOneOneCount']} "
    f"({model3_metrics['ModalOneOnePct']:.2f}%)"
)


print()
print("GOAL MAE")
print("========")

print(
    f"Home goal MAE:  "
    f"{home_goal_mae:.4f}"
)

print(
    f"Away goal MAE:  "
    f"{away_goal_mae:.4f}"
)

print(
    f"Total goal MAE: "
    f"{total_goal_mae:.4f}"
)

print(
    "Note: Model 2 and Model 3A intentionally "
    "share the same xG estimates, so goal MAE "
    "is identical."
)


print()
print("DIXON-COLES RHO BY TEST SEASON")
print("==============================")

print(
    pd.DataFrame(
        season_rhos
    ).to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.5f}"
        ),
    )
)


print()
print("COMPARISON BY SEASON")
print("====================")

print(
    season_summary.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)


# --------------------------------------------------
# Verdict
# --------------------------------------------------

print()
print("EXPERIMENT VERDICT")
print("==================")

model3_better_log_loss = (
    model3_metrics[
        "LogLoss"
    ]
    <
    model2_metrics[
        "LogLoss"
    ]
)

model3_better_brier = (
    model3_metrics[
        "Brier"
    ]
    <
    model2_metrics[
        "Brier"
    ]
)

if (
    model3_better_log_loss
    and
    model3_better_brier
):

    print(
        "Model 3A improves both probability "
        "quality metrics versus Model 2."
    )

    print(
        "It is suitable for consideration as "
        "a prospective shadow challenger."
    )

elif (
    model3_better_log_loss
    or
    model3_better_brier
):

    print(
        "Model 3A produces mixed probability "
        "quality results."
    )

    print(
        "Review draw diagnostics and seasonal "
        "stability before shadow deployment."
    )

else:

    print(
        "Model 3A does not improve either "
        "probability quality metric."
    )

    print(
        "Do not promote it solely because its "
        "draw or scoreline behaviour looks "
        "different."
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
    RESULTS_FILE,
    index=False,
)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)

season_summary.to_csv(
    SEASON_FILE,
    index=False,
)


print()
print("FILES SAVED")
print("===========")

print(
    RESULTS_FILE
)

print(
    SUMMARY_FILE
)

print(
    SEASON_FILE
)

print()
print("MODEL 3A BACKTEST COMPLETE")
print("==========================")