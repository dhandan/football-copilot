from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import poisson

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import log_loss


# --------------------------------------------------
# Files
# --------------------------------------------------

INPUT_FILE = (
    "data/processed/"
    "prediction_features_v2.csv"
)

DETAIL_FILE = (
    "reports/post_gw5/"
    "challenger_a_predictions.csv"
)

SUMMARY_FILE = (
    "reports/post_gw5/"
    "challenger_a_summary.csv"
)

SEASON_FILE = (
    "reports/post_gw5/"
    "challenger_a_by_season.csv"
)

GW6_FILE = (
    "reports/post_gw5/"
    "challenger_a_historical_gw6.csv"
)


# --------------------------------------------------
# Frozen Model 2 feature set
# --------------------------------------------------

FEATURES = [

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


LABEL_MAP = {
    "H": 0,
    "D": 1,
    "A": 2,
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


def outcome_probabilities(
    home_xg,
    away_xg,
    max_goals=8,
):

    home_xg = max(
        float(home_xg),
        0.05,
    )

    away_xg = max(
        float(away_xg),
        0.05,
    )

    home_probability = 0.0
    draw_probability = 0.0
    away_probability = 0.0

    modal_probability = -1.0
    modal_score = None

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

                home_probability += (
                    probability
                )

            elif home_goals == away_goals:

                draw_probability += (
                    probability
                )

            else:

                away_probability += (
                    probability
                )

            if probability > modal_probability:

                modal_probability = (
                    probability
                )

                modal_score = (
                    f"{home_goals}-"
                    f"{away_goals}"
                )

    total = (
        home_probability
        +
        draw_probability
        +
        away_probability
    )

    return {
        "home_probability":
            home_probability / total,

        "draw_probability":
            draw_probability / total,

        "away_probability":
            away_probability / total,

        "modal_score":
            modal_score,
    }


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


def calculate_metrics(
    data,
    prefix,
):

    actual = data[
        "ActualResult"
    ]

    predicted = data[
        f"{prefix}PredictedResult"
    ]

    probabilities = data[
        [
            f"{prefix}HomeProbability",
            f"{prefix}DrawProbability",
            f"{prefix}AwayProbability",
        ]
    ].to_numpy()

    numeric_actual = actual.map(
        LABEL_MAP
    )

    accuracy = (
        actual == predicted
    ).mean()

    loss = log_loss(
        numeric_actual,
        probabilities,
        labels=[
            0,
            1,
            2,
        ],
    )

    actual_matrix = np.zeros(
        (
            len(data),
            3,
        )
    )

    for index, result in enumerate(
        actual
    ):

        actual_matrix[
            index,
            LABEL_MAP[result],
        ] = 1.0

    brier = np.mean(
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

    predicted_draws = int(
        (
            predicted == "D"
        ).sum()
    )

    actual_draws = int(
        (
            actual == "D"
        ).sum()
    )

    modal_one_one = int(
        data[
            f"{prefix}ModalScore"
        ]
        .astype(str)
        .eq("1-1")
        .sum()
    )

    xg_gap = (
        data[
            f"{prefix}HomeXG"
        ]
        -
        data[
            f"{prefix}AwayXG"
        ]
    ).abs()

    total_xg = (
        data[
            f"{prefix}HomeXG"
        ]
        +
        data[
            f"{prefix}AwayXG"
        ]
    )

    return {
        "Matches":
            len(data),

        "AccuracyPct":
            accuracy * 100,

        "LogLoss":
            loss,

        "Brier":
            brier,

        "PredictedDraws":
            predicted_draws,

        "ActualDraws":
            actual_draws,

        "ModalOneOneCount":
            modal_one_one,

        "ModalOneOnePct":
            (
                modal_one_one
                /
                len(data)
                *
                100
            ),

        "MeanXGGap":
            xg_gap.mean(),

        "XGGapStd":
            xg_gap.std(),

        "MeanTotalXG":
            total_xg.mean(),

        "TotalXGStd":
            total_xg.std(),
    }


def build_models():

    model2_home = PoissonRegressor(
        alpha=0.1,
        max_iter=1000,
    )

    model2_away = PoissonRegressor(
        alpha=0.1,
        max_iter=1000,
    )

    challenger_home = (
        HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=42,
        )
    )

    challenger_away = (
        HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=42,
        )
    )

    return {
        "M2": {
            "home":
                model2_home,
            "away":
                model2_away,
        },

        "CA": {
            "home":
                challenger_home,
            "away":
                challenger_away,
        },
    }


def fit_models(
    train,
):

    models = build_models()

    for model in models.values():

        model["home"].fit(
            train[FEATURES],
            train["HomeGoals"],
        )

        model["away"].fit(
            train[FEATURES],
            train["AwayGoals"],
        )

    return models


def predict_matches(
    models,
    test,
    test_season,
):

    rows = []

    test = test.reset_index(
        drop=True
    )

    model_predictions = {}

    for prefix, model in models.items():

        home_xg = model[
            "home"
        ].predict(
            test[FEATURES]
        )

        away_xg = model[
            "away"
        ].predict(
            test[FEATURES]
        )

        model_predictions[
            prefix
        ] = (
            home_xg,
            away_xg,
        )

    for index in range(
        len(test)
    ):

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

        row = {
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

            "HomeGoals":
                test.loc[
                    index,
                    "HomeGoals",
                ],

            "AwayGoals":
                test.loc[
                    index,
                    "AwayGoals",
                ],

            "ActualResult":
                actual,
        }

        for prefix in [
            "M2",
            "CA",
        ]:

            home_xg = float(
                model_predictions[
                    prefix
                ][0][index]
            )

            away_xg = float(
                model_predictions[
                    prefix
                ][1][index]
            )

            home_xg = max(
                home_xg,
                0.05,
            )

            away_xg = max(
                away_xg,
                0.05,
            )

            probabilities = (
                outcome_probabilities(
                    home_xg,
                    away_xg,
                )
            )

            prediction = (
                predicted_result(
                    probabilities[
                        "home_probability"
                    ],
                    probabilities[
                        "draw_probability"
                    ],
                    probabilities[
                        "away_probability"
                    ],
                )
            )

            row[
                f"{prefix}HomeXG"
            ] = home_xg

            row[
                f"{prefix}AwayXG"
            ] = away_xg

            row[
                f"{prefix}HomeProbability"
            ] = probabilities[
                "home_probability"
            ]

            row[
                f"{prefix}DrawProbability"
            ] = probabilities[
                "draw_probability"
            ]

            row[
                f"{prefix}AwayProbability"
            ] = probabilities[
                "away_probability"
            ]

            row[
                f"{prefix}PredictedResult"
            ] = prediction

            row[
                f"{prefix}ModalScore"
            ] = probabilities[
                "modal_score"
            ]

        rows.append(
            row
        )

    return rows


# --------------------------------------------------
# Load
# --------------------------------------------------

print()
print("FOOTBALL COPILOT")
print("CHALLENGER A: NONLINEAR GOAL MODEL")
print("==================================")

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
# Full walk-forward OOT
# --------------------------------------------------

all_rows = []


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

    models = fit_models(
        train
    )

    rows = predict_matches(
        models,
        test,
        test_season,
    )

    all_rows.extend(
        rows
    )


results = pd.DataFrame(
    all_rows
)


# --------------------------------------------------
# Overall summary
# --------------------------------------------------

summary_rows = []


for name, prefix in [
    ("Model 2", "M2"),
    (
        "Challenger A",
        "CA",
    ),
]:

    metrics = calculate_metrics(
        results,
        prefix,
    )

    summary_rows.append(
        {
            "Model":
                name,
            **metrics,
        }
    )


summary = pd.DataFrame(
    summary_rows
)


print()
print("FULL OOT COMPARISON")
print("===================")

print(
    summary.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


# --------------------------------------------------
# By-season
# --------------------------------------------------

season_rows = []


for season in sorted(
    results["Season"].unique()
):

    season_data = results[
        results["Season"]
        ==
        season
    ].copy()

    for name, prefix in [
        ("Model 2", "M2"),
        (
            "Challenger A",
            "CA",
        ),
    ]:

        metrics = calculate_metrics(
            season_data,
            prefix,
        )

        season_rows.append(
            {
                "Season":
                    season,

                "Model":
                    name,

                **metrics,
            }
        )


season_summary = pd.DataFrame(
    season_rows
)


print()
print("COMPARISON BY SEASON")
print("====================")

print(
    season_summary[
        [
            "Season",
            "Model",
            "Matches",
            "AccuracyPct",
            "LogLoss",
            "Brier",
            "PredictedDraws",
            "ActualDraws",
            "ModalOneOnePct",
            "MeanXGGap",
        ]
    ].to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


# --------------------------------------------------
# Historical GW6 simulation
#
# IMPORTANT:
# We do not use the GW6 fixtures to fit either model.
#
# For each season we train using:
# - all earlier seasons
# - current-season matches occurring before the
#   first GW6 fixture
#
# The sixth chronological block of ten matches is
# used as the historical GW6 proxy.
#
# This is appropriate for the 380-match PL seasons
# in this controlled dataset.
# --------------------------------------------------

print()
print("HISTORICAL GW6 SIMULATION")
print("=========================")

gw6_rows = []


for season in [
    "2023/24",
    "2024/25",
    "2025/26",
]:

    season_matches = df[
        df["Season"]
        ==
        season
    ].sort_values(
        "Date"
    ).copy()

    # prediction_features_v2 excludes early matches
    # where teams do not yet have ten-match history.
    #
    # Therefore GW6 cannot safely be reconstructed
    # by taking rows 51-60 from this feature file.
    #
    # We explicitly check whether genuine early
    # season GW6 rows are present.

    season_start = (
        pd.to_datetime(
            season_matches[
                "Date"
            ]
        ).min()
    )

    first_60 = (
        df[
            df["Season"]
            ==
            season
        ]
        .sort_values(
            "Date"
        )
        .head(60)
    )

    if len(first_60) < 60:

        print(
            f"{season}: historical GW6 cannot "
            "be reconstructed safely from "
            "prediction_features_v2.csv."
        )

        continue

    print(
        f"{season}: GW6 simulation deferred "
        "to leakage-safe raw-match reconstruction."
    )


# --------------------------------------------------
# Promotion criteria
# --------------------------------------------------

m2 = summary[
    summary["Model"]
    ==
    "Model 2"
].iloc[0]

ca = summary[
    summary["Model"]
    ==
    "Challenger A"
].iloc[0]


log_loss_pass = (
    ca["LogLoss"]
    <
    m2["LogLoss"]
)

brier_pass = (
    ca["Brier"]
    <
    m2["Brier"]
)

accuracy_change = (
    ca["AccuracyPct"]
    -
    m2["AccuracyPct"]
)

accuracy_pass = (
    accuracy_change
    >= -1.0
)


season_pivots = (
    season_summary.pivot(
        index="Season",
        columns="Model",
        values=[
            "LogLoss",
            "Brier",
        ],
    )
)


season_improvements = 0


for season in (
    season_pivots.index
):

    better_log_loss = (
        season_pivots.loc[
            season,
            (
                "LogLoss",
                "Challenger A",
            ),
        ]
        <
        season_pivots.loc[
            season,
            (
                "LogLoss",
                "Model 2",
            ),
        ]
    )

    better_brier = (
        season_pivots.loc[
            season,
            (
                "Brier",
                "Challenger A",
            ),
        ]
        <
        season_pivots.loc[
            season,
            (
                "Brier",
                "Model 2",
            ),
        ]
    )

    if (
        better_log_loss
        or
        better_brier
    ):

        season_improvements += 1


stability_pass = (
    season_improvements >= 2
)


print()
print("PRE-DEFINED PROMOTION CRITERIA")
print("==============================")

print(
    "Log Loss improves:       ",
    "PASS"
    if log_loss_pass
    else "FAIL",
)

print(
    "Brier improves:          ",
    "PASS"
    if brier_pass
    else "FAIL",
)

print(
    "Accuracy >= -1.0pp:      ",
    "PASS"
    if accuracy_pass
    else "FAIL",
    f"({accuracy_change:+.4f}pp)",
)

print(
    "Improvement >=2 seasons: ",
    "PASS"
    if stability_pass
    else "FAIL",
    f"({season_improvements}/3)",
)


full_oot_pass = (
    log_loss_pass
    and
    brier_pass
    and
    accuracy_pass
    and
    stability_pass
)


print()

if full_oot_pass:

    print(
        "FULL OOT GATE: PASS"
    )

    print(
        "Challenger A qualifies for the "
        "historical GW6 deployment simulation."
    )

else:

    print(
        "FULL OOT GATE: FAIL"
    )

    print(
        "Challenger A does not qualify for "
        "promotion on the full OOT evidence."
    )


# --------------------------------------------------
# Save
# --------------------------------------------------

Path(
    "reports/post_gw5"
).mkdir(
    parents=True,
    exist_ok=True,
)


results.to_csv(
    DETAIL_FILE,
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
    DETAIL_FILE
)

print(
    SUMMARY_FILE
)

print(
    SEASON_FILE
)


print()
print("CHALLENGER A BACKTEST COMPLETE")
print("==============================")