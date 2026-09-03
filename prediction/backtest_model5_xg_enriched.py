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

MODEL2_FILE = (
    "data/processed/"
    "prediction_features_v2.csv"
)

XG_FILE = (
    "data/processed/"
    "premier_league_matches_xg_enriched.csv"
)

OUTPUT_MATCHES = (
    "reports/model5/"
    "model5_xg_enriched_backtest.csv"
)

OUTPUT_SUMMARY = (
    "reports/model5/"
    "model5_xg_enriched_summary.csv"
)

OUTPUT_BY_SEASON = (
    "reports/model5/"
    "model5_xg_enriched_by_season.csv"
)


# ==================================================
# LOCKED MODEL 2 FEATURES
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
# LOCKED MODEL 5 XG FEATURES
# ==================================================

MODEL_5_XG_FEATURES = [

    "HomeXGForAvg5",
    "HomeXGAgainstAvg5",

    "AwayXGForAvg5",
    "AwayXGAgainstAvg5",

    "HomeXGForAvg10",
    "HomeXGAgainstAvg10",

    "AwayXGForAvg10",
    "AwayXGAgainstAvg10",

    "HomeXGDifferenceAvg5",
    "AwayXGDifferenceAvg5",

    "HomeXGForTrend",
    "AwayXGForTrend",

    "HomeXGAgainstTrend",
    "AwayXGAgainstTrend",
]


MODEL_5_FEATURES = (
    MODEL_2_FEATURES
    +
    MODEL_5_XG_FEATURES
)


# ==================================================
# LOCKED TEST SEASONS
# ==================================================

TEST_SEASONS = [
    "2023/24",
    "2024/25",
]


# ==================================================
# LOCKED SUCCESS THRESHOLDS
# ==================================================

TARGET_ACCURACY = 54.5

TARGET_LOG_LOSS = 0.975

TARGET_BRIER = 0.580


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
print("MODEL 5 - XG ENRICHED POISSON")
print("==============================")
print()

model2 = pd.read_csv(
    MODEL2_FILE
)

xg = pd.read_csv(
    XG_FILE
)

model2["Date"] = pd.to_datetime(
    model2["Date"],
    errors="coerce",
)

xg["Date"] = pd.to_datetime(
    xg["Date"],
    errors="coerce",
)


# ==================================================
# JOIN MODEL 2 + XG
# ==================================================

JOIN_KEYS = [
    "Season",
    "Date",
    "HomeTeam",
    "AwayTeam",
]


xg_columns = (
    JOIN_KEYS
    +
    MODEL_5_XG_FEATURES
)


df = model2.merge(
    xg[
        xg_columns
    ],
    on=JOIN_KEYS,
    how="inner",
    validate="one_to_one",
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


print(
    "Model 2 rows:",
    len(model2),
)

print(
    "Joined Model 2 + xG rows:",
    len(df),
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

NUMBER_TO_RESULT = {
    0: "H",
    1: "D",
    2: "A",
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
    in MODEL_5_FEATURES
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing Model 5 features: "
        +
        ", ".join(
            missing_features
        )
    )


df[
    MODEL_5_FEATURES
] = (
    df[
        MODEL_5_FEATURES
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
# WALK-FORWARD
# ==================================================

all_rows = []

season_summaries = []


for test_season in TEST_SEASONS:

    all_seasons = sorted(
        df["Season"]
        .dropna()
        .unique()
    )

    season_position = (
        all_seasons.index(
            test_season
        )
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
    # MODEL 2 BASELINE
    # ==============================================

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


    # ==============================================
    # MODEL 5 XG-ENRICHED
    # ==============================================

    model5_home = PoissonRegressor(
        alpha=0.1,
        max_iter=1000,
    )

    model5_away = PoissonRegressor(
        alpha=0.1,
        max_iter=1000,
    )

    model5_home.fit(
        train[
            MODEL_5_FEATURES
        ],
        train[
            "HomeGoals"
        ],
    )

    model5_away.fit(
        train[
            MODEL_5_FEATURES
        ],
        train[
            "AwayGoals"
        ],
    )

    model5_home_xg = (
        model5_home.predict(
            test[
                MODEL_5_FEATURES
            ]
        )
    )

    model5_away_xg = (
        model5_away.predict(
            test[
                MODEL_5_FEATURES
            ]
        )
    )


    # ==============================================
    # OUTCOME PROBABILITIES
    # ==============================================

    model2_probabilities = []
    model5_probabilities = []

    model2_predictions = []
    model5_predictions = []


    for i in range(
        len(test)
    ):

        m2_probs = (
            independent_poisson_probabilities(
                home_xg=float(
                    model2_home_xg[i]
                ),
                away_xg=float(
                    model2_away_xg[i]
                ),
                max_goals=8,
            )
        )

        m5_probs = (
            independent_poisson_probabilities(
                home_xg=float(
                    model5_home_xg[i]
                ),
                away_xg=float(
                    model5_away_xg[i]
                ),
                max_goals=8,
            )
        )


        m2 = [
            float(
                m2_probs[
                    "home_probability"
                ]
            ),
            float(
                m2_probs[
                    "draw_probability"
                ]
            ),
            float(
                m2_probs[
                    "away_probability"
                ]
            ),
        ]

        m5 = [
            float(
                m5_probs[
                    "home_probability"
                ]
            ),
            float(
                m5_probs[
                    "draw_probability"
                ]
            ),
            float(
                m5_probs[
                    "away_probability"
                ]
            ),
        ]


        model2_probabilities.append(
            m2
        )

        model5_probabilities.append(
            m5
        )


        model2_predictions.append(
            int(
                np.argmax(
                    m2
                )
            )
        )

        model5_predictions.append(
            int(
                np.argmax(
                    m5
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

    model2_probabilities = np.asarray(
        model2_probabilities
    )

    model5_probabilities = np.asarray(
        model5_probabilities
    )

    model2_predictions = np.asarray(
        model2_predictions
    )

    model5_predictions = np.asarray(
        model5_predictions
    )


    model2_accuracy = (
        accuracy_score(
            actual_numeric,
            model2_predictions,
        )
        *
        100.0
    )

    model5_accuracy = (
        accuracy_score(
            actual_numeric,
            model5_predictions,
        )
        *
        100.0
    )


    model2_logloss = log_loss(
        actual_numeric,
        model2_probabilities,
        labels=[
            0,
            1,
            2,
        ],
    )

    model5_logloss = log_loss(
        actual_numeric,
        model5_probabilities,
        labels=[
            0,
            1,
            2,
        ],
    )


    model2_brier = (
        multiclass_brier_score(
            actual_numeric,
            model2_probabilities,
        )
    )

    model5_brier = (
        multiclass_brier_score(
            actual_numeric,
            model5_probabilities,
        )
    )


    actual_draws = int(
        np.sum(
            actual_numeric == 1
        )
    )

    model2_draws = int(
        np.sum(
            model2_predictions == 1
        )
    )

    model5_draws = int(
        np.sum(
            model5_predictions == 1
        )
    )


    print(
        f"Model 2 Accuracy: "
        f"{model2_accuracy:.4f}%"
    )

    print(
        f"Model 5 Accuracy: "
        f"{model5_accuracy:.4f}%"
    )

    print(
        f"Model 2 Log Loss: "
        f"{model2_logloss:.4f}"
    )

    print(
        f"Model 5 Log Loss: "
        f"{model5_logloss:.4f}"
    )

    print(
        f"Model 2 Brier: "
        f"{model2_brier:.4f}"
    )

    print(
        f"Model 5 Brier: "
        f"{model5_brier:.4f}"
    )

    print(
        f"Actual draws: "
        f"{actual_draws}"
    )

    print(
        f"Model 2 predicted draws: "
        f"{model2_draws}"
    )

    print(
        f"Model 5 predicted draws: "
        f"{model5_draws}"
    )


    # ==============================================
    # MATCH LEVEL OUTPUT
    # ==============================================

    for i in range(
        len(test)
    ):

        all_rows.append(
            {
                "Season":
                    test_season,

                "Date":
                    test.loc[
                        i,
                        "Date",
                    ],

                "HomeTeam":
                    test.loc[
                        i,
                        "HomeTeam",
                    ],

                "AwayTeam":
                    test.loc[
                        i,
                        "AwayTeam",
                    ],

                "HomeGoals":
                    int(
                        test.loc[
                            i,
                            "HomeGoals",
                        ]
                    ),

                "AwayGoals":
                    int(
                        test.loc[
                            i,
                            "AwayGoals",
                        ]
                    ),

                "ActualResult":
                    test.loc[
                        i,
                        "ActualResult",
                    ],

                "Model2HomeProbability":
                    model2_probabilities[
                        i,
                        0,
                    ],

                "Model2DrawProbability":
                    model2_probabilities[
                        i,
                        1,
                    ],

                "Model2AwayProbability":
                    model2_probabilities[
                        i,
                        2,
                    ],

                "Model2Prediction":
                    NUMBER_TO_RESULT[
                        int(
                            model2_predictions[
                                i
                            ]
                        )
                    ],

                "Model5HomeProbability":
                    model5_probabilities[
                        i,
                        0,
                    ],

                "Model5DrawProbability":
                    model5_probabilities[
                        i,
                        1,
                    ],

                "Model5AwayProbability":
                    model5_probabilities[
                        i,
                        2,
                    ],

                "Model5Prediction":
                    NUMBER_TO_RESULT[
                        int(
                            model5_predictions[
                                i
                            ]
                        )
                    ],

                "Model2HomeExpectedGoals":
                    float(
                        model2_home_xg[
                            i
                        ]
                    ),

                "Model2AwayExpectedGoals":
                    float(
                        model2_away_xg[
                            i
                        ]
                    ),

                "Model5HomeExpectedGoals":
                    float(
                        model5_home_xg[
                            i
                        ]
                    ),

                "Model5AwayExpectedGoals":
                    float(
                        model5_away_xg[
                            i
                        ]
                    ),
            }
        )


    season_summaries.append(
        {
            "Season":
                test_season,

            "Matches":
                len(test),

            "Model2Accuracy":
                model2_accuracy,

            "Model5Accuracy":
                model5_accuracy,

            "AccuracyUpliftPP":
                (
                    model5_accuracy
                    -
                    model2_accuracy
                ),

            "Model2LogLoss":
                model2_logloss,

            "Model5LogLoss":
                model5_logloss,

            "LogLossImprovement":
                (
                    model2_logloss
                    -
                    model5_logloss
                ),

            "Model2Brier":
                model2_brier,

            "Model5Brier":
                model5_brier,

            "BrierImprovement":
                (
                    model2_brier
                    -
                    model5_brier
                ),

            "ActualDraws":
                actual_draws,

            "Model2PredictedDraws":
                model2_draws,

            "Model5PredictedDraws":
                model5_draws,
        }
    )


# ==================================================
# COMBINE RESULTS
# ==================================================

results = pd.DataFrame(
    all_rows
)

by_season = pd.DataFrame(
    season_summaries
)


# ==================================================
# OVERALL METRICS
# ==================================================

actual_numeric_all = (
    results[
        "ActualResult"
    ]
    .map(
        RESULT_TO_NUMBER
    )
    .astype(int)
    .to_numpy()
)


model2_probabilities_all = (
    results[
        [
            "Model2HomeProbability",
            "Model2DrawProbability",
            "Model2AwayProbability",
        ]
    ]
    .to_numpy()
)


model5_probabilities_all = (
    results[
        [
            "Model5HomeProbability",
            "Model5DrawProbability",
            "Model5AwayProbability",
        ]
    ]
    .to_numpy()
)


model2_predictions_all = (
    results[
        "Model2Prediction"
    ]
    .map(
        RESULT_TO_NUMBER
    )
    .astype(int)
    .to_numpy()
)


model5_predictions_all = (
    results[
        "Model5Prediction"
    ]
    .map(
        RESULT_TO_NUMBER
    )
    .astype(int)
    .to_numpy()
)


model2_accuracy_all = (
    accuracy_score(
        actual_numeric_all,
        model2_predictions_all,
    )
    *
    100.0
)


model5_accuracy_all = (
    accuracy_score(
        actual_numeric_all,
        model5_predictions_all,
    )
    *
    100.0
)


model2_logloss_all = log_loss(
    actual_numeric_all,
    model2_probabilities_all,
    labels=[
        0,
        1,
        2,
    ],
)


model5_logloss_all = log_loss(
    actual_numeric_all,
    model5_probabilities_all,
    labels=[
        0,
        1,
        2,
    ],
)


model2_brier_all = (
    multiclass_brier_score(
        actual_numeric_all,
        model2_probabilities_all,
    )
)


model5_brier_all = (
    multiclass_brier_score(
        actual_numeric_all,
        model5_probabilities_all,
    )
)


actual_draws_all = int(
    np.sum(
        actual_numeric_all == 1
    )
)


model2_draws_all = int(
    np.sum(
        model2_predictions_all == 1
    )
)


model5_draws_all = int(
    np.sum(
        model5_predictions_all == 1
    )
)


# ==================================================
# UPLIFT
# ==================================================

accuracy_uplift = (
    model5_accuracy_all
    -
    model2_accuracy_all
)

logloss_improvement = (
    model2_logloss_all
    -
    model5_logloss_all
)

brier_improvement = (
    model2_brier_all
    -
    model5_brier_all
)


# ==================================================
# LOCKED SUCCESS TEST
# ==================================================

accuracy_target_met = (
    model5_accuracy_all
    >=
    TARGET_ACCURACY
)

logloss_target_met = (
    model5_logloss_all
    <=
    TARGET_LOG_LOSS
)

brier_target_met = (
    model5_brier_all
    <=
    TARGET_BRIER
)

probability_quality_improved = (
    model5_logloss_all
    <
    model2_logloss_all
    and
    model5_brier_all
    <
    model2_brier_all
)


if (
    accuracy_target_met
    and
    logloss_target_met
    and
    brier_target_met
    and
    probability_quality_improved
):

    verdict = "PROMOTE"

elif (
    probability_quality_improved
    and
    (
        accuracy_uplift >= 0.5
        or
        logloss_improvement >= 0.005
        or
        brier_improvement >= 0.005
    )
):

    verdict = "PROMISING"

else:

    verdict = "REJECT"


# ==================================================
# SUMMARY
# ==================================================

summary = pd.DataFrame(
    [
        {
            "Matches":
                len(results),

            "Model2Accuracy":
                model2_accuracy_all,

            "Model5Accuracy":
                model5_accuracy_all,

            "AccuracyUpliftPP":
                accuracy_uplift,

            "Model2LogLoss":
                model2_logloss_all,

            "Model5LogLoss":
                model5_logloss_all,

            "LogLossImprovement":
                logloss_improvement,

            "Model2Brier":
                model2_brier_all,

            "Model5Brier":
                model5_brier_all,

            "BrierImprovement":
                brier_improvement,

            "ActualDraws":
                actual_draws_all,

            "Model2PredictedDraws":
                model2_draws_all,

            "Model5PredictedDraws":
                model5_draws_all,

            "AccuracyTarget":
                TARGET_ACCURACY,

            "AccuracyTargetMet":
                accuracy_target_met,

            "LogLossTarget":
                TARGET_LOG_LOSS,

            "LogLossTargetMet":
                logloss_target_met,

            "BrierTarget":
                TARGET_BRIER,

            "BrierTargetMet":
                brier_target_met,

            "ProbabilityQualityImproved":
                probability_quality_improved,

            "Verdict":
                verdict,
        }
    ]
)


# ==================================================
# PRINT RESULTS
# ==================================================

print()
print("OVERALL RESULT")
print("==============")

print(
    summary.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)


print()
print("BY SEASON")
print("=========")

print(
    by_season.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)


print()
print("LOCKED SUCCESS THRESHOLDS")
print("=========================")

print(
    f"Accuracy >= "
    f"{TARGET_ACCURACY:.2f}%: "
    f"{accuracy_target_met}"
)

print(
    f"Log Loss <= "
    f"{TARGET_LOG_LOSS:.3f}: "
    f"{logloss_target_met}"
)

print(
    f"Brier <= "
    f"{TARGET_BRIER:.3f}: "
    f"{brier_target_met}"
)

print(
    "Log Loss and Brier both "
    f"improved vs Model 2: "
    f"{probability_quality_improved}"
)


print()
print("FINAL MODEL 5 VERDICT")
print("=====================")

print(
    verdict
)


# ==================================================
# SAVE RESULTS
# ==================================================

Path(
    "reports/model5"
).mkdir(
    parents=True,
    exist_ok=True,
)


results.to_csv(
    OUTPUT_MATCHES,
    index=False,
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)

by_season.to_csv(
    OUTPUT_BY_SEASON,
    index=False,
)


print()
print("FILES SAVED")
print("===========")

print(
    OUTPUT_MATCHES
)

print(
    OUTPUT_SUMMARY
)

print(
    OUTPUT_BY_SEASON
)


print()
print("MODEL 5 BACKTEST COMPLETE")
print("=========================")