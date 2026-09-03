from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
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

OUTPUT_MATCHES = (
    "reports/model4/"
    "model4_direct_1x2_backtest.csv"
)

OUTPUT_SUMMARY = (
    "reports/model4/"
    "model4_direct_1x2_summary.csv"
)

OUTPUT_BY_SEASON = (
    "reports/model4/"
    "model4_direct_1x2_by_season.csv"
)

OUTPUT_FEATURE_IMPORTANCE = (
    "reports/model4/"
    "model4_feature_importance.csv"
)


# ==================================================
# LOCKED SUCCESS THRESHOLDS
# ==================================================

TARGET_ACCURACY = 54.0

TARGET_LOG_LOSS = 0.980

TARGET_BRIER = 0.585


# ==================================================
# ELO SETTINGS
#
# Preserved from existing V3 feature pipeline
# ==================================================

STARTING_ELO = 1500.0

K_FACTOR = 20.0

HOME_ELO_ADVANTAGE = 65.0


# ==================================================
# MODEL 2 FEATURES
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
# MODEL 4 ADDITIONAL FEATURES
# ==================================================

MODEL_4_ENGINEERED_FEATURES = [

    "HomeElo",
    "AwayElo",
    "EloDifference",
    "EloExpectedHome",

    "HomeFormAcceleration",
    "AwayFormAcceleration",
    "FormAccelerationDifference",

    "HomeAttackAcceleration",
    "AwayAttackAcceleration",
    "AttackAccelerationDifference",

    "HomeDefenceChange",
    "AwayDefenceChange",
    "DefenceChangeDifference",

    "RecentGoalDifferenceHome",
    "RecentGoalDifferenceAway",
    "RecentGoalDifferenceGap",

    "TenMatchGoalDifferenceHome",
    "TenMatchGoalDifferenceAway",
    "TenMatchGoalDifferenceGap",

    "VenuePPGDifference",
    "VenueGoalsForDifference",

    "HomeStrengthComposite",
    "AwayStrengthComposite",
    "StrengthCompositeDifference",

    "RecentVsSeasonHome",
    "RecentVsSeasonAway",
    "RecentVsSeasonDifference",
]


MODEL_4_FEATURES = (
    MODEL_2_FEATURES
    +
    MODEL_4_ENGINEERED_FEATURES
)


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


def expected_score(
    rating_a: float,
    rating_b: float,
) -> float:

    return 1.0 / (
        1.0
        +
        10.0
        **
        (
            (
                rating_b
                -
                rating_a
            )
            /
            400.0
        )
    )


def update_elo(
    home_elo: float,
    away_elo: float,
    home_goals: int,
    away_goals: int,
):

    adjusted_home_elo = (
        home_elo
        +
        HOME_ELO_ADVANTAGE
    )

    expected_home = expected_score(
        adjusted_home_elo,
        away_elo,
    )

    expected_away = (
        1.0
        -
        expected_home
    )

    if home_goals > away_goals:

        actual_home = 1.0
        actual_away = 0.0

    elif home_goals == away_goals:

        actual_home = 0.5
        actual_away = 0.5

    else:

        actual_home = 0.0
        actual_away = 1.0

    new_home_elo = (
        home_elo
        +
        K_FACTOR
        *
        (
            actual_home
            -
            expected_home
        )
    )

    new_away_elo = (
        away_elo
        +
        K_FACTOR
        *
        (
            actual_away
            -
            expected_away
        )
    )

    return (
        new_home_elo,
        new_away_elo,
    )


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


def safe_pct(
    numerator,
    denominator,
):

    if denominator == 0:
        return np.nan

    return (
        numerator
        /
        denominator
        *
        100.0
    )


# ==================================================
# LOAD DATA
# ==================================================

print()
print("FOOTBALL COPILOT")
print("MODEL 4 - DIRECT 1X2 GRADIENT BOOSTING")
print("=======================================")

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
        int(
            row["HomeGoals"]
        ),
        int(
            row["AwayGoals"]
        ),
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
# BUILD LEAKAGE-SAFE PRE-MATCH ELO
# ==================================================

print()
print("Building pre-match Elo ratings...")

elo_ratings = {}

elo_rows = []

for _, row in df.iterrows():

    home_team = row[
        "HomeTeam"
    ]

    away_team = row[
        "AwayTeam"
    ]

    if home_team not in elo_ratings:

        elo_ratings[
            home_team
        ] = STARTING_ELO

    if away_team not in elo_ratings:

        elo_ratings[
            away_team
        ] = STARTING_ELO

    home_elo = float(
        elo_ratings[
            home_team
        ]
    )

    away_elo = float(
        elo_ratings[
            away_team
        ]
    )

    adjusted_home_elo = (
        home_elo
        +
        HOME_ELO_ADVANTAGE
    )

    expected_home = expected_score(
        adjusted_home_elo,
        away_elo,
    )

    elo_rows.append(
        {
            "HomeElo":
                home_elo,

            "AwayElo":
                away_elo,

            "EloDifference":
                (
                    adjusted_home_elo
                    -
                    away_elo
                ),

            "EloExpectedHome":
                expected_home,
        }
    )

    (
        new_home_elo,
        new_away_elo,
    ) = update_elo(
        home_elo,
        away_elo,
        int(
            row[
                "HomeGoals"
            ]
        ),
        int(
            row[
                "AwayGoals"
            ]
        ),
    )

    elo_ratings[
        home_team
    ] = new_home_elo

    elo_ratings[
        away_team
    ] = new_away_elo


elo_df = pd.DataFrame(
    elo_rows
)

df = pd.concat(
    [
        df,
        elo_df,
    ],
    axis=1,
)


# ==================================================
# ENGINEER RECENCY / RELATIVE-STRENGTH FEATURES
#
# No post-match information is used.
# These are transformations of existing
# pre-match V2 variables.
# ==================================================

df[
    "HomeFormAcceleration"
] = (
    df["HomeRecentPPG"]
    -
    df["Home10PPG"]
)

df[
    "AwayFormAcceleration"
] = (
    df["AwayRecentPPG"]
    -
    df["Away10PPG"]
)

df[
    "FormAccelerationDifference"
] = (
    df["HomeFormAcceleration"]
    -
    df["AwayFormAcceleration"]
)


df[
    "HomeAttackAcceleration"
] = (
    df["HomeRecentGoalsFor"]
    -
    df["Home10GoalsFor"]
)

df[
    "AwayAttackAcceleration"
] = (
    df["AwayRecentGoalsFor"]
    -
    df["Away10GoalsFor"]
)

df[
    "AttackAccelerationDifference"
] = (
    df["HomeAttackAcceleration"]
    -
    df["AwayAttackAcceleration"]
)


df[
    "HomeDefenceChange"
] = (
    df["HomeRecentGoalsAgainst"]
    -
    df["Home10GoalsAgainst"]
)

df[
    "AwayDefenceChange"
] = (
    df["AwayRecentGoalsAgainst"]
    -
    df["Away10GoalsAgainst"]
)

df[
    "DefenceChangeDifference"
] = (
    df["HomeDefenceChange"]
    -
    df["AwayDefenceChange"]
)


df[
    "RecentGoalDifferenceHome"
] = (
    df["HomeRecentGoalsFor"]
    -
    df["HomeRecentGoalsAgainst"]
)

df[
    "RecentGoalDifferenceAway"
] = (
    df["AwayRecentGoalsFor"]
    -
    df["AwayRecentGoalsAgainst"]
)

df[
    "RecentGoalDifferenceGap"
] = (
    df["RecentGoalDifferenceHome"]
    -
    df["RecentGoalDifferenceAway"]
)


df[
    "TenMatchGoalDifferenceHome"
] = (
    df["Home10GoalsFor"]
    -
    df["Home10GoalsAgainst"]
)

df[
    "TenMatchGoalDifferenceAway"
] = (
    df["Away10GoalsFor"]
    -
    df["Away10GoalsAgainst"]
)

df[
    "TenMatchGoalDifferenceGap"
] = (
    df["TenMatchGoalDifferenceHome"]
    -
    df["TenMatchGoalDifferenceAway"]
)


df[
    "VenuePPGDifference"
] = (
    df["HomeVenuePPG"]
    -
    df["AwayVenuePPG"]
)

df[
    "VenueGoalsForDifference"
] = (
    df["HomeVenueGoalsFor"]
    -
    df["AwayVenueGoalsFor"]
)


df[
    "HomeStrengthComposite"
] = (
    0.50
    *
    df["HomeRecentPPG"]
    +
    0.30
    *
    df["Home10PPG"]
    +
    0.20
    *
    df["HomeSeasonPPG"]
)

df[
    "AwayStrengthComposite"
] = (
    0.50
    *
    df["AwayRecentPPG"]
    +
    0.30
    *
    df["Away10PPG"]
    +
    0.20
    *
    df["AwaySeasonPPG"]
)

df[
    "StrengthCompositeDifference"
] = (
    df["HomeStrengthComposite"]
    -
    df["AwayStrengthComposite"]
)


df[
    "RecentVsSeasonHome"
] = (
    df["HomeRecentPPG"]
    -
    df["HomeSeasonPPG"]
)

df[
    "RecentVsSeasonAway"
] = (
    df["AwayRecentPPG"]
    -
    df["AwaySeasonPPG"]
)

df[
    "RecentVsSeasonDifference"
] = (
    df["RecentVsSeasonHome"]
    -
    df["RecentVsSeasonAway"]
)


# ==================================================
# VALIDATE FEATURES
# ==================================================

missing_features = [
    feature
    for feature
    in MODEL_4_FEATURES
    if feature
    not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing Model 4 features: "
        +
        ", ".join(
            missing_features
        )
    )


df[
    MODEL_4_FEATURES
] = (
    df[
        MODEL_4_FEATURES
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
# WALK-FORWARD SEASONS
# ==================================================

seasons = sorted(
    df["Season"]
    .dropna()
    .unique()
)

print()
print(
    "Seasons:",
    seasons,
)


all_rows = []

season_summaries = []

feature_importance_rows = []


# ==================================================
# WALK-FORWARD TEST
# ==================================================

for test_index in range(
    2,
    len(seasons),
):

    test_season = seasons[
        test_index
    ]

    training_seasons = seasons[
        :test_index
    ]

    train = df[
        df["Season"]
        .isin(
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


    # ==============================================
    # MODEL 2 REFERENCE
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

    model2_home_xg = (
        home_poisson
        .predict(
            test[
                MODEL_2_FEATURES
            ]
        )
    )

    model2_away_xg = (
        away_poisson
        .predict(
            test[
                MODEL_2_FEATURES
            ]
        )
    )

    model2_probabilities = []

    model2_predictions = []

    for i in range(
        len(test)
    ):

        probs = (
            independent_poisson_probabilities(
                home_xg=float(
                    model2_home_xg[
                        i
                    ]
                ),
                away_xg=float(
                    model2_away_xg[
                        i
                    ]
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

        model2_probabilities.append(
            [
                h,
                d,
                a,
            ]
        )

        model2_predictions.append(
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
    # MODEL 4
    #
    # Direct H/D/A classifier
    # ==============================================

    model4 = (
        HistGradientBoostingClassifier(
            loss="log_loss",

            learning_rate=0.05,

            max_iter=300,

            max_leaf_nodes=15,

            max_depth=4,

            min_samples_leaf=20,

            l2_regularization=1.0,

            early_stopping=True,

            validation_fraction=0.15,

            n_iter_no_change=30,

            random_state=42,
        )
    )

    model4.fit(
        train[
            MODEL_4_FEATURES
        ],
        train[
            "Target"
        ],
    )

    model4_probabilities = (
        model4.predict_proba(
            test[
                MODEL_4_FEATURES
            ]
        )
    )

    model4_predictions = (
        np.argmax(
            model4_probabilities,
            axis=1,
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

    actual_labels = (
        test[
            "ActualResult"
        ]
        .tolist()
    )

    model2_accuracy = (
        accuracy_score(
            actual_numeric,
            model2_predictions,
        )
        *
        100.0
    )

    model4_accuracy = (
        accuracy_score(
            actual_numeric,
            model4_predictions,
        )
        *
        100.0
    )

    model2_logloss = (
        log_loss(
            actual_numeric,
            model2_probabilities,
            labels=[
                0,
                1,
                2,
            ],
        )
    )

    model4_logloss = (
        log_loss(
            actual_numeric,
            model4_probabilities,
            labels=[
                0,
                1,
                2,
            ],
        )
    )

    model2_brier = (
        multiclass_brier_score(
            actual_numeric,
            model2_probabilities,
        )
    )

    model4_brier = (
        multiclass_brier_score(
            actual_numeric,
            model4_probabilities,
        )
    )

    actual_draws = int(
        np.sum(
            actual_numeric
            ==
            1
        )
    )

    model2_draws = int(
        np.sum(
            np.asarray(
                model2_predictions
            )
            ==
            1
        )
    )

    model4_draws = int(
        np.sum(
            model4_predictions
            ==
            1
        )
    )

    model4_correct_draws = int(
        np.sum(
            (
                model4_predictions
                ==
                1
            )
            &
            (
                actual_numeric
                ==
                1
            )
        )
    )

    model4_draw_recall = (
        safe_pct(
            model4_correct_draws,
            actual_draws,
        )
    )

    model4_draw_precision = (
        safe_pct(
            model4_correct_draws,
            model4_draws,
        )
    )

    print(
        f"Model 2 accuracy: "
        f"{model2_accuracy:.2f}%"
    )

    print(
        f"Model 4 accuracy: "
        f"{model4_accuracy:.2f}%"
    )

    print(
        f"Model 2 log loss: "
        f"{model2_logloss:.4f}"
    )

    print(
        f"Model 4 log loss: "
        f"{model4_logloss:.4f}"
    )

    print(
        f"Model 2 Brier: "
        f"{model2_brier:.4f}"
    )

    print(
        f"Model 4 Brier: "
        f"{model4_brier:.4f}"
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
        f"Model 4 predicted draws: "
        f"{model4_draws}"
    )


    # ==============================================
    # FEATURE IMPORTANCE
    #
    # Permutation importance on unseen test season
    # using negative log loss.
    # ==============================================

    importance = (
        permutation_importance(
            model4,
            test[
                MODEL_4_FEATURES
            ],
            actual_numeric,

            scoring="neg_log_loss",

            n_repeats=5,

            random_state=42,

            n_jobs=-1,
        )
    )

    for feature, mean_importance in zip(
        MODEL_4_FEATURES,
        importance.importances_mean,
    ):

        feature_importance_rows.append(
            {
                "Season":
                    test_season,

                "Feature":
                    feature,

                "Importance":
                    float(
                        mean_importance
                    ),
            }
        )


    # ==============================================
    # MATCH LEVEL OUTPUT
    # ==============================================

    for i in range(
        len(test)
    ):

        m2 = (
            model2_probabilities[
                i
            ]
        )

        m4 = (
            model4_probabilities[
                i
            ]
        )

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
                    actual_labels[
                        i
                    ],

                "Model2HomeProbability":
                    m2[
                        0
                    ],

                "Model2DrawProbability":
                    m2[
                        1
                    ],

                "Model2AwayProbability":
                    m2[
                        2
                    ],

                "Model2Prediction":
                    NUMBER_TO_RESULT[
                        model2_predictions[
                            i
                        ]
                    ],

                "Model4HomeProbability":
                    m4[
                        0
                    ],

                "Model4DrawProbability":
                    m4[
                        1
                    ],

                "Model4AwayProbability":
                    m4[
                        2
                    ],

                "Model4Prediction":
                    NUMBER_TO_RESULT[
                        int(
                            model4_predictions[
                                i
                            ]
                        )
                    ],

                "HomeElo":
                    test.loc[
                        i,
                        "HomeElo",
                    ],

                "AwayElo":
                    test.loc[
                        i,
                        "AwayElo",
                    ],

                "EloDifference":
                    test.loc[
                        i,
                        "EloDifference",
                    ],

                "EloExpectedHome":
                    test.loc[
                        i,
                        "EloExpectedHome",
                    ],
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

            "Model4Accuracy":
                model4_accuracy,

            "Model2LogLoss":
                model2_logloss,

            "Model4LogLoss":
                model4_logloss,

            "Model2Brier":
                model2_brier,

            "Model4Brier":
                model4_brier,

            "ActualDraws":
                actual_draws,

            "Model2PredictedDraws":
                model2_draws,

            "Model4PredictedDraws":
                model4_draws,

            "Model4CorrectDraws":
                model4_correct_draws,

            "Model4DrawRecall":
                model4_draw_recall,

            "Model4DrawPrecision":
                model4_draw_precision,

            "Model4Iterations":
                model4.n_iter_,
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

feature_importance = pd.DataFrame(
    feature_importance_rows
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

model4_probabilities_all = (
    results[
        [
            "Model4HomeProbability",
            "Model4DrawProbability",
            "Model4AwayProbability",
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

model4_predictions_all = (
    results[
        "Model4Prediction"
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

model4_accuracy_all = (
    accuracy_score(
        actual_numeric_all,
        model4_predictions_all,
    )
    *
    100.0
)

model2_logloss_all = (
    log_loss(
        actual_numeric_all,
        model2_probabilities_all,
        labels=[
            0,
            1,
            2,
        ],
    )
)

model4_logloss_all = (
    log_loss(
        actual_numeric_all,
        model4_probabilities_all,
        labels=[
            0,
            1,
            2,
        ],
    )
)

model2_brier_all = (
    multiclass_brier_score(
        actual_numeric_all,
        model2_probabilities_all,
    )
)

model4_brier_all = (
    multiclass_brier_score(
        actual_numeric_all,
        model4_probabilities_all,
    )
)

actual_draws_all = int(
    np.sum(
        actual_numeric_all
        ==
        1
    )
)

model2_draws_all = int(
    np.sum(
        model2_predictions_all
        ==
        1
    )
)

model4_draws_all = int(
    np.sum(
        model4_predictions_all
        ==
        1
    )
)

model4_correct_draws_all = int(
    np.sum(
        (
            model4_predictions_all
            ==
            1
        )
        &
        (
            actual_numeric_all
            ==
            1
        )
    )
)

model4_draw_recall_all = (
    safe_pct(
        model4_correct_draws_all,
        actual_draws_all,
    )
)

model4_draw_precision_all = (
    safe_pct(
        model4_correct_draws_all,
        model4_draws_all,
    )
)


# ==================================================
# SUCCESS THRESHOLDS
# ==================================================

accuracy_target_met = (
    model4_accuracy_all
    >=
    TARGET_ACCURACY
)

logloss_target_met = (
    model4_logloss_all
    <=
    TARGET_LOG_LOSS
)

brier_target_met = (
    model4_brier_all
    <=
    TARGET_BRIER
)

targets_met = sum(
    [
        accuracy_target_met,
        logloss_target_met,
        brier_target_met,
    ]
)


# ==================================================
# PRACTICAL UPLIFT
# ==================================================

accuracy_uplift = (
    model4_accuracy_all
    -
    model2_accuracy_all
)

logloss_improvement = (
    model2_logloss_all
    -
    model4_logloss_all
)

brier_improvement = (
    model2_brier_all
    -
    model4_brier_all
)


# ==================================================
# VERDICT
# ==================================================

if (
    targets_met == 3
    and
    model4_logloss_all
    <
    model2_logloss_all
    and
    model4_brier_all
    <
    model2_brier_all
):

    verdict = "PROMOTE"

elif (
    targets_met >= 2
    and
    model4_logloss_all
    <
    model2_logloss_all
    and
    model4_brier_all
    <
    model2_brier_all
):

    verdict = "PROMISING"

elif (
    accuracy_uplift >= 1.0
    and
    logloss_improvement > 0.005
    and
    brier_improvement > 0.005
):

    verdict = "PROMISING"

else:

    verdict = "REJECT"


summary = pd.DataFrame(
    [
        {
            "Matches":
                len(results),

            "Model2Accuracy":
                model2_accuracy_all,

            "Model4Accuracy":
                model4_accuracy_all,

            "AccuracyUpliftPP":
                accuracy_uplift,

            "Model2LogLoss":
                model2_logloss_all,

            "Model4LogLoss":
                model4_logloss_all,

            "LogLossImprovement":
                logloss_improvement,

            "Model2Brier":
                model2_brier_all,

            "Model4Brier":
                model4_brier_all,

            "BrierImprovement":
                brier_improvement,

            "ActualDraws":
                actual_draws_all,

            "ActualDrawPct":
                safe_pct(
                    actual_draws_all,
                    len(results),
                ),

            "Model2PredictedDraws":
                model2_draws_all,

            "Model4PredictedDraws":
                model4_draws_all,

            "Model4CorrectDraws":
                model4_correct_draws_all,

            "Model4DrawRecall":
                model4_draw_recall_all,

            "Model4DrawPrecision":
                model4_draw_precision_all,

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

            "TargetsMet":
                targets_met,

            "Verdict":
                verdict,
        }
    ]
)


# ==================================================
# AGGREGATE FEATURE IMPORTANCE
# ==================================================

feature_importance_summary = (
    feature_importance
    .groupby(
        "Feature",
        as_index=False,
    )
    .agg(
        MeanImportance=(
            "Importance",
            "mean",
        ),
        MaxImportance=(
            "Importance",
            "max",
        ),
    )
    .sort_values(
        "MeanImportance",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
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
print("TOP 15 MODEL 4 FEATURES")
print("=======================")

print(
    feature_importance_summary
    .head(
        15
    )
    .to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.6f}"
        ),
    )
)


print()
print("PRE-AGREED SUCCESS THRESHOLDS")
print("=============================")

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


print()
print("FINAL MODEL 4 VERDICT")
print("=====================")

print(
    verdict
)

if verdict == "PROMOTE":

    print(
        "Model 4 produced a material improvement "
        "against Model 2 and met the agreed "
        "promotion hurdle."
    )

elif verdict == "PROMISING":

    print(
        "Model 4 produced a meaningful enough "
        "improvement to justify further validation, "
        "but has not cleared the full promotion hurdle."
    )

else:

    print(
        "Model 4 did not produce enough uplift. "
        "Do not continue making small algorithmic "
        "changes with the same dataset. "
        "The next step should be richer data."
    )


# ==================================================
# SAVE RESULTS
# ==================================================

Path(
    "reports/model4"
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

feature_importance_summary.to_csv(
    OUTPUT_FEATURE_IMPORTANCE,
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

print(
    OUTPUT_FEATURE_IMPORTANCE
)


print()
print("MODEL 4 BACKTEST COMPLETE")
print("=========================")