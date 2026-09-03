from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss

from prediction.dixon_coles import independent_poisson_probabilities


# ============================================================
# FILES
# ============================================================

MODEL2_FILE = Path(
    "data/processed/prediction_features_v2.csv"
)

HISTORICAL_XG_FILE = Path(
    "data/processed/premier_league_matches_xg_enriched.csv"
)

MODEL5_2025_FILE = Path(
    "data/processed/"
    "premier_league_2025_26_xg_features_with_history.csv"
)

REPORT_DIR = Path(
    "reports/model5"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MODEL 2 FEATURES
# Frozen production feature set
# ============================================================

MODEL2_FEATURES = [
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


# ============================================================
# MODEL 5 XG FEATURES
# Frozen after historical Model 5 experiment
# ============================================================

MODEL5_XG_FEATURES = [
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


MODEL5_FEATURES = (
    MODEL2_FEATURES
    +
    MODEL5_XG_FEATURES
)


# ============================================================
# HELPERS
# ============================================================

def multiclass_brier(
    y_true,
    probabilities,
):
    """
    Multiclass Brier score using one-hot actual outcomes.
    Lower is better.
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    actual = np.zeros_like(
        probabilities
    )

    actual[
        np.arange(
            len(y_true)
        ),
        y_true,
    ] = 1.0

    return np.mean(
        np.sum(
            (
                probabilities
                -
                actual
            )
            ** 2,
            axis=1,
        )
    )


def outcome_to_class(
    frame,
):
    """
    0 = Home
    1 = Draw
    2 = Away
    """

    return np.where(
        frame["HomeGoals"]
        >
        frame["AwayGoals"],
        0,
        np.where(
            frame["HomeGoals"]
            ==
            frame["AwayGoals"],
            1,
            2,
        ),
    )


def probability_matrix(
    home_xg,
    away_xg,
):
    """
    Convert expected goals into H/D/A probabilities using
    exactly the same independent Poisson mechanism used by
    the controlled Model 2 / Model 5 experiments.
    """

    probabilities = []

    for home_goal_mean, away_goal_mean in zip(
        home_xg,
        away_xg,
    ):

        result = independent_poisson_probabilities(
            home_goal_mean,
            away_goal_mean,
            max_goals=8,
        )

        probabilities.append(
            [
                result["home_probability"],
                result["draw_probability"],
                result["away_probability"],
            ]
        )

    return np.asarray(
        probabilities,
        dtype=float,
    )


def fit_goal_models(
    train,
    features,
):
    """
    Fit independent home-goal and away-goal Poisson
    regressions using the frozen alpha and max_iter.
    """

    home_model = PoissonRegressor(
        alpha=0.1,
        max_iter=1000,
    )

    away_model = PoissonRegressor(
        alpha=0.1,
        max_iter=1000,
    )

    home_model.fit(
        train[features],
        train["HomeGoals"],
    )

    away_model.fit(
        train[features],
        train["AwayGoals"],
    )

    return (
        home_model,
        away_model,
    )

def evaluate(
    train,
    test,
    features,
):
    """
    Train goal models, generate H/D/A probabilities,
    and return predictions and metrics.
    """

    home_model, away_model = fit_goal_models(
        train,
        features,
    )

    home_xg = home_model.predict(
        test[features]
    )

    away_xg = away_model.predict(
        test[features]
    )

    probabilities = probability_matrix(
        home_xg,
        away_xg,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    actual = outcome_to_class(
        test
    )

    metrics = {
        "Accuracy":
            accuracy_score(
                actual,
                predictions,
            ),

        "LogLoss":
            log_loss(
                actual,
                probabilities,
                labels=[
                    0,
                    1,
                    2,
                ],
            ),

        "Brier":
            multiclass_brier(
                actual,
                probabilities,
            ),

        "ActualDraws":
            int(
                np.sum(
                    actual == 1
                )
            ),

        "PredictedDraws":
            int(
                np.sum(
                    predictions == 1
                )
            ),
    }

    return (
        home_xg,
        away_xg,
        probabilities,
        predictions,
        actual,
        metrics,
    )


# ============================================================
# LOAD MODEL 2 DATA
# ============================================================

print()
print("FOOTBALL COPILOT")
print("MODEL 5 2025/26 OUT-OF-TIME VALIDATION")
print("=======================================")
print()

model2 = pd.read_csv(
    MODEL2_FILE
)

model2["Date"] = pd.to_datetime(
    model2["Date"],
    format="%Y-%m-%d",
    errors="raise",
)


# ============================================================
# SPLIT MODEL 2 TRAIN / TEST
# ============================================================

model2_train = model2[
    model2["Season"]
    !=
    "2025/26"
].copy()


model2_test = model2[
    model2["Season"]
    ==
    "2025/26"
].copy()


# ============================================================
# LOAD HISTORICAL XG FEATURES
# ============================================================

historical_xg = pd.read_csv(
    HISTORICAL_XG_FILE
)

historical_xg["Date"] = pd.to_datetime(
    historical_xg["Date"],
    errors="raise",
)


historical_xg = historical_xg[
    historical_xg["Season"].isin(
        [
            "2021/22",
            "2022/23",
            "2023/24",
            "2024/25",
        ]
    )
].copy()


# ============================================================
# JOIN HISTORICAL MODEL 2 + XG
# This recreates the Model 5 training population.
# ============================================================

historical_model5 = model2_train.merge(
    historical_xg[
        [
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",
        ]
        +
        MODEL5_XG_FEATURES
    ],
    on=[
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
    ],
    how="inner",
    validate="one_to_one",
)


# ============================================================
# LOAD UNTOUCHED 2025/26 XG FEATURES
# ============================================================

current_xg = pd.read_csv(
    MODEL5_2025_FILE
)

current_xg["Date"] = pd.to_datetime(
    current_xg["Date"],
    format="%Y-%m-%d",
    errors="raise",
)


# ============================================================
# LOCK 2025/26 COMMON COHORT
# ============================================================

model5_test = model2_test.merge(
    current_xg[
        [
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",
        ]
        +
        MODEL5_XG_FEATURES
    ],
    on=[
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
    ],
    how="inner",
    validate="one_to_one",
)


# ============================================================
# SORT TEST COHORTS IDENTICALLY
# ============================================================

sort_columns = [
    "Date",
    "HomeTeam",
    "AwayTeam",
]


model2_test = (
    model2_test
    .sort_values(
        sort_columns
    )
    .reset_index(
        drop=True
    )
)


model5_test = (
    model5_test
    .sort_values(
        sort_columns
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# SAFETY GATES
# ============================================================

print("DATA GATES")
print("==========")

print(
    "Model 2 training matches:",
    len(model2_train),
)

print(
    "Model 5 training matches:",
    len(historical_model5),
)

print(
    "Model 2 2025/26 test matches:",
    len(model2_test),
)

print(
    "Model 5 2025/26 test matches:",
    len(model5_test),
)


if len(model2_test) != 370:
    raise ValueError(
        "Expected frozen Model 2 test cohort of 370 fixtures."
    )


if len(model5_test) != 370:
    raise ValueError(
        "Expected frozen Model 5 test cohort of 370 fixtures."
    )


model2_keys = model2_test[
    [
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
    ]
].reset_index(
    drop=True
)


model5_keys = model5_test[
    [
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
    ]
].reset_index(
    drop=True
)


if not model2_keys.equals(
    model5_keys
):
    raise ValueError(
        "Model 2 and Model 5 test cohorts are not identical."
    )


missing_m2_train = int(
    model2_train[
        MODEL2_FEATURES
    ].isna().sum().sum()
)


missing_m5_train = int(
    historical_model5[
        MODEL5_FEATURES
    ].isna().sum().sum()
)


missing_m2_test = int(
    model2_test[
        MODEL2_FEATURES
    ].isna().sum().sum()
)


missing_m5_test = int(
    model5_test[
        MODEL5_FEATURES
    ].isna().sum().sum()
)


print()
print("MISSING FEATURE VALUES")
print("======================")

print(
    "Model 2 train:",
    missing_m2_train,
)

print(
    "Model 5 train:",
    missing_m5_train,
)

print(
    "Model 2 test:",
    missing_m2_test,
)

print(
    "Model 5 test:",
    missing_m5_test,
)


if (
    missing_m2_train
    or missing_m5_train
    or missing_m2_test
    or missing_m5_test
):
    raise ValueError(
        "Missing feature values detected. "
        "Evaluation aborted."
    )


print()
print(
    "PASS: frozen 370-fixture cohort "
    "and feature completeness confirmed."
)


# ============================================================
# MODEL 2
# ============================================================

(
    model2_home_xg,
    model2_away_xg,
    model2_probabilities,
    model2_predictions,
    actual,
    model2_metrics,
) = evaluate(
    model2_train,
    model2_test,
    MODEL2_FEATURES,
)


# ============================================================
# MODEL 5
# ============================================================

(
    model5_home_xg,
    model5_away_xg,
    model5_probabilities,
    model5_predictions,
    model5_actual,
    model5_metrics,
) = evaluate(
    historical_model5,
    model5_test,
    MODEL5_FEATURES,
)


if not np.array_equal(
    actual,
    model5_actual,
):
    raise ValueError(
        "Actual outcomes differ between Model 2 and Model 5."
    )


# ============================================================
# RESULTS
# ============================================================

accuracy_uplift = (
    model5_metrics["Accuracy"]
    -
    model2_metrics["Accuracy"]
)


logloss_improvement = (
    model2_metrics["LogLoss"]
    -
    model5_metrics["LogLoss"]
)


brier_improvement = (
    model2_metrics["Brier"]
    -
    model5_metrics["Brier"]
)


print()
print("2025/26 OUT-OF-TIME RESULTS")
print("===========================")

print()
print(
    f"Matches: {len(model2_test)}"
)

print()
print(
    f"Model 2 Accuracy: "
    f"{model2_metrics['Accuracy']:.4%}"
)

print(
    f"Model 5 Accuracy: "
    f"{model5_metrics['Accuracy']:.4%}"
)

print(
    f"Accuracy uplift: "
    f"{accuracy_uplift:+.4%}"
)


print()
print(
    f"Model 2 Log Loss: "
    f"{model2_metrics['LogLoss']:.4f}"
)

print(
    f"Model 5 Log Loss: "
    f"{model5_metrics['LogLoss']:.4f}"
)

print(
    f"Log Loss improvement: "
    f"{logloss_improvement:+.4f}"
)


print()
print(
    f"Model 2 Brier: "
    f"{model2_metrics['Brier']:.4f}"
)

print(
    f"Model 5 Brier: "
    f"{model5_metrics['Brier']:.4f}"
)

print(
    f"Brier improvement: "
    f"{brier_improvement:+.4f}"
)


print()
print(
    "Actual draws:",
    model2_metrics["ActualDraws"],
)

print(
    "Model 2 predicted draws:",
    model2_metrics["PredictedDraws"],
)

print(
    "Model 5 predicted draws:",
    model5_metrics["PredictedDraws"],
)


# ============================================================
# PRE-REGISTERED GENERALISATION CRITERIA
# ============================================================

probability_quality_pass = (
    model5_metrics["LogLoss"]
    <
    model2_metrics["LogLoss"]
    and
    model5_metrics["Brier"]
    <
    model2_metrics["Brier"]
)


accuracy_not_materially_worse = (
    accuracy_uplift
    >=
    -0.01
)


generalisation_pass = (
    probability_quality_pass
    and
    accuracy_not_materially_worse
)


print()
print("GENERALISATION CRITERIA")
print("=======================")

print(
    "Model 5 improves Log Loss:",
    model5_metrics["LogLoss"]
    <
    model2_metrics["LogLoss"],
)

print(
    "Model 5 improves Brier:",
    model5_metrics["Brier"]
    <
    model2_metrics["Brier"],
)

print(
    "Accuracy deterioration <= 1pp:",
    accuracy_not_materially_worse,
)


print()
print("OUT-OF-TIME VERDICT")
print("===================")

if generalisation_pass:

    print(
        "PASS: Model 5 xG enrichment "
        "generalises to untouched 2025/26."
    )

else:

    print(
        "FAIL: Model 5 historical uplift "
        "does not meet the locked "
        "2025/26 generalisation criteria."
    )


# ============================================================
# SAVE MATCH-LEVEL RESULTS
# ============================================================

results = model2_test[
    [
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "HomeGoals",
        "AwayGoals",
    ]
].copy()


results[
    "ActualClass"
] = actual


results[
    "Model2HomeXG"
] = model2_home_xg

results[
    "Model2AwayXG"
] = model2_away_xg

results[
    "Model2HomeProbability"
] = model2_probabilities[:, 0]

results[
    "Model2DrawProbability"
] = model2_probabilities[:, 1]

results[
    "Model2AwayProbability"
] = model2_probabilities[:, 2]

results[
    "Model2Prediction"
] = model2_predictions


results[
    "Model5HomeXG"
] = model5_home_xg

results[
    "Model5AwayXG"
] = model5_away_xg

results[
    "Model5HomeProbability"
] = model5_probabilities[:, 0]

results[
    "Model5DrawProbability"
] = model5_probabilities[:, 1]

results[
    "Model5AwayProbability"
] = model5_probabilities[:, 2]

results[
    "Model5Prediction"
] = model5_predictions


results.to_csv(
    REPORT_DIR
    /
    "model5_2025_26_out_of_time.csv",
    index=False,
)


summary = pd.DataFrame(
    [
        {
            "Matches":
                len(model2_test),

            "Model2Accuracy":
                model2_metrics["Accuracy"],

            "Model5Accuracy":
                model5_metrics["Accuracy"],

            "AccuracyUplift":
                accuracy_uplift,

            "Model2LogLoss":
                model2_metrics["LogLoss"],

            "Model5LogLoss":
                model5_metrics["LogLoss"],

            "LogLossImprovement":
                logloss_improvement,

            "Model2Brier":
                model2_metrics["Brier"],

            "Model5Brier":
                model5_metrics["Brier"],

            "BrierImprovement":
                brier_improvement,

            "ActualDraws":
                model2_metrics["ActualDraws"],

            "Model2PredictedDraws":
                model2_metrics["PredictedDraws"],

            "Model5PredictedDraws":
                model5_metrics["PredictedDraws"],

            "GeneralisationPass":
                generalisation_pass,
        }
    ]
)


summary.to_csv(
    REPORT_DIR
    /
    "model5_2025_26_out_of_time_summary.csv",
    index=False,
)


print()
print(
    "Saved:",
    REPORT_DIR
    /
    "model5_2025_26_out_of_time.csv",
)

print(
    "Saved:",
    REPORT_DIR
    /
    "model5_2025_26_out_of_time_summary.csv",
)

print()
print("VALIDATION COMPLETE")
print("===================")