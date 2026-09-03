from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import log_loss


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from prediction.dixon_coles import independent_poisson_probabilities


INPUT_FILE = "data/processed/prediction_features_v2.csv"

OUTPUT_MATCHES = "reports/model3/model3b_draw_calibration_backtest.csv"
OUTPUT_SUMMARY = "reports/model3/model3b_draw_calibration_summary.csv"
OUTPUT_BY_SEASON = "reports/model3/model3b_draw_calibration_by_season.csv"


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


def actual_result(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def predicted_result(
    home_probability: float,
    draw_probability: float,
    away_probability: float,
) -> str:
    probs = {
        "H": home_probability,
        "D": draw_probability,
        "A": away_probability,
    }
    return max(probs, key=probs.get)


def calibrate_draw_probability(
    home_probability: float,
    draw_probability: float,
    away_probability: float,
    multiplier: float,
):
    adjusted_home = home_probability
    adjusted_draw = draw_probability * multiplier
    adjusted_away = away_probability

    total = adjusted_home + adjusted_draw + adjusted_away

    return (
        adjusted_home / total,
        adjusted_draw / total,
        adjusted_away / total,
    )


def brier_score_multiclass(
    actual_results,
    probabilities,
):
    mapping = {
        "H": np.array([1.0, 0.0, 0.0]),
        "D": np.array([0.0, 1.0, 0.0]),
        "A": np.array([0.0, 0.0, 1.0]),
    }

    total = 0.0

    for actual, probs in zip(actual_results, probabilities):
        target = mapping[actual]
        total += np.sum((np.asarray(probs) - target) ** 2)

    return total / len(actual_results)


def build_model2_predictions(
    train: pd.DataFrame,
    predict_df: pd.DataFrame,
):
    home_model = PoissonRegressor(
        alpha=0.1,
        max_iter=1000,
    )

    away_model = PoissonRegressor(
        alpha=0.1,
        max_iter=1000,
    )

    home_model.fit(
        train[MODEL_2_FEATURES],
        train["HomeGoals"],
    )

    away_model.fit(
        train[MODEL_2_FEATURES],
        train["AwayGoals"],
    )

    home_xg = home_model.predict(
        predict_df[MODEL_2_FEATURES]
    )

    away_xg = away_model.predict(
        predict_df[MODEL_2_FEATURES]
    )

    rows = []

    for i in range(len(predict_df)):
        hxg = float(home_xg[i])
        axg = float(away_xg[i])

        probs = independent_poisson_probabilities(
            home_xg=hxg,
            away_xg=axg,
            max_goals=8,
        )

        rows.append(
            {
                "HomeXG": hxg,
                "AwayXG": axg,
                "HomeProbability": probs["home_probability"],
                "DrawProbability": probs["draw_probability"],
                "AwayProbability": probs["away_probability"],
                "ModalScore": probs["modal_score"],
            }
        )

    return pd.DataFrame(rows)


def objective_for_multiplier(
    multiplier: float,
    actual_results,
    probabilities_df: pd.DataFrame,
):
    adjusted_probs = []

    for _, row in probabilities_df.iterrows():
        h, d, a = calibrate_draw_probability(
            row["HomeProbability"],
            row["DrawProbability"],
            row["AwayProbability"],
            multiplier,
        )

        adjusted_probs.append(
            [h, d, a]
        )

    y_true = [
        {"H": 0, "D": 1, "A": 2}[result]
        for result in actual_results
    ]

    return log_loss(
        y_true,
        adjusted_probs,
        labels=[0, 1, 2],
    )


print()
print("FOOTBALL COPILOT")
print("MODEL 3B - DRAW CALIBRATION")
print("===========================")

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce",
)

df = df.sort_values(
    "Date"
).reset_index(drop=True)

seasons = sorted(df["Season"].unique())

print()
print("Seasons:", seasons)

all_rows = []
season_rows = []

for test_index in range(2, len(seasons)):

    test_season = seasons[test_index]
    training_seasons = seasons[:test_index]

    train = df[
        df["Season"].isin(training_seasons)
    ].copy()

    test = df[
        df["Season"] == test_season
    ].copy().reset_index(drop=True)

    print()
    print(
        "Training on:",
        ", ".join(training_seasons),
    )

    print(
        "Testing on:",
        test_season,
    )

    #
    # IMPORTANT:
    # calibration multiplier must be fitted only on
    # historical training-period predictions
    #
    calibration_rows = []

    for validation_index in range(1, len(training_seasons)):

        validation_season = training_seasons[
            validation_index
        ]

        prior_seasons = training_seasons[
            :validation_index
        ]

        calibration_train = df[
            df["Season"].isin(prior_seasons)
        ].copy()

        calibration_test = df[
            df["Season"] == validation_season
        ].copy().reset_index(drop=True)

        if len(calibration_train) == 0:
            continue

        preds = build_model2_predictions(
            calibration_train,
            calibration_test,
        )

        for i in range(len(calibration_test)):
            calibration_rows.append(
                {
                    "ActualResult": actual_result(
                        int(
                            calibration_test.loc[
                                i,
                                "HomeGoals",
                            ]
                        ),
                        int(
                            calibration_test.loc[
                                i,
                                "AwayGoals",
                            ]
                        ),
                    ),
                    "HomeProbability":
                        preds.loc[
                            i,
                            "HomeProbability",
                        ],
                    "DrawProbability":
                        preds.loc[
                            i,
                            "DrawProbability",
                        ],
                    "AwayProbability":
                        preds.loc[
                            i,
                            "AwayProbability",
                        ],
                }
            )

    calibration_df = pd.DataFrame(
        calibration_rows
    )

    if len(calibration_df) == 0:
        fitted_multiplier = 1.0
    else:
        optimisation = minimize_scalar(
            objective_for_multiplier,
            bounds=(0.50, 2.00),
            method="bounded",
            args=(
                calibration_df["ActualResult"].tolist(),
                calibration_df[
                    [
                        "HomeProbability",
                        "DrawProbability",
                        "AwayProbability",
                    ]
                ],
            ),
        )

        fitted_multiplier = float(
            optimisation.x
        )

    print(
        f"Fitted draw multiplier: "
        f"{fitted_multiplier:.4f}"
    )

    test_predictions = build_model2_predictions(
        train,
        test,
    )

    season_model2_probs = []
    season_model3b_probs = []

    season_actual = []
    season_model2_predictions = []
    season_model3b_predictions = []

    for i in range(len(test)):

        actual = actual_result(
            int(test.loc[i, "HomeGoals"]),
            int(test.loc[i, "AwayGoals"]),
        )

        h2 = float(
            test_predictions.loc[
                i,
                "HomeProbability",
            ]
        )

        d2 = float(
            test_predictions.loc[
                i,
                "DrawProbability",
            ]
        )

        a2 = float(
            test_predictions.loc[
                i,
                "AwayProbability",
            ]
        )

        h3, d3, a3 = calibrate_draw_probability(
            h2,
            d2,
            a2,
            fitted_multiplier,
        )

        pred2 = predicted_result(
            h2,
            d2,
            a2,
        )

        pred3 = predicted_result(
            h3,
            d3,
            a3,
        )

        season_actual.append(actual)

        season_model2_probs.append(
            [h2, d2, a2]
        )

        season_model3b_probs.append(
            [h3, d3, a3]
        )

        season_model2_predictions.append(
            pred2
        )

        season_model3b_predictions.append(
            pred3
        )

        all_rows.append(
            {
                "Season": test_season,
                "Date": test.loc[i, "Date"],
                "HomeTeam": test.loc[i, "HomeTeam"],
                "AwayTeam": test.loc[i, "AwayTeam"],
                "HomeGoals": int(
                    test.loc[i, "HomeGoals"]
                ),
                "AwayGoals": int(
                    test.loc[i, "AwayGoals"]
                ),
                "ActualResult": actual,

                "HomeXG":
                    test_predictions.loc[
                        i,
                        "HomeXG",
                    ],

                "AwayXG":
                    test_predictions.loc[
                        i,
                        "AwayXG",
                    ],

                "Model2HomeProbability": h2,
                "Model2DrawProbability": d2,
                "Model2AwayProbability": a2,

                "Model3BHomeProbability": h3,
                "Model3BDrawProbability": d3,
                "Model3BAwayProbability": a3,

                "Model2Prediction": pred2,
                "Model3BPrediction": pred3,

                "Model2ModalScore":
                    test_predictions.loc[
                        i,
                        "ModalScore",
                    ],

                "DrawMultiplier":
                    fitted_multiplier,
            }
        )

    actual_numeric = [
        {"H": 0, "D": 1, "A": 2}[x]
        for x in season_actual
    ]

    model2_accuracy = (
        np.mean(
            np.array(
                season_model2_predictions
            )
            ==
            np.array(
                season_actual
            )
        )
        * 100
    )

    model3b_accuracy = (
        np.mean(
            np.array(
                season_model3b_predictions
            )
            ==
            np.array(
                season_actual
            )
        )
        * 100
    )

    model2_logloss = log_loss(
        actual_numeric,
        season_model2_probs,
        labels=[0, 1, 2],
    )

    model3b_logloss = log_loss(
        actual_numeric,
        season_model3b_probs,
        labels=[0, 1, 2],
    )

    model2_brier = brier_score_multiclass(
        season_actual,
        season_model2_probs,
    )

    model3b_brier = brier_score_multiclass(
        season_actual,
        season_model3b_probs,
    )

    model2_draws = sum(
        x == "D"
        for x in season_model2_predictions
    )

    model3b_draws = sum(
        x == "D"
        for x in season_model3b_predictions
    )

    actual_draws = sum(
        x == "D"
        for x in season_actual
    )

    model3b_draw_recall = (
        sum(
            predicted == "D"
            and actual == "D"
            for predicted, actual
            in zip(
                season_model3b_predictions,
                season_actual,
            )
        )
        /
        actual_draws
        * 100
        if actual_draws
        else np.nan
    )

    season_rows.append(
        {
            "Season":
                test_season,

            "Matches":
                len(test),

            "DrawMultiplier":
                fitted_multiplier,

            "Model2Accuracy":
                model2_accuracy,

            "Model3BAccuracy":
                model3b_accuracy,

            "Model2LogLoss":
                model2_logloss,

            "Model3BLogLoss":
                model3b_logloss,

            "Model2Brier":
                model2_brier,

            "Model3BBrier":
                model3b_brier,

            "ActualDraws":
                actual_draws,

            "Model2PredictedDraws":
                model2_draws,

            "Model3BPredictedDraws":
                model3b_draws,

            "Model3BDrawRecall":
                model3b_draw_recall,
        }
    )


results = pd.DataFrame(
    all_rows
)

by_season = pd.DataFrame(
    season_rows
)


actual_all = results[
    "ActualResult"
].tolist()

actual_numeric_all = [
    {"H": 0, "D": 1, "A": 2}[x]
    for x in actual_all
]

model2_probs_all = results[
    [
        "Model2HomeProbability",
        "Model2DrawProbability",
        "Model2AwayProbability",
    ]
].to_numpy()

model3b_probs_all = results[
    [
        "Model3BHomeProbability",
        "Model3BDrawProbability",
        "Model3BAwayProbability",
    ]
].to_numpy()

model2_predictions_all = results[
    "Model2Prediction"
].tolist()

model3b_predictions_all = results[
    "Model3BPrediction"
].tolist()


model2_accuracy_all = (
    np.mean(
        np.array(
            model2_predictions_all
        )
        ==
        np.array(
            actual_all
        )
    )
    * 100
)

model3b_accuracy_all = (
    np.mean(
        np.array(
            model3b_predictions_all
        )
        ==
        np.array(
            actual_all
        )
    )
    * 100
)

model2_logloss_all = log_loss(
    actual_numeric_all,
    model2_probs_all,
    labels=[0, 1, 2],
)

model3b_logloss_all = log_loss(
    actual_numeric_all,
    model3b_probs_all,
    labels=[0, 1, 2],
)

model2_brier_all = brier_score_multiclass(
    actual_all,
    model2_probs_all,
)

model3b_brier_all = brier_score_multiclass(
    actual_all,
    model3b_probs_all,
)

actual_draws_all = sum(
    result == "D"
    for result in actual_all
)

model2_draws_all = sum(
    result == "D"
    for result in model2_predictions_all
)

model3b_draws_all = sum(
    result == "D"
    for result in model3b_predictions_all
)

model3b_correct_draws = sum(
    predicted == "D"
    and actual == "D"
    for predicted, actual
    in zip(
        model3b_predictions_all,
        actual_all,
    )
)

model3b_draw_recall = (
    model3b_correct_draws
    /
    actual_draws_all
    * 100
    if actual_draws_all
    else np.nan
)

model3b_draw_precision = (
    model3b_correct_draws
    /
    model3b_draws_all
    * 100
    if model3b_draws_all
    else np.nan
)


summary = pd.DataFrame(
    [
        {
            "Matches":
                len(results),

            "ActualDraws":
                actual_draws_all,

            "ActualDrawPct":
                (
                    actual_draws_all
                    /
                    len(results)
                    * 100
                ),

            "Model2Accuracy":
                model2_accuracy_all,

            "Model3BAccuracy":
                model3b_accuracy_all,

            "Model2LogLoss":
                model2_logloss_all,

            "Model3BLogLoss":
                model3b_logloss_all,

            "Model2Brier":
                model2_brier_all,

            "Model3BBrier":
                model3b_brier_all,

            "Model2PredictedDraws":
                model2_draws_all,

            "Model2PredictedDrawPct":
                (
                    model2_draws_all
                    /
                    len(results)
                    * 100
                ),

            "Model3BPredictedDraws":
                model3b_draws_all,

            "Model3BPredictedDrawPct":
                (
                    model3b_draws_all
                    /
                    len(results)
                    * 100
                ),

            "Model3BCorrectDraws":
                model3b_correct_draws,

            "Model3BDrawRecall":
                model3b_draw_recall,

            "Model3BDrawPrecision":
                model3b_draw_precision,

            "MeanDrawMultiplier":
                by_season[
                    "DrawMultiplier"
                ].mean(),
        }
    ]
)


print()
print("OVERALL RESULT")
print("==============")

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print()
print("BY SEASON")
print("=========")

print(
    by_season.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print()
print("VERDICT")
print("=======")

if (
    model3b_logloss_all < model2_logloss_all
    and
    model3b_brier_all < model2_brier_all
):
    print(
        "Model 3B improves both probability quality metrics."
    )

elif (
    model3b_logloss_all < model2_logloss_all
    or
    model3b_brier_all < model2_brier_all
):
    print(
        "Model 3B improves one probability quality metric "
        "but not both."
    )

else:
    print(
        "Model 3B does not improve probability quality. "
        "Do not promote it solely because it predicts more draws."
    )


Path(
    "reports/model3"
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

print(OUTPUT_MATCHES)
print(OUTPUT_SUMMARY)
print(OUTPUT_BY_SEASON)

print()
print("MODEL 3B BACKTEST COMPLETE")
print("==========================")