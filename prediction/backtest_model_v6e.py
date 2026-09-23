from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==================================================
# FOOTBALL COPILOT - MODEL 6E
#
# Two-stage architecture:
#
# Stage 1:
#   Learn P(Draw) from pre-match competitiveness.
#
# Stage 2:
#   Preserve the opening market's Home-v-Away
#   directional judgement conditional on NOT draw.
#
#   P(H | not D) = MarketH / (MarketH + MarketA)
#   P(A | not D) = MarketA / (MarketH + MarketA)
#
# Final:
#   P(H) = (1 - P(D)) * P(H | not D)
#   P(A) = (1 - P(D)) * P(A | not D)
#
# No closing-market fields.
# No Matchweek.
# No historical GW6 holdout.
# Same controlled 1,100-match OOT population.
# ==================================================


INPUT_FILE = (
    "data/processed/"
    "prediction_features_v6.csv"
)

OUTPUT_DIR = Path(
    "reports/model6"
)

DETAIL_OUTPUT = (
    OUTPUT_DIR
    /
    "model6e_walk_forward_predictions.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    /
    "model6e_walk_forward_summary.csv"
)

SEASON_OUTPUT = (
    OUTPUT_DIR
    /
    "model6e_walk_forward_by_season.csv"
)


CLASS_ORDER = [
    "H",
    "D",
    "A",
]


WALK_FORWARD = [
    (
        [
            "2021/22",
            "2022/23",
        ],
        "2023/24",
    ),
    (
        [
            "2021/22",
            "2022/23",
            "2023/24",
        ],
        "2024/25",
    ),
    (
        [
            "2021/22",
            "2022/23",
            "2023/24",
            "2024/25",
        ],
        "2025/26",
    ),
]


EXPECTED_TEST_COUNTS = {
    "2023/24": 360,
    "2024/25": 370,
    "2025/26": 370,
}


# ==================================================
# PRE-REGISTERED DRAW FEATURES
#
# These target competitiveness / draw probability,
# not the H-v-A direction.
#
# All are available pre-match and already exist in
# prediction_features_v6.csv.
# ==================================================

DRAW_FEATURES = [

    # Opening 1X2 market shape
    "MarketOpenHomeProb",
    "MarketOpenDrawProb",
    "MarketOpenAwayProb",
    "MarketOpenHomeAwayDiff",

    # Goal environment
    "MarketOpenOver25Prob",
    "MarketOpenUnder25Prob",

    # Asian handicap / competitiveness
    "MarketOpenAHLine",
    "MarketOpenAHHomeProb",
    "MarketOpenAHAwayProb",

    # Result-strength separation
    "RecentPPGDifference",
    "TenMatchPPGDifference",
    "SeasonPPGDifference",

    # Underlying-performance separation
    "XGDDifference",
    "EWXGDDifference",
    "ShotDifference",
    "SOTDifference",

    # Context
    "RestDaysDifference",
]


ENGINEERED_DRAW_FEATURES = [
    "AbsMarketHomeAwayDiff",
    "MarketTopProbability",
    "MarketMargin",
    "AbsAHLine",
    "AbsRecentPPGDifference",
    "AbsTenMatchPPGDifference",
    "AbsSeasonPPGDifference",
    "AbsXGDDifference",
    "AbsEWXGDDifference",
    "AbsShotDifference",
    "AbsSOTDifference",
]


MODEL_FEATURES = (
    DRAW_FEATURES
    +
    ENGINEERED_DRAW_FEATURES
)


# ==================================================
# FIXED MODEL SPECIFICATION
#
# Regularised logistic regression is intentional:
# compact, interpretable and much less prone to the
# instability already seen in Model 6B.
# ==================================================

LOGISTIC_C = 0.1


def fail(message):
    raise RuntimeError(message)


def add_engineered_features(
    frame,
):

    output = frame.copy()

    market_probs = output[
        [
            "MarketOpenHomeProb",
            "MarketOpenDrawProb",
            "MarketOpenAwayProb",
        ]
    ].to_numpy(
        dtype=float
    )

    market_probs = (
        market_probs
        /
        market_probs.sum(
            axis=1,
            keepdims=True,
        )
    )

    sorted_probs = np.sort(
        market_probs,
        axis=1,
    )

    output[
        "AbsMarketHomeAwayDiff"
    ] = np.abs(
        output[
            "MarketOpenHomeAwayDiff"
        ]
    )

    output[
        "MarketTopProbability"
    ] = sorted_probs[
        :,
        -1
    ]

    output[
        "MarketMargin"
    ] = (
        sorted_probs[
            :,
            -1
        ]
        -
        sorted_probs[
            :,
            -2
        ]
    )

    output[
        "AbsAHLine"
    ] = np.abs(
        output[
            "MarketOpenAHLine"
        ]
    )

    output[
        "AbsRecentPPGDifference"
    ] = np.abs(
        output[
            "RecentPPGDifference"
        ]
    )

    output[
        "AbsTenMatchPPGDifference"
    ] = np.abs(
        output[
            "TenMatchPPGDifference"
        ]
    )

    output[
        "AbsSeasonPPGDifference"
    ] = np.abs(
        output[
            "SeasonPPGDifference"
        ]
    )

    output[
        "AbsXGDDifference"
    ] = np.abs(
        output[
            "XGDDifference"
        ]
    )

    output[
        "AbsEWXGDDifference"
    ] = np.abs(
        output[
            "EWXGDDifference"
        ]
    )

    output[
        "AbsShotDifference"
    ] = np.abs(
        output[
            "ShotDifference"
        ]
    )

    output[
        "AbsSOTDifference"
    ] = np.abs(
        output[
            "SOTDifference"
        ]
    )

    return output


def market_probabilities(
    frame,
):

    probabilities = frame[
        [
            "MarketOpenHomeProb",
            "MarketOpenDrawProb",
            "MarketOpenAwayProb",
        ]
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        probabilities
    ).all():

        fail(
            "Opening market probabilities "
            "contain missing/non-finite values"
        )

    row_sums = probabilities.sum(
        axis=1,
        keepdims=True,
    )

    if np.any(
        row_sums <= 0
    ):

        fail(
            "Invalid opening market "
            "probability rows"
        )

    return (
        probabilities
        /
        row_sums
    )


def labels_from_probabilities(
    probabilities,
):

    indices = np.argmax(
        probabilities,
        axis=1,
    )

    return np.array(
        [
            CLASS_ORDER[index]
            for index in indices
        ]
    )


def calculate_metrics(
    actual,
    probabilities,
):

    actual = np.asarray(
        actual
    )

    predicted = (
        labels_from_probabilities(
            probabilities
        )
    )

    accuracy = np.mean(
        predicted
        ==
        actual
    )

    actual_indices = np.array(
        [
            CLASS_ORDER.index(label)
            for label in actual
        ],
        dtype=int,
    )

    clipped = np.clip(
        probabilities,
        1e-15,
        1.0,
    )

    logloss = -np.mean(
        np.log(
            clipped[
                np.arange(
                    len(actual_indices)
                ),
                actual_indices,
            ]
        )
    )

    one_hot = np.zeros_like(
        probabilities,
        dtype=float,
    )

    one_hot[
        np.arange(
            len(actual_indices)
        ),
        actual_indices,
    ] = 1.0

    brier = np.mean(
        np.sum(
            (
                probabilities
                -
                one_hot
            )
            ** 2,
            axis=1,
        )
    )

    predicted_draws = int(
        np.sum(
            predicted
            ==
            "D"
        )
    )

    correct_draws = int(
        np.sum(
            (
                predicted
                ==
                "D"
            )
            &
            (
                actual
                ==
                "D"
            )
        )
    )

    actual_draws = int(
        np.sum(
            actual
            ==
            "D"
        )
    )

    draw_recall = (
        correct_draws
        /
        actual_draws
        if actual_draws
        else 0.0
    )

    draw_precision = (
        correct_draws
        /
        predicted_draws
        if predicted_draws
        else 0.0
    )

    return {
        "Matches":
            len(actual),

        "Accuracy":
            accuracy,

        "LogLoss":
            logloss,

        "Brier":
            brier,

        "PredictedDraws":
            predicted_draws,

        "CorrectDraws":
            correct_draws,

        "ActualDraws":
            actual_draws,

        "DrawRecall":
            draw_recall,

        "DrawPrecision":
            draw_precision,
    }


def make_draw_model():

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    C=LOGISTIC_C,
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def model6e_probabilities(
    frame,
    draw_probability,
):

    market_probs = (
        market_probabilities(
            frame
        )
    )

    market_home = market_probs[
        :,
        0
    ]

    market_away = market_probs[
        :,
        2
    ]

    non_draw_market = (
        market_home
        +
        market_away
    )

    if np.any(
        non_draw_market
        <= 0
    ):

        fail(
            "Invalid non-draw opening "
            "market probabilities"
        )

    conditional_home = (
        market_home
        /
        non_draw_market
    )

    conditional_away = (
        market_away
        /
        non_draw_market
    )

    draw_probability = np.clip(
        draw_probability,
        1e-6,
        1.0 - 1e-6,
    )

    non_draw_probability = (
        1.0
        -
        draw_probability
    )

    home_probability = (
        non_draw_probability
        *
        conditional_home
    )

    away_probability = (
        non_draw_probability
        *
        conditional_away
    )

    probabilities = np.column_stack(
        [
            home_probability,
            draw_probability,
            away_probability,
        ]
    )

    probabilities = (
        probabilities
        /
        probabilities.sum(
            axis=1,
            keepdims=True,
        )
    )

    return probabilities


# ==================================================
# LOAD / VALIDATE
# ==================================================

print()
print("FOOTBALL COPILOT")
print("MODEL 6E TWO-STAGE DRAW MODEL")
print("=============================")


df = pd.read_csv(
    INPUT_FILE
)

df = add_engineered_features(
    df
)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce",
)

if df[
    "Date"
].isna().any():

    fail(
        "Invalid dates in Model 6 data"
    )


required_columns = set(
    [
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "ActualResult",
    ]
    +
    MODEL_FEATURES
)

missing = sorted(
    required_columns
    -
    set(
        df.columns
    )
)

if missing:

    fail(
        "Missing required columns: "
        +
        ", ".join(
            missing
        )
    )


if len(
    df
) != 1733:

    fail(
        f"Expected 1733 rows, "
        f"found {len(df)}"
    )


for season, expected in (
    EXPECTED_TEST_COUNTS.items()
):

    count = int(
        (
            df[
                "Season"
            ]
            ==
            season
        ).sum()
    )

    if count != expected:

        fail(
            f"{season}: expected "
            f"{expected} rows, "
            f"found {count}"
        )


print(
    f"Input rows: {len(df)}"
)

print(
    "Controlled OOT rows: 1100"
)

print(
    "Historical GW6 holdout: untouched"
)

print(
    "Draw model: regularised logistic regression"
)

print(
    f"Logistic C: {LOGISTIC_C}"
)


# ==================================================
# WALK-FORWARD
# ==================================================

prediction_frames = []
season_rows = []


for training_seasons, test_season in (
    WALK_FORWARD
):

    print()
    print(
        "TRAIN:",
        ", ".join(
            training_seasons
        ),
    )

    print(
        "TEST :",
        test_season,
    )

    train = (
        df[
            df[
                "Season"
            ].isin(
                training_seasons
            )
        ]
        .copy()
    )

    test = (
        df[
            df[
                "Season"
            ]
            ==
            test_season
        ]
        .copy()
    )

    y_train_draw = (
        train[
            "ActualResult"
        ]
        ==
        "D"
    ).astype(int)

    draw_model = (
        make_draw_model()
    )

    draw_model.fit(
        train[
            MODEL_FEATURES
        ],
        y_train_draw,
    )

    draw_probability = (
        draw_model.predict_proba(
            test[
                MODEL_FEATURES
            ]
        )[
            :,
            1
        ]
    )

    market_probs = (
        market_probabilities(
            test
        )
    )

    model6e_probs = (
        model6e_probabilities(
            test,
            draw_probability,
        )
    )

    actual = (
        test[
            "ActualResult"
        ]
        .to_numpy()
    )

    market_metrics = (
        calculate_metrics(
            actual,
            market_probs,
        )
    )

    model6e_metrics = (
        calculate_metrics(
            actual,
            model6e_probs,
        )
    )

    market_metrics[
        "Season"
    ] = test_season

    market_metrics[
        "Model"
    ] = "MarketOpen"

    model6e_metrics[
        "Season"
    ] = test_season

    model6e_metrics[
        "Model"
    ] = "Model6E"

    season_rows.extend(
        [
            market_metrics,
            model6e_metrics,
        ]
    )

    output = (
        test[
            [
                "Season",
                "Date",
                "HomeTeam",
                "AwayTeam",
                "ActualResult",
            ]
        ]
        .copy()
    )

    output[
        "MarketHomeProb"
    ] = market_probs[
        :,
        0
    ]

    output[
        "MarketDrawProb"
    ] = market_probs[
        :,
        1
    ]

    output[
        "MarketAwayProb"
    ] = market_probs[
        :,
        2
    ]

    output[
        "MarketPick"
    ] = (
        labels_from_probabilities(
            market_probs
        )
    )

    output[
        "Model6EDrawModelProb"
    ] = draw_probability

    output[
        "Model6EHomeProb"
    ] = model6e_probs[
        :,
        0
    ]

    output[
        "Model6EDrawProb"
    ] = model6e_probs[
        :,
        1
    ]

    output[
        "Model6EAwayProb"
    ] = model6e_probs[
        :,
        2
    ]

    output[
        "Model6EPick"
    ] = (
        labels_from_probabilities(
            model6e_probs
        )
    )

    prediction_frames.append(
        output
    )


# ==================================================
# COMBINE OOT
# ==================================================

predictions = pd.concat(
    prediction_frames,
    ignore_index=True,
)

if len(
    predictions
) != 1100:

    fail(
        f"Expected 1100 OOT rows, "
        f"found {len(predictions)}"
    )


actual = predictions[
    "ActualResult"
].to_numpy()


market_probs = predictions[
    [
        "MarketHomeProb",
        "MarketDrawProb",
        "MarketAwayProb",
    ]
].to_numpy(
    dtype=float
)


model6e_probs = predictions[
    [
        "Model6EHomeProb",
        "Model6EDrawProb",
        "Model6EAwayProb",
    ]
].to_numpy(
    dtype=float
)


market_overall = (
    calculate_metrics(
        actual,
        market_probs,
    )
)

market_overall[
    "Model"
] = "MarketOpen"


model6e_overall = (
    calculate_metrics(
        actual,
        model6e_probs,
    )
)

model6e_overall[
    "Model"
] = "Model6E"


summary = pd.DataFrame(
    [
        market_overall,
        model6e_overall,
    ]
)


by_season = pd.DataFrame(
    season_rows
)


# ==================================================
# OUTPUT
# ==================================================

print()
print("OVERALL 1,100-MATCH OOT RESULTS")
print("===============================")


display = summary.copy()

display[
    "AccuracyPct"
] = (
    display[
        "Accuracy"
    ]
    *
    100
)

display[
    "DrawRecallPct"
] = (
    display[
        "DrawRecall"
    ]
    *
    100
)

display[
    "DrawPrecisionPct"
] = (
    display[
        "DrawPrecision"
    ]
    *
    100
)


print(
    display[
        [
            "Model",
            "Matches",
            "AccuracyPct",
            "LogLoss",
            "Brier",
            "PredictedDraws",
            "CorrectDraws",
            "ActualDraws",
            "DrawRecallPct",
            "DrawPrecisionPct",
        ]
    ].to_string(
        index=False,
        formatters={
            "AccuracyPct":
                lambda x:
                    f"{x:.4f}",

            "LogLoss":
                lambda x:
                    f"{x:.4f}",

            "Brier":
                lambda x:
                    f"{x:.4f}",

            "DrawRecallPct":
                lambda x:
                    f"{x:.2f}",

            "DrawPrecisionPct":
                lambda x:
                    f"{x:.2f}",
        },
    )
)


print()
print("RESULTS BY TEST SEASON")
print("======================")


season_display = (
    by_season.copy()
)

season_display[
    "AccuracyPct"
] = (
    season_display[
        "Accuracy"
    ]
    *
    100
)

season_display[
    "DrawRecallPct"
] = (
    season_display[
        "DrawRecall"
    ]
    *
    100
)


print(
    season_display[
        [
            "Season",
            "Model",
            "Matches",
            "AccuracyPct",
            "LogLoss",
            "Brier",
            "PredictedDraws",
            "CorrectDraws",
            "DrawRecallPct",
        ]
    ].to_string(
        index=False,
        formatters={
            "AccuracyPct":
                lambda x:
                    f"{x:.4f}",

            "LogLoss":
                lambda x:
                    f"{x:.4f}",

            "Brier":
                lambda x:
                    f"{x:.4f}",

            "DrawRecallPct":
                lambda x:
                    f"{x:.2f}",
        },
    )
)


# ==================================================
# DECISION
#
# Original significant-uplift gate remains unchanged.
#
# Additionally, Model 6E must add genuine value over
# the stronger opening-market benchmark.
# ==================================================

market_row = summary[
    summary[
        "Model"
    ]
    ==
    "MarketOpen"
].iloc[0]

model6e_row = summary[
    summary[
        "Model"
    ]
    ==
    "Model6E"
].iloc[0]


original_accuracy_gate = (
    model6e_row[
        "Accuracy"
    ]
    >=
    0.55
)

original_logloss_gate = (
    model6e_row[
        "LogLoss"
    ]
    <
    1.0037
)

original_brier_gate = (
    model6e_row[
        "Brier"
    ]
    <
    0.5999
)


market_accuracy_gate = (
    model6e_row[
        "Accuracy"
    ]
    >
    market_row[
        "Accuracy"
    ]
)

market_logloss_gate = (
    model6e_row[
        "LogLoss"
    ]
    <
    market_row[
        "LogLoss"
    ]
)

market_brier_gate = (
    model6e_row[
        "Brier"
    ]
    <
    market_row[
        "Brier"
    ]
)


season_market = (
    by_season[
        by_season[
            "Model"
        ]
        ==
        "MarketOpen"
    ]
    .set_index(
        "Season"
    )
)

season_6e = (
    by_season[
        by_season[
            "Model"
        ]
        ==
        "Model6E"
    ]
    .set_index(
        "Season"
    )
)


better_probability_seasons = 0

for season in (
    EXPECTED_TEST_COUNTS
):

    if (
        season_6e.loc[
            season,
            "LogLoss"
        ]
        <
        season_market.loc[
            season,
            "LogLoss"
        ]
        and
        season_6e.loc[
            season,
            "Brier"
        ]
        <
        season_market.loc[
            season,
            "Brier"
        ]
    ):

        better_probability_seasons += 1


stability_gate = (
    better_probability_seasons
    >=
    2
)


print()
print("MODEL 6E DECISION")
print("=================")

print(
    "Original accuracy >= 55%:",
    "PASS"
    if original_accuracy_gate
    else "FAIL",
)

print(
    "Original Log Loss < 1.0037:",
    "PASS"
    if original_logloss_gate
    else "FAIL",
)

print(
    "Original Brier < 0.5999:",
    "PASS"
    if original_brier_gate
    else "FAIL",
)

print()
print(
    "Accuracy > opening market:",
    "PASS"
    if market_accuracy_gate
    else "FAIL",
)

print(
    "Log Loss < opening market:",
    "PASS"
    if market_logloss_gate
    else "FAIL",
)

print(
    "Brier < opening market:",
    "PASS"
    if market_brier_gate
    else "FAIL",
)

print(
    "LL + Brier better than market "
    "in >=2/3 seasons:",
    (
        "PASS"
        if stability_gate
        else "FAIL"
    ),
    f"({better_probability_seasons}/3)",
)


full_gate = (
    original_accuracy_gate
    and
    original_logloss_gate
    and
    original_brier_gate
    and
    market_accuracy_gate
    and
    market_logloss_gate
    and
    market_brier_gate
    and
    stability_gate
)


print()
print(
    "MODEL 6E FULL QUALIFICATION:",
    "PASS"
    if full_gate
    else "FAIL",
)


# ==================================================
# SAVE
# ==================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


predictions.to_csv(
    DETAIL_OUTPUT,
    index=False,
)

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)

by_season.to_csv(
    SEASON_OUTPUT,
    index=False,
)


print()
print("Saved:")
print(
    " ",
    DETAIL_OUTPUT,
)

print(
    " ",
    SUMMARY_OUTPUT,
)

print(
    " ",
    SEASON_OUTPUT,
)

print()
print(
    "MODEL 6E WALK-FORWARD BACKTEST COMPLETE"
)

print(
    "======================================="
)