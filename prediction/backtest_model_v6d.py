from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==================================================
# FOOTBALL COPILOT - MODEL 6D
#
# Hypothesis:
# The opening market is already a strong prior.
# Football Copilot should only override that prior
# where historical football-v-market disagreement
# provides evidence that the market's argmax is wrong.
#
# This experiment does NOT:
# - use closing market information
# - touch the historical GW6 holdout
# - retune Model 6B / 6C
# - change the Model 2 baseline
#
# It tests selective market overrides only.
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
    "model6d_walk_forward_predictions.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    /
    "model6d_walk_forward_summary.csv"
)

SEASON_OUTPUT = (
    OUTPUT_DIR
    /
    "model6d_walk_forward_by_season.csv"
)

SELECTION_OUTPUT = (
    OUTPUT_DIR
    /
    "model6d_inner_selection.csv"
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
# FEATURES
#
# Deliberately compact.
#
# We are not asking a model to relearn 1X2.
# We are asking:
#
# "When the football evidence disagrees with the
# opening market, is that disagreement informative?"
# ==================================================

DISAGREEMENT_FEATURES = [

    # Market strength / uncertainty
    "MarketOpenHomeProb",
    "MarketOpenDrawProb",
    "MarketOpenAwayProb",
    "MarketOpenHomeAwayDiff",
    "MarketOpenOver25Prob",
    "MarketOpenUnder25Prob",
    "MarketOpenAHLine",
    "MarketOpenAHHomeProb",
    "MarketOpenAHAwayProb",

    # Existing result-strength differences
    "RecentPPGDifference",
    "TenMatchPPGDifference",
    "SeasonPPGDifference",

    # Underlying performance differences
    "XGDDifference",
    "EWXGDDifference",
    "ShotDifference",
    "SOTDifference",

    # Attack / defence matchup
    "AttackVsDefenceHome",
    "AttackVsDefenceAway",
    "XGAttackVsDefenceHome",
    "XGAttackVsDefenceAway",
    "SOTAttackVsDefenceHome",
    "SOTAttackVsDefenceAway",

    # Context
    "RestDaysDifference",
]


# ==================================================
# PRE-REGISTERED SELECTIVE-OVERRIDE GRID
#
# Selection occurs using training history only.
#
# min_confidence:
# probability that market argmax is wrong.
#
# max_market_margin:
# only challenge relatively uncertain market calls.
#
# The final OOT season is never used to choose these.
# ==================================================

CONFIDENCE_GRID = [
    0.55,
    0.60,
    0.65,
    0.70,
]

MARKET_MARGIN_GRID = [
    0.05,
    0.10,
    0.15,
    0.20,
]


# ==================================================
# HELPERS
# ==================================================

def fail(message):
    raise RuntimeError(message)


def market_probabilities(frame):

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


def market_margin(
    probabilities,
):

    sorted_probabilities = np.sort(
        probabilities,
        axis=1,
    )

    return (
        sorted_probabilities[
            :,
            -1
        ]
        -
        sorted_probabilities[
            :,
            -2
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
    }


def make_override_model():

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
                    C=0.1,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def add_disagreement_features(
    frame,
):

    output = frame.copy()

    # ----------------------------------------------
    # Football directional signals
    #
    # Positive = home stronger
    # Negative = away stronger
    # ----------------------------------------------

    output[
        "FootballComposite"
    ] = (
        output[
            "RecentPPGDifference"
        ]
        +
        output[
            "TenMatchPPGDifference"
        ]
        +
        output[
            "SeasonPPGDifference"
        ]
        +
        output[
            "XGDDifference"
        ]
        +
        output[
            "EWXGDDifference"
        ]
    )

    output[
        "MarketDirectionalStrength"
    ] = (
        output[
            "MarketOpenHomeProb"
        ]
        -
        output[
            "MarketOpenAwayProb"
        ]
    )

    output[
        "FootballVsMarketDirection"
    ] = (
        output[
            "FootballComposite"
        ]
        *
        output[
            "MarketDirectionalStrength"
        ]
    )

    output[
        "AbsoluteFootballComposite"
    ] = np.abs(
        output[
            "FootballComposite"
        ]
    )

    output[
        "AbsoluteMarketDirection"
    ] = np.abs(
        output[
            "MarketDirectionalStrength"
        ]
    )

    output[
        "PPGvsMarketDirection"
    ] = (
        output[
            "TenMatchPPGDifference"
        ]
        *
        output[
            "MarketDirectionalStrength"
        ]
    )

    output[
        "XGvsMarketDirection"
    ] = (
        output[
            "EWXGDDifference"
        ]
        *
        output[
            "MarketDirectionalStrength"
        ]
    )

    output[
        "SOTvsMarketDirection"
    ] = (
        output[
            "SOTDifference"
        ]
        *
        output[
            "MarketDirectionalStrength"
        ]
    )

    return output


ENGINEERED_FEATURES = [
    "FootballComposite",
    "MarketDirectionalStrength",
    "FootballVsMarketDirection",
    "AbsoluteFootballComposite",
    "AbsoluteMarketDirection",
    "PPGvsMarketDirection",
    "XGvsMarketDirection",
    "SOTvsMarketDirection",
]

MODEL_FEATURES = (
    DISAGREEMENT_FEATURES
    +
    ENGINEERED_FEATURES
)


def build_wrong_market_target(
    frame,
):

    market_probs = (
        market_probabilities(
            frame
        )
    )

    market_pick = (
        labels_from_probabilities(
            market_probs
        )
    )

    actual = frame[
        "ActualResult"
    ].to_numpy()

    return (
        market_pick
        !=
        actual
    ).astype(int)


def challenger_result(
    frame,
):

    market_probs = (
        market_probabilities(
            frame
        )
    )

    market_pick = (
        labels_from_probabilities(
            market_probs
        )
    )

    # ----------------------------------------------
    # If challenging H, choose strongest D/A.
    # If challenging A, choose strongest H/D.
    # If market ever selects D, choose strongest H/A.
    #
    # This is deterministic and uses only opening
    # market probabilities.
    # ----------------------------------------------

    challenger = []

    for i, pick in enumerate(
        market_pick
    ):

        probabilities = {
            "H":
                market_probs[
                    i,
                    0
                ],

            "D":
                market_probs[
                    i,
                    1
                ],

            "A":
                market_probs[
                    i,
                    2
                ],
        }

        alternatives = {
            label:
                probability

            for label, probability
            in probabilities.items()

            if label != pick
        }

        challenger.append(
            max(
                alternatives,
                key=alternatives.get,
            )
        )

    return np.array(
        challenger
    )


def override_probabilities(
    frame,
    wrong_probability,
    min_confidence,
    max_market_margin,
):

    market_probs = (
        market_probabilities(
            frame
        )
    )

    market_pick = (
        labels_from_probabilities(
            market_probs
        )
    )

    challenger = (
        challenger_result(
            frame
        )
    )

    margins = market_margin(
        market_probs
    )

    override = (
        (
            wrong_probability
            >=
            min_confidence
        )
        &
        (
            margins
            <=
            max_market_margin
        )
    )

    adjusted = (
        market_probs.copy()
    )

    # ----------------------------------------------
    # Selective override:
    #
    # We do NOT invent new probability magnitudes.
    # Where an override is triggered, swap the
    # market probabilities of the market pick and
    # strongest alternative.
    #
    # This keeps the market's calibration structure
    # while changing only the selected ordering.
    # ----------------------------------------------

    for i in np.where(
        override
    )[0]:

        original_label = (
            market_pick[i]
        )

        challenger_label = (
            challenger[i]
        )

        original_index = (
            CLASS_ORDER.index(
                original_label
            )
        )

        challenger_index = (
            CLASS_ORDER.index(
                challenger_label
            )
        )

        temp = (
            adjusted[
                i,
                original_index
            ]
        )

        adjusted[
            i,
            original_index
        ] = (
            adjusted[
                i,
                challenger_index
            ]
        )

        adjusted[
            i,
            challenger_index
        ] = temp

    return (
        adjusted,
        override,
        market_pick,
        challenger,
        margins,
    )


def choose_override_rule(
    historical_train,
):

    seasons = (
        historical_train[
            "Season"
        ]
        .drop_duplicates()
        .tolist()
    )

    if len(seasons) < 2:

        fail(
            "Model 6D needs at least two "
            "historical seasons for inner "
            "temporal selection"
        )

    validation_season = (
        seasons[-1]
    )

    inner_train_seasons = (
        seasons[:-1]
    )

    inner_train = (
        historical_train[
            historical_train[
                "Season"
            ].isin(
                inner_train_seasons
            )
        ]
        .copy()
    )

    validation = (
        historical_train[
            historical_train[
                "Season"
            ]
            ==
            validation_season
        ]
        .copy()
    )

    if len(
        inner_train
    ) == 0 or len(
        validation
    ) == 0:

        fail(
            "Invalid inner temporal split"
        )

    target = (
        build_wrong_market_target(
            inner_train
        )
    )

    model = (
        make_override_model()
    )

    model.fit(
        inner_train[
            MODEL_FEATURES
        ],
        target,
    )

    wrong_probability = (
        model.predict_proba(
            validation[
                MODEL_FEATURES
            ]
        )[
            :,
            1
        ]
    )

    market_probs = (
        market_probabilities(
            validation
        )
    )

    market_metrics = (
        calculate_metrics(
            validation[
                "ActualResult"
            ].to_numpy(),
            market_probs,
        )
    )

    candidates = []

    for confidence in (
        CONFIDENCE_GRID
    ):

        for margin in (
            MARKET_MARGIN_GRID
        ):

            (
                adjusted,
                override,
                _,
                _,
                _,
            ) = (
                override_probabilities(
                    validation,
                    wrong_probability,
                    confidence,
                    margin,
                )
            )

            metrics = (
                calculate_metrics(
                    validation[
                        "ActualResult"
                    ].to_numpy(),
                    adjusted,
                )
            )

            candidates.append(
                {
                    "InnerValidationSeason":
                        validation_season,

                    "MinConfidence":
                        confidence,

                    "MaxMarketMargin":
                        margin,

                    "Overrides":
                        int(
                            override.sum()
                        ),

                    "Accuracy":
                        metrics[
                            "Accuracy"
                        ],

                    "LogLoss":
                        metrics[
                            "LogLoss"
                        ],

                    "Brier":
                        metrics[
                            "Brier"
                        ],

                    "AccuracyVsMarket":
                        (
                            metrics[
                                "Accuracy"
                            ]
                            -
                            market_metrics[
                                "Accuracy"
                            ]
                        ),

                    "LogLossVsMarket":
                        (
                            metrics[
                                "LogLoss"
                            ]
                            -
                            market_metrics[
                                "LogLoss"
                            ]
                        ),

                    "BrierVsMarket":
                        (
                            metrics[
                                "Brier"
                            ]
                            -
                            market_metrics[
                                "Brier"
                            ]
                        ),
                }
            )

    candidates = (
        pd.DataFrame(
            candidates
        )
    )

    # ----------------------------------------------
    # Selection rule:
    #
    # 1. Must not worsen validation Log Loss.
    # 2. Must not worsen validation Brier.
    # 3. Among surviving rules, maximise accuracy.
    # 4. Then lowest Log Loss.
    # 5. Then lowest Brier.
    # 6. Then fewer overrides.
    #
    # If no rule survives, choose NO OVERRIDE.
    # ----------------------------------------------

    eligible = (
        candidates[
            (
                candidates[
                    "LogLoss"
                ]
                <=
                market_metrics[
                    "LogLoss"
                ]
            )
            &
            (
                candidates[
                    "Brier"
                ]
                <=
                market_metrics[
                    "Brier"
                ]
            )
        ]
        .copy()
    )

    if len(
        eligible
    ) == 0:

        return (
            None,
            candidates,
            validation_season,
            market_metrics,
        )

    eligible = (
        eligible.sort_values(
            [
                "Accuracy",
                "LogLoss",
                "Brier",
                "Overrides",
            ],
            ascending=[
                False,
                True,
                True,
                True,
            ],
        )
    )

    selected = (
        eligible.iloc[0]
    )

    return (
        selected,
        candidates,
        validation_season,
        market_metrics,
    )


# ==================================================
# LOAD / VALIDATE
# ==================================================

print()
print("FOOTBALL COPILOT")
print("MODEL 6D SELECTIVE MARKET OVERRIDE")
print("==================================")

df = pd.read_csv(
    INPUT_FILE
)

# Build the Model 6D-only disagreement features before
# validating MODEL_FEATURES.  These columns are derived
# inside this script and are not expected to exist in
# prediction_features_v6.csv.
df = add_disagreement_features(
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


# ==================================================
# WALK-FORWARD TEST
# ==================================================

prediction_frames = []
season_rows = []
selection_rows = []


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

    # ----------------------------------------------
    # Inner temporal selection using training
    # history only.
    # ----------------------------------------------

    (
        selected,
        candidates,
        validation_season,
        inner_market_metrics,
    ) = (
        choose_override_rule(
            train
        )
    )

    candidates[
        "OuterTestSeason"
    ] = test_season

    candidates[
        "Selected"
    ] = False

    if selected is None:

        selected_confidence = None
        selected_margin = None

        print(
            "  Inner validation:",
            validation_season,
        )

        print(
            "  Selected rule: "
            "NO OVERRIDE"
        )

    else:

        selected_confidence = float(
            selected[
                "MinConfidence"
            ]
        )

        selected_margin = float(
            selected[
                "MaxMarketMargin"
            ]
        )

        selected_mask = (
            (
                candidates[
                    "MinConfidence"
                ]
                ==
                selected_confidence
            )
            &
            (
                candidates[
                    "MaxMarketMargin"
                ]
                ==
                selected_margin
            )
        )

        candidates.loc[
            selected_mask,
            "Selected",
        ] = True

        print(
            "  Inner validation:",
            validation_season,
        )

        print(
            "  Selected confidence:",
            f"{selected_confidence:.2f}",
        )

        print(
            "  Selected max market margin:",
            f"{selected_margin:.2f}",
        )

    selection_rows.append(
        candidates
    )

    # ----------------------------------------------
    # Refit override detector on ALL information
    # available before the outer test season.
    # ----------------------------------------------

    target = (
        build_wrong_market_target(
            train
        )
    )

    final_model = (
        make_override_model()
    )

    final_model.fit(
        train[
            MODEL_FEATURES
        ],
        target,
    )

    wrong_probability = (
        final_model.predict_proba(
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

    if selected is None:

        model6d_probs = (
            market_probs.copy()
        )

        override = np.zeros(
            len(test),
            dtype=bool,
        )

        market_pick = (
            labels_from_probabilities(
                market_probs
            )
        )

        challenger = (
            challenger_result(
                test
            )
        )

        margins = (
            market_margin(
                market_probs
            )
        )

    else:

        (
            model6d_probs,
            override,
            market_pick,
            challenger,
            margins,
        ) = (
            override_probabilities(
                test,
                wrong_probability,
                selected_confidence,
                selected_margin,
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

    model6d_metrics = (
        calculate_metrics(
            actual,
            model6d_probs,
        )
    )

    market_metrics[
        "Season"
    ] = test_season

    market_metrics[
        "Model"
    ] = "MarketOpen"

    model6d_metrics[
        "Season"
    ] = test_season

    model6d_metrics[
        "Model"
    ] = "Model6D"

    model6d_metrics[
        "Overrides"
    ] = int(
        override.sum()
    )

    model6d_metrics[
        "CorrectOverrides"
    ] = int(
        np.sum(
            override
            &
            (
                challenger
                ==
                actual
            )
        )
    )

    season_rows.extend(
        [
            market_metrics,
            model6d_metrics,
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
    ] = market_pick

    output[
        "MarketMargin"
    ] = margins

    output[
        "MarketWrongProbability"
    ] = wrong_probability

    output[
        "Override"
    ] = override

    output[
        "ChallengerPick"
    ] = challenger

    output[
        "Model6DHomeProb"
    ] = model6d_probs[
        :,
        0
    ]

    output[
        "Model6DDrawProb"
    ] = model6d_probs[
        :,
        1
    ]

    output[
        "Model6DAwayProb"
    ] = model6d_probs[
        :,
        2
    ]

    output[
        "Model6DPick"
    ] = (
        labels_from_probabilities(
            model6d_probs
        )
    )

    prediction_frames.append(
        output
    )


# ==================================================
# COMBINE OOT RESULTS
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


market_probs = predictions[
    [
        "MarketHomeProb",
        "MarketDrawProb",
        "MarketAwayProb",
    ]
].to_numpy(
    dtype=float
)

model6d_probs = predictions[
    [
        "Model6DHomeProb",
        "Model6DDrawProb",
        "Model6DAwayProb",
    ]
].to_numpy(
    dtype=float
)

actual = predictions[
    "ActualResult"
].to_numpy()


market_overall = (
    calculate_metrics(
        actual,
        market_probs,
    )
)

market_overall[
    "Model"
] = "MarketOpen"


model6d_overall = (
    calculate_metrics(
        actual,
        model6d_probs,
    )
)

model6d_overall[
    "Model"
] = "Model6D"

model6d_overall[
    "Overrides"
] = int(
    predictions[
        "Override"
    ].sum()
)

model6d_overall[
    "CorrectOverrides"
] = int(
    (
        predictions[
            "Override"
        ]
        &
        (
            predictions[
                "ChallengerPick"
            ]
            ==
            predictions[
                "ActualResult"
            ]
        )
    ).sum()
)


summary = pd.DataFrame(
    [
        market_overall,
        model6d_overall,
    ]
)

by_season = pd.DataFrame(
    season_rows
)

selection = pd.concat(
    selection_rows,
    ignore_index=True,
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
            "Overrides",
            "CorrectOverrides",
        ]
    ].to_string(
        index=False,
        formatters={
            "AccuracyPct":
                lambda value:
                    f"{value:.4f}",

            "LogLoss":
                lambda value:
                    f"{value:.4f}",

            "Brier":
                lambda value:
                    f"{value:.4f}",
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
            "Overrides",
            "CorrectOverrides",
        ]
    ].to_string(
        index=False,
        formatters={
            "AccuracyPct":
                lambda value:
                    f"{value:.4f}",

            "LogLoss":
                lambda value:
                    f"{value:.4f}",

            "Brier":
                lambda value:
                    f"{value:.4f}",
        },
    )
)


# ==================================================
# DECISION AGAINST OPENING MARKET
#
# Model 6D only has value if the selective overrides
# add to the already-strong opening market.
# ==================================================

market_row = summary[
    summary[
        "Model"
    ]
    ==
    "MarketOpen"
].iloc[0]

model6d_row = summary[
    summary[
        "Model"
    ]
    ==
    "Model6D"
].iloc[0]


accuracy_improved = (
    model6d_row[
        "Accuracy"
    ]
    >
    market_row[
        "Accuracy"
    ]
)

logloss_not_worse = (
    model6d_row[
        "LogLoss"
    ]
    <=
    market_row[
        "LogLoss"
    ]
)

brier_not_worse = (
    model6d_row[
        "Brier"
    ]
    <=
    market_row[
        "Brier"
    ]
)


print()
print("MODEL 6D DECISION")
print("=================")

print(
    "Accuracy > opening market:",
    "PASS"
    if accuracy_improved
    else "FAIL",
)

print(
    "Log Loss <= opening market:",
    "PASS"
    if logloss_not_worse
    else "FAIL",
)

print(
    "Brier <= opening market:",
    "PASS"
    if brier_not_worse
    else "FAIL",
)

print(
    "MODEL 6D ADDS VALUE:",
    (
        "PASS"
        if (
            accuracy_improved
            and
            logloss_not_worse
            and
            brier_not_worse
        )
        else
        "FAIL"
    ),
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

selection.to_csv(
    SELECTION_OUTPUT,
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
print(
    " ",
    SELECTION_OUTPUT,
)

print()
print(
    "MODEL 6D WALK-FORWARD BACKTEST COMPLETE"
)
print(
    "======================================="
)