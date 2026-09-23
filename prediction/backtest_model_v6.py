from pathlib import Path

import numpy as np
import pandas as pd

from scipy.special import softmax
from scipy.stats import poisson

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==================================================
# FILES
# ==================================================

INPUT_FILE = "data/processed/prediction_features_v6.csv"

DETAIL_OUTPUT = (
    "reports/model6/model6_walk_forward_predictions.csv"
)

SUMMARY_OUTPUT = (
    "reports/model6/model6_walk_forward_summary.csv"
)

SEASON_OUTPUT = (
    "reports/model6/model6_walk_forward_by_season.csv"
)


# ==================================================
# FROZEN CONTROLLED MODEL 2 FEATURES
# ==================================================

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


# ==================================================
# MODEL 6 FEATURE SETS
#
# Closing-market fields are deliberately excluded.
# Matchweek is deliberately excluded.
#
# 6A = opening market only
# 6B = opening market + compact football enrichment
# 6C = market prior + learned residual correction
# ==================================================

MARKET_FEATURES = [
    "MarketOpenHomeProb",
    "MarketOpenDrawProb",
    "MarketOpenAwayProb",
    "MarketOpenHomeAwayDiff",
    "MarketOpenOver25Prob",
    "MarketOpenUnder25Prob",
    "MarketOpenAHLine",
    "MarketOpenAHHomeProb",
    "MarketOpenAHAwayProb",
]

FOOTBALL_FEATURES = [
    # Existing Model 2 relative-strength information
    "RecentPPGDifference",
    "TenMatchPPGDifference",
    "SeasonPPGDifference",
    "AttackVsDefenceHome",
    "AttackVsDefenceAway",

    # Underlying xG
    "HomeXGFor10",
    "HomeXGAgainst10",
    "HomeXGD10",
    "AwayXGFor10",
    "AwayXGAgainst10",
    "AwayXGD10",
    "HomeEWXGFor",
    "HomeEWXGAgainst",
    "HomeEWXGD",
    "AwayEWXGFor",
    "AwayEWXGAgainst",
    "AwayEWXGD",

    # Venue xG
    "HomeVenueXGFor5",
    "HomeVenueXGAgainst5",
    "AwayVenueXGFor5",
    "AwayVenueXGAgainst5",

    # Matchup / relative xG
    "XGDDifference",
    "EWXGDDifference",
    "XGAttackVsDefenceHome",
    "XGAttackVsDefenceAway",

    # Shots / shots on target
    "ShotDifference",
    "SOTDifference",
    "SOTAttackVsDefenceHome",
    "SOTAttackVsDefenceAway",

    # Efficiency / over-under performance
    "HomeXGPerShot10",
    "AwayXGPerShot10",
    "HomeSOTRate10",
    "AwaySOTRate10",
    "HomeFinishingVsXG10",
    "AwayFinishingVsXG10",
    "HomeDefenceVsXGA10",
    "AwayDefenceVsXGA10",

    # Context
    "HomeRestDays",
    "AwayRestDays",
    "RestDaysDifference",
]

MODEL6B_FEATURES = (
    MARKET_FEATURES
    +
    FOOTBALL_FEATURES
)

RESIDUAL_FEATURES = FOOTBALL_FEATURES


# ==================================================
# WALK-FORWARD DESIGN
# ==================================================

WALK_FORWARD = [
    (
        ["2021/22", "2022/23"],
        "2023/24",
    ),
    (
        ["2021/22", "2022/23", "2023/24"],
        "2024/25",
    ),
    (
        ["2021/22", "2022/23", "2023/24", "2024/25"],
        "2025/26",
    ),
]

CLASS_ORDER = ["H", "D", "A"]

# Pre-agreed significant-uplift research gate.
TARGET_ACCURACY = 0.55
MODEL2_REFERENCE_LOGLOSS = 1.0037
MODEL2_REFERENCE_BRIER = 0.5999


# ==================================================
# HELPERS
# ==================================================

def fail(message):
    raise RuntimeError(message)


def actual_result(home_goals, away_goals):
    if home_goals > away_goals:
        return "H"
    if home_goals == away_goals:
        return "D"
    return "A"


def poisson_probabilities(
    home_expected_goals,
    away_expected_goals,
    max_goals=8,
):
    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    for home_goals in range(max_goals + 1):
        home_probability = poisson.pmf(
            home_goals,
            home_expected_goals,
        )

        for away_goals in range(max_goals + 1):
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

    total = home_win + draw + away_win

    return np.array(
        [
            home_win / total,
            draw / total,
            away_win / total,
        ],
        dtype=float,
    )


def aligned_probabilities(model, X):
    raw = model.predict_proba(X)

    output = np.zeros(
        (len(X), len(CLASS_ORDER)),
        dtype=float,
    )

    for source_index, label in enumerate(
        model.classes_
    ):
        target_index = CLASS_ORDER.index(label)

        output[:, target_index] = (
            raw[:, source_index]
        )

    row_sums = output.sum(
        axis=1,
        keepdims=True,
    )

    if np.any(row_sums <= 0):
        fail(
            "Classifier produced invalid "
            "probability rows"
        )

    return output / row_sums


def market_probabilities(frame):
    probabilities = frame[
        [
            "MarketOpenHomeProb",
            "MarketOpenDrawProb",
            "MarketOpenAwayProb",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(
        probabilities
    ).all():
        fail(
            "Opening 1X2 market probabilities "
            "contain missing/non-finite values"
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


def residual_targets(
    actual,
    base_probabilities,
    epsilon=1e-12,
):
    """
    Multinomial residual target.

    For each training observation:
      residual = one-hot(actual) - market prior

    Model 6C learns a probability correction using
    football features only.  At prediction time the
    correction is added in log-probability space and
    renormalised with softmax.

    This preserves the opening market as the prior
    rather than asking the model to relearn it.
    """

    one_hot = np.zeros(
        (
            len(actual),
            len(CLASS_ORDER),
        ),
        dtype=float,
    )

    for i, label in enumerate(actual):
        one_hot[
            i,
            CLASS_ORDER.index(label),
        ] = 1.0

    residual = (
        one_hot
        -
        base_probabilities
    )

    return residual


def brier_score(actual, probabilities):
    one_hot = np.zeros_like(
        probabilities,
        dtype=float,
    )

    for i, label in enumerate(actual):
        one_hot[
            i,
            CLASS_ORDER.index(label),
        ] = 1.0

    return np.mean(
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


def metric_row(
    name,
    actual,
    probabilities,
):
    predicted = np.array(
        [
            CLASS_ORDER[index]
            for index in np.argmax(
                probabilities,
                axis=1,
            )
        ]
    )

    accuracy = np.mean(
        predicted
        ==
        np.asarray(actual)
    )

    # Calculate multiclass Log Loss explicitly using the
    # project's probability order H, D, A.
    #
    # This avoids sklearn's lexicographic class ordering
    # (A, D, H) being applied to H, D, A probability columns.
    actual_indices = np.array(
        [
            CLASS_ORDER.index(label)
            for label in actual
        ],
        dtype=int,
    )

    clipped_probabilities = np.clip(
        probabilities,
        1e-15,
        1.0 - 1e-15,
    )

    ll = -np.mean(
        np.log(
            clipped_probabilities[
                np.arange(len(actual_indices)),
                actual_indices,
            ]
        )
    )

    brier = brier_score(
        actual,
        probabilities,
    )

    predicted_draws = int(
        np.sum(
            predicted == "D"
        )
    )

    actual_draws = int(
        np.sum(
            np.asarray(actual) == "D"
        )
    )

    correct_draws = int(
        np.sum(
            (predicted == "D")
            &
            (
                np.asarray(actual)
                ==
                "D"
            )
        )
    )

    return {
        "Model": name,
        "Matches": len(actual),
        "Accuracy": accuracy,
        "LogLoss": ll,
        "Brier": brier,
        "PredictedDraws": predicted_draws,
        "ActualDraws": actual_draws,
        "CorrectDraws": correct_draws,
    }


def make_hgb_classifier():
    """
    Fixed specification.  No test-season tuning.
    """
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=250,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def make_residual_model():
    """
    One regularised linear residual model per class.

    A deliberately conservative residual learner is
    used to reduce over-fitting to football features.
    """
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
                    random_state=42,
                ),
            ),
        ]
    )


def fit_residual_correction(
    X,
    actual,
    base_probabilities,
):
    """
    Convert the continuous market residual into
    three binary correction directions.

    Each class model learns whether the realised
    outcome supplied positive evidence relative to
    the market prior.  Its centred probability is
    later used as a conservative logit adjustment.
    """
    residual = residual_targets(
        actual,
        base_probabilities,
    )

    models = []

    for class_index in range(
        len(CLASS_ORDER)
    ):
        target = (
            residual[
                :,
                class_index,
            ]
            >
            0
        ).astype(int)

        model = make_residual_model()

        model.fit(
            X,
            target,
        )

        models.append(model)

    return models


def apply_residual_correction(
    models,
    X,
    base_probabilities,
    correction_scale=0.35,
):
    corrections = np.zeros_like(
        base_probabilities,
        dtype=float,
    )

    for class_index, model in enumerate(
        models
    ):
        positive_probability = (
            model.predict_proba(X)[
                :,
                1
            ]
        )

        corrections[
            :,
            class_index
        ] = (
            positive_probability
            -
            0.5
        )

    log_prior = np.log(
        np.clip(
            base_probabilities,
            1e-12,
            1.0,
        )
    )

    adjusted_logits = (
        log_prior
        +
        correction_scale
        *
        corrections
    )

    return softmax(
        adjusted_logits,
        axis=1,
    )


# ==================================================
# LOAD / VALIDATE
# ==================================================

print()
print("FOOTBALL COPILOT")
print("MODEL 6 WALK-FORWARD BACKTEST")
print("=============================")

df = pd.read_csv(
    INPUT_FILE
)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce",
)

if df["Date"].isna().any():
    fail(
        "Invalid dates in Model 6 feature file"
    )

required_columns = set(
    [
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "HomeGoals",
        "AwayGoals",
        "ActualResult",
    ]
    +
    MODEL2_FEATURES
    +
    MODEL6B_FEATURES
)

missing_columns = sorted(
    required_columns
    -
    set(df.columns)
)

if missing_columns:
    fail(
        "Missing required columns: "
        +
        ", ".join(missing_columns)
    )

if len(df) != 1733:
    fail(
        f"Expected 1733 Model 6 rows, "
        f"found {len(df)}"
    )

if df[
    [
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
    ]
].duplicated().any():
    fail(
        "Duplicate fixture keys in Model 6 data"
    )

expected_test_counts = {
    "2023/24": 360,
    "2024/25": 370,
    "2025/26": 370,
}

for season, expected in (
    expected_test_counts.items()
):
    actual = int(
        (
            df["Season"]
            ==
            season
        ).sum()
    )

    if actual != expected:
        fail(
            f"{season}: expected {expected} "
            f"rows, found {actual}"
        )

print(
    f"Input rows: {len(df)}"
)

print(
    "Controlled OOT rows: 1100"
)


# ==================================================
# WALK-FORWARD
# ==================================================

prediction_frames = []
season_metric_rows = []

for training_seasons, test_season in (
    WALK_FORWARD
):
    print()
    print(
        "TRAIN:",
        ", ".join(training_seasons),
    )

    print(
        "TEST :",
        test_season,
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

    if len(test) != (
        expected_test_counts[
            test_season
        ]
    ):
        fail(
            f"Unexpected test count for "
            f"{test_season}"
        )

    y_train = train[
        "ActualResult"
    ].to_numpy()

    y_test = test[
        "ActualResult"
    ].to_numpy()

    # ----------------------------------------------
    # CONTROLLED MODEL 2 RECREATION
    # ----------------------------------------------

    model2_home = PoissonRegressor(
        alpha=0.1,
        max_iter=1000,
    )

    model2_away = PoissonRegressor(
        alpha=0.1,
        max_iter=1000,
    )

    model2_home.fit(
        train[MODEL2_FEATURES],
        train["HomeGoals"],
    )

    model2_away.fit(
        train[MODEL2_FEATURES],
        train["AwayGoals"],
    )

    model2_home_xg = model2_home.predict(
        test[MODEL2_FEATURES]
    )

    model2_away_xg = model2_away.predict(
        test[MODEL2_FEATURES]
    )

    model2_probs = np.vstack(
        [
            poisson_probabilities(
                home_xg,
                away_xg,
            )
            for home_xg, away_xg
            in zip(
                model2_home_xg,
                model2_away_xg,
            )
        ]
    )

    # ----------------------------------------------
    # OPENING MARKET BASELINE
    # ----------------------------------------------

    market_train_probs = (
        market_probabilities(train)
    )

    market_test_probs = (
        market_probabilities(test)
    )

    # ----------------------------------------------
    # MODEL 6A
    #
    # Opening market itself.  Kept as an explicit
    # benchmark so we know whether later models add
    # anything beyond the prior.
    # ----------------------------------------------

    model6a_probs = market_test_probs.copy()

    # ----------------------------------------------
    # MODEL 6B
    #
    # Direct nonlinear model using opening market
    # plus genuinely enriched football information.
    # ----------------------------------------------

    model6b = make_hgb_classifier()

    model6b.fit(
        train[MODEL6B_FEATURES],
        y_train,
    )

    model6b_probs = aligned_probabilities(
        model6b,
        test[MODEL6B_FEATURES],
    )

    # ----------------------------------------------
    # MODEL 6C
    #
    # Market-prior residual correction.
    # Football information adjusts rather than
    # replaces the opening market.
    # ----------------------------------------------

    residual_models = (
        fit_residual_correction(
            train[RESIDUAL_FEATURES],
            y_train,
            market_train_probs,
        )
    )

    model6c_probs = (
        apply_residual_correction(
            residual_models,
            test[RESIDUAL_FEATURES],
            market_test_probs,
        )
    )

    # ----------------------------------------------
    # RECORD FIXTURE-LEVEL PREDICTIONS
    # ----------------------------------------------

    output = test[
        [
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",
            "HomeGoals",
            "AwayGoals",
            "ActualResult",
        ]
    ].copy()

    probability_sets = {
        "Model2": model2_probs,
        "MarketOpen": market_test_probs,
        "Model6A": model6a_probs,
        "Model6B": model6b_probs,
        "Model6C": model6c_probs,
    }

    for prefix, probabilities in (
        probability_sets.items()
    ):
        output[
            f"{prefix}HomeProb"
        ] = probabilities[:, 0]

        output[
            f"{prefix}DrawProb"
        ] = probabilities[:, 1]

        output[
            f"{prefix}AwayProb"
        ] = probabilities[:, 2]

        output[
            f"{prefix}PredictedResult"
        ] = [
            CLASS_ORDER[index]
            for index in np.argmax(
                probabilities,
                axis=1,
            )
        ]

    prediction_frames.append(
        output
    )

    # ----------------------------------------------
    # SEASON METRICS
    # ----------------------------------------------

    for name, probabilities in (
        probability_sets.items()
    ):
        row = metric_row(
            name,
            y_test,
            probabilities,
        )

        row["Season"] = test_season

        season_metric_rows.append(
            row
        )


# ==================================================
# COMBINE 1,100 OOT PREDICTIONS
# ==================================================

predictions = pd.concat(
    prediction_frames,
    ignore_index=True,
)

if len(predictions) != 1100:
    fail(
        f"Expected 1100 OOT predictions, "
        f"found {len(predictions)}"
    )


# ==================================================
# OVERALL METRICS
# ==================================================

overall_rows = []

overall_probability_columns = {
    "Model2": [
        "Model2HomeProb",
        "Model2DrawProb",
        "Model2AwayProb",
    ],
    "MarketOpen": [
        "MarketOpenHomeProb",
        "MarketOpenDrawProb",
        "MarketOpenAwayProb",
    ],
    "Model6A": [
        "Model6AHomeProb",
        "Model6ADrawProb",
        "Model6AAwayProb",
    ],
    "Model6B": [
        "Model6BHomeProb",
        "Model6BDrawProb",
        "Model6BAwayProb",
    ],
    "Model6C": [
        "Model6CHomeProb",
        "Model6CDrawProb",
        "Model6CAwayProb",
    ],
}

for name, columns in (
    overall_probability_columns.items()
):
    probabilities = predictions[
        columns
    ].to_numpy(dtype=float)

    overall_rows.append(
        metric_row(
            name,
            predictions[
                "ActualResult"
            ].to_numpy(),
            probabilities,
        )
    )

summary = pd.DataFrame(
    overall_rows
)

by_season = pd.DataFrame(
    season_metric_rows
)


# ==================================================
# SIGNIFICANT-UPLIFT GATE
# ==================================================

print()
print("OVERALL 1,100-MATCH OOT RESULTS")
print("===============================")

display_summary = summary.copy()

display_summary["AccuracyPct"] = (
    100
    *
    display_summary["Accuracy"]
)

print(
    display_summary[
        [
            "Model",
            "Matches",
            "AccuracyPct",
            "LogLoss",
            "Brier",
            "PredictedDraws",
            "CorrectDraws",
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

season_display["AccuracyPct"] = (
    100
    *
    season_display["Accuracy"]
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
print("PRE-AGREED SIGNIFICANT-UPLIFT GATE")
print("==================================")

for model_name in [
    "Model6B",
    "Model6C",
]:
    row = summary[
        summary["Model"]
        ==
        model_name
    ].iloc[0]

    accuracy_pass = (
        row["Accuracy"]
        >=
        TARGET_ACCURACY
    )

    logloss_pass = (
        row["LogLoss"]
        <
        MODEL2_REFERENCE_LOGLOSS
    )

    brier_pass = (
        row["Brier"]
        <
        MODEL2_REFERENCE_BRIER
    )

    season_rows = by_season[
        by_season["Model"]
        ==
        model_name
    ].copy()

    model2_season_rows = (
        by_season[
            by_season["Model"]
            ==
            "Model2"
        ]
        .set_index("Season")
    )

    improved_seasons = 0

    for _, season_row in (
        season_rows.iterrows()
    ):
        baseline = model2_season_rows.loc[
            season_row["Season"]
        ]

        if (
            season_row["LogLoss"]
            <
            baseline["LogLoss"]
            and
            season_row["Brier"]
            <
            baseline["Brier"]
        ):
            improved_seasons += 1

    stability_pass = (
        improved_seasons >= 2
    )

    overall_pass = (
        accuracy_pass
        and
        logloss_pass
        and
        brier_pass
        and
        stability_pass
    )

    print()
    print(model_name)

    print(
        "  Accuracy >= 55%:",
        "PASS"
        if accuracy_pass
        else "FAIL",
    )

    print(
        "  Log Loss < Model 2:",
        "PASS"
        if logloss_pass
        else "FAIL",
    )

    print(
        "  Brier < Model 2:",
        "PASS"
        if brier_pass
        else "FAIL",
    )

    print(
        "  LL + Brier better in >=2/3 seasons:",
        "PASS"
        if stability_pass
        else "FAIL",
        f"({improved_seasons}/3)",
    )

    print(
        "  SIGNIFICANT-UPLIFT GATE:",
        "PASS"
        if overall_pass
        else "FAIL",
    )


# ==================================================
# SAVE
# ==================================================

Path(
    "reports/model6"
).mkdir(
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
print(" ", DETAIL_OUTPUT)
print(" ", SUMMARY_OUTPUT)
print(" ", SEASON_OUTPUT)

print()
print("MODEL 6 WALK-FORWARD BACKTEST COMPLETE")
print("======================================")
