from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss


# --------------------------------------------------
# Files
# --------------------------------------------------

INPUT_FILE = (
    "reports/post_gw5/"
    "challenger_b_predictions.csv"
)

DETAIL_FILE = (
    "reports/post_gw5/"
    "challenger_c_predictions.csv"
)

SUMMARY_FILE = (
    "reports/post_gw5/"
    "challenger_c_summary.csv"
)

SEASON_FILE = (
    "reports/post_gw5/"
    "challenger_c_by_season.csv"
)

WEIGHT_FILE = (
    "reports/post_gw5/"
    "challenger_c_selected_weights.csv"
)


# --------------------------------------------------
# Pre-declared candidate weights
#
# Weight = Challenger B contribution.
# Model 2 receives 1 - weight.
#
# No weights outside this set will be searched.
# --------------------------------------------------

CANDIDATE_WEIGHTS = [
    0.10,
    0.20,
    0.30,
]


LABEL_MAP = {
    "H": 0,
    "D": 1,
    "A": 2,
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def probability_columns(prefix):

    return [
        f"{prefix}HomeProbability",
        f"{prefix}DrawProbability",
        f"{prefix}AwayProbability",
    ]


def actual_numeric(data):

    return (
        data["ActualResult"]
        .map(LABEL_MAP)
        .to_numpy()
    )


def brier_score(
    actual,
    probabilities,
):

    matrix = np.zeros(
        (
            len(actual),
            3,
        )
    )

    for index, value in enumerate(
        actual
    ):

        matrix[
            index,
            LABEL_MAP[value],
        ] = 1.0

    return np.mean(
        np.sum(
            (
                probabilities
                -
                matrix
            )
            ** 2,
            axis=1,
        )
    )


def calculate_metrics(
    data,
    prefix,
):

    probabilities = (
        data[
            probability_columns(
                prefix
            )
        ]
        .to_numpy()
    )

    actual = data[
        "ActualResult"
    ]

    numeric = actual_numeric(
        data
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    reverse_map = {
        0: "H",
        1: "D",
        2: "A",
    }

    predicted_results = np.array(
        [
            reverse_map[value]
            for value
            in predictions
        ]
    )

    accuracy = np.mean(
        predicted_results
        ==
        actual.to_numpy()
    )

    loss = log_loss(
        numeric,
        probabilities,
        labels=[
            0,
            1,
            2,
        ],
    )

    brier = brier_score(
        actual,
        probabilities,
    )

    predicted_draws = int(
        np.sum(
            predicted_results
            ==
            "D"
        )
    )

    actual_draws = int(
        np.sum(
            actual.to_numpy()
            ==
            "D"
        )
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
    }


def blend_probabilities(
    data,
    weight,
):

    m2 = data[
        probability_columns(
            "M2"
        )
    ].to_numpy()

    cb = data[
        probability_columns(
            "CB"
        )
    ].to_numpy()

    return (
        (1.0 - weight)
        * m2
        +
        weight
        * cb
    )


def evaluate_blend(
    data,
    weight,
):

    probabilities = (
        blend_probabilities(
            data,
            weight,
        )
    )

    numeric = actual_numeric(
        data
    )

    loss = log_loss(
        numeric,
        probabilities,
        labels=[
            0,
            1,
            2,
        ],
    )

    brier = brier_score(
        data["ActualResult"],
        probabilities,
    )

    return {
        "Weight":
            weight,

        "LogLoss":
            loss,

        "Brier":
            brier,

        # Equal importance to the two
        # probability-quality metrics.
        #
        # Ranking is calculated separately
        # so their different numeric scales
        # do not distort selection.
    }


def select_weight(
    training_data,
):

    candidates = []

    for weight in (
        CANDIDATE_WEIGHTS
    ):

        candidates.append(
            evaluate_blend(
                training_data,
                weight,
            )
        )

    candidate_frame = (
        pd.DataFrame(
            candidates
        )
    )

    candidate_frame[
        "LogLossRank"
    ] = candidate_frame[
        "LogLoss"
    ].rank(
        method="min"
    )

    candidate_frame[
        "BrierRank"
    ] = candidate_frame[
        "Brier"
    ].rank(
        method="min"
    )

    candidate_frame[
        "CombinedRank"
    ] = (
        candidate_frame[
            "LogLossRank"
        ]
        +
        candidate_frame[
            "BrierRank"
        ]
    )

    candidate_frame = (
        candidate_frame.sort_values(
            [
                "CombinedRank",
                "LogLoss",
                "Brier",
                "Weight",
            ]
        )
    )

    selected = (
        candidate_frame.iloc[0]
    )

    return (
        float(
            selected["Weight"]
        ),
        candidate_frame,
    )


def attach_blend(
    data,
    weight,
):

    data = data.copy()

    probabilities = (
        blend_probabilities(
            data,
            weight,
        )
    )

    data[
        "CCHomeProbability"
    ] = probabilities[:, 0]

    data[
        "CCDrawProbability"
    ] = probabilities[:, 1]

    data[
        "CCAwayProbability"
    ] = probabilities[:, 2]

    labels = np.array(
        [
            "H",
            "D",
            "A",
        ]
    )

    data[
        "CCPredictedResult"
    ] = labels[
        np.argmax(
            probabilities,
            axis=1,
        )
    ]

    data[
        "CCWeightCB"
    ] = weight

    return data


# --------------------------------------------------
# Load existing OOT predictions
# --------------------------------------------------

print()
print("FOOTBALL COPILOT")
print("CHALLENGER C: PROBABILITY ENSEMBLE")
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


TEST_SEASONS = sorted(
    df["Season"].unique()
)


print()
print(
    "Available OOT seasons:",
    TEST_SEASONS,
)


# --------------------------------------------------
# Important constraint
#
# challenger_b_predictions.csv contains only the
# three OOT seasons.
#
# Therefore, for the first OOT season (2023/24),
# there is no earlier Challenger-B OOT season
# available from which to select a weight.
#
# We must NOT tune on 2023/24 and then report
# 2023/24 as untouched OOT.
#
# So:
#
# 2023/24 = calibration season for ensemble weight.
# 2024/25 = first untouched C test.
#
# For 2025/26, the weight may be selected from
# 2023/24 + 2024/25 only.
#
# This preserves temporal integrity.
# --------------------------------------------------

if len(
    TEST_SEASONS
) < 3:

    raise RuntimeError(
        "Expected at least three OOT seasons."
    )


calibration_season = (
    TEST_SEASONS[0]
)

evaluation_seasons = (
    TEST_SEASONS[1:]
)


print()
print(
    "Ensemble calibration starts with:",
    calibration_season,
)

print(
    "Untouched ensemble test seasons:",
    ", ".join(
        evaluation_seasons
    ),
)


# --------------------------------------------------
# Temporal ensemble evaluation
# --------------------------------------------------

all_test_rows = []
weight_rows = []


for test_season in (
    evaluation_seasons
):

    training_seasons = [
        season
        for season in TEST_SEASONS
        if season < test_season
    ]

    training_data = df[
        df["Season"].isin(
            training_seasons
        )
    ].copy()

    test_data = df[
        df["Season"]
        ==
        test_season
    ].copy()

    selected_weight, candidates = (
        select_weight(
            training_data
        )
    )

    print()
    print(
        "Training ensemble weight on:",
        ", ".join(
            training_seasons
        ),
    )

    print(
        "Testing frozen weight on:",
        test_season,
    )

    print(
        "Selected Challenger B weight:",
        f"{selected_weight:.2f}",
    )

    print()
    print(
        candidates[
            [
                "Weight",
                "LogLoss",
                "Brier",
                "CombinedRank",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    test_data = attach_blend(
        test_data,
        selected_weight,
    )

    all_test_rows.append(
        test_data
    )

    weight_rows.append(
        {
            "TestSeason":
                test_season,

            "TrainingSeasons":
                ", ".join(
                    training_seasons
                ),

            "SelectedCBWeight":
                selected_weight,

            "SelectedM2Weight":
                1.0
                -
                selected_weight,
        }
    )


results = pd.concat(
    all_test_rows,
    ignore_index=True,
)

weights = pd.DataFrame(
    weight_rows
)


# --------------------------------------------------
# Overall untouched comparison
# --------------------------------------------------

summary_rows = []


for name, prefix in [
    (
        "Model 2",
        "M2",
    ),
    (
        "Challenger B",
        "CB",
    ),
    (
        "Challenger C",
        "CC",
    ),
]:

    summary_rows.append(
        {
            "Model":
                name,

            **calculate_metrics(
                results,
                prefix,
            ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


print()
print("UNTOUCHED OOT COMPARISON")
print("========================")

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


for season in (
    evaluation_seasons
):

    season_data = results[
        results["Season"]
        ==
        season
    ].copy()

    for name, prefix in [
        (
            "Model 2",
            "M2",
        ),
        (
            "Challenger B",
            "CB",
        ),
        (
            "Challenger C",
            "CC",
        ),
    ]:

        season_rows.append(
            {
                "Season":
                    season,

                "Model":
                    name,

                **calculate_metrics(
                    season_data,
                    prefix,
                ),
            }
        )


season_summary = pd.DataFrame(
    season_rows
)


print()
print("COMPARISON BY TEST SEASON")
print("=========================")

print(
    season_summary.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


# --------------------------------------------------
# Close fixtures
#
# Same H6 definition:
# Model 2 top-v-second probability margin <= 10pp.
# --------------------------------------------------

m2_probabilities = results[
    probability_columns(
        "M2"
    )
].to_numpy()

sorted_probabilities = np.sort(
    m2_probabilities,
    axis=1,
)

results[
    "M2DecisionMargin"
] = (
    sorted_probabilities[:, 2]
    -
    sorted_probabilities[:, 1]
)


close = results[
    results[
        "M2DecisionMargin"
    ]
    <= 0.10
].copy()


print()
print("CLOSE FIXTURES: MODEL 2 MARGIN <=10PP")
print("=====================================")


close_rows = []


for name, prefix in [
    (
        "Model 2",
        "M2",
    ),
    (
        "Challenger B",
        "CB",
    ),
    (
        "Challenger C",
        "CC",
    ),
]:

    close_rows.append(
        {
            "Model":
                name,

            **calculate_metrics(
                close,
                prefix,
            ),
        }
    )


close_summary = pd.DataFrame(
    close_rows
)


print(
    close_summary.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


# --------------------------------------------------
# Promotion criteria
#
# These remain the same in principle:
#
# - Log Loss improves vs Model 2
# - Brier improves vs Model 2
# - Accuracy deterioration no worse than 1pp
# - Improvement not isolated to one test season
#
# With only two untouched C seasons available,
# stability requires probability improvement in
# BOTH 2024/25 and 2025/26.
# --------------------------------------------------

m2 = summary[
    summary["Model"]
    ==
    "Model 2"
].iloc[0]

cc = summary[
    summary["Model"]
    ==
    "Challenger C"
].iloc[0]


log_loss_pass = (
    cc["LogLoss"]
    <
    m2["LogLoss"]
)

brier_pass = (
    cc["Brier"]
    <
    m2["Brier"]
)

accuracy_change = (
    cc["AccuracyPct"]
    -
    m2["AccuracyPct"]
)

accuracy_pass = (
    accuracy_change
    >= -1.0
)


season_pivot = (
    season_summary.pivot(
        index="Season",
        columns="Model",
        values=[
            "LogLoss",
            "Brier",
        ],
    )
)


season_passes = 0


for season in (
    evaluation_seasons
):

    better_log_loss = (
        season_pivot.loc[
            season,
            (
                "LogLoss",
                "Challenger C",
            ),
        ]
        <
        season_pivot.loc[
            season,
            (
                "LogLoss",
                "Model 2",
            ),
        ]
    )

    better_brier = (
        season_pivot.loc[
            season,
            (
                "Brier",
                "Challenger C",
            ),
        ]
        <
        season_pivot.loc[
            season,
            (
                "Brier",
                "Model 2",
            ),
        ]
    )

    if (
        better_log_loss
        and
        better_brier
    ):

        season_passes += 1


stability_pass = (
    season_passes
    ==
    len(
        evaluation_seasons
    )
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
    "Both test seasons improve:",
    "PASS"
    if stability_pass
    else "FAIL",
    (
        f"({season_passes}/"
        f"{len(evaluation_seasons)})"
    ),
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
        "Challenger C qualifies for "
        "historical GW6 simulation."
    )

else:

    print(
        "FULL OOT GATE: FAIL"
    )

    print(
        "Challenger C does not qualify "
        "for promotion."
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

weights.to_csv(
    WEIGHT_FILE,
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

print(
    WEIGHT_FILE
)


print()
print("CHALLENGER C BACKTEST COMPLETE")
print("==============================")