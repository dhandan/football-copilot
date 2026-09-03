from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.linear_model import PoissonRegressor


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
)


# --------------------------------------------------
# Files
# --------------------------------------------------

INPUT_FILE = (
    "data/processed/"
    "prediction_features_v2.csv"
)

MATCH_RESULTS_FILE = (
    "reports/model3/"
    "draw_xg_match_diagnostics.csv"
)

XG_GAP_FILE = (
    "reports/model3/"
    "draw_xg_gap_summary.csv"
)

TOTAL_XG_FILE = (
    "reports/model3/"
    "draw_total_xg_summary.csv"
)

DRAW_CALIBRATION_FILE = (
    "reports/model3/"
    "draw_probability_calibration.csv"
)

SUMMARY_FILE = (
    "reports/model3/"
    "draw_xg_diagnostic_summary.csv"
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
# Helpers
# --------------------------------------------------

def actual_result(
    home_goals: int,
    away_goals: int,
) -> str:

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

    probabilities = {
        "H": home_probability,
        "D": draw_probability,
        "A": away_probability,
    }

    return max(
        probabilities,
        key=probabilities.get,
    )


def safe_percentage(
    numerator,
    denominator,
):

    if denominator == 0:
        return np.nan

    return (
        numerator
        / denominator
        * 100
    )


# --------------------------------------------------
# Load data
# --------------------------------------------------

print()
print("FOOTBALL COPILOT")
print("DRAW / XG DIAGNOSTIC")
print("====================")

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

seasons = sorted(
    df["Season"].unique()
)

print()
print(
    "Seasons:",
    seasons,
)


# --------------------------------------------------
# Walk-forward predictions
# --------------------------------------------------

rows = []


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

    test = test.reset_index(
        drop=True
    )

    home_xg_values = home_model.predict(
        test[
            MODEL_2_FEATURES
        ]
    )

    away_xg_values = away_model.predict(
        test[
            MODEL_2_FEATURES
        ]
    )

    for index in range(
        len(test)
    ):

        home_xg = float(
            home_xg_values[index]
        )

        away_xg = float(
            away_xg_values[index]
        )

        probabilities = (
            independent_poisson_probabilities(
                home_xg=home_xg,
                away_xg=away_xg,
                max_goals=8,
            )
        )

        home_probability = (
            probabilities[
                "home_probability"
            ]
        )

        draw_probability = (
            probabilities[
                "draw_probability"
            ]
        )

        away_probability = (
            probabilities[
                "away_probability"
            ]
        )

        prediction = predicted_result(
            home_probability,
            draw_probability,
            away_probability,
        )

        actual = actual_result(
            int(
                test.loc[
                    index,
                    "HomeGoals",
                ]
            ),
            int(
                test.loc[
                    index,
                    "AwayGoals",
                ]
            ),
        )

        xg_gap = abs(
            home_xg
            -
            away_xg
        )

        total_xg = (
            home_xg
            +
            away_xg
        )

        highest_non_draw_probability = max(
            home_probability,
            away_probability,
        )

        draw_margin_to_top = (
            draw_probability
            -
            highest_non_draw_probability
        )

        rows.append(
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

                "HomeGoals":
                    int(
                        test.loc[
                            index,
                            "HomeGoals",
                        ]
                    ),

                "AwayGoals":
                    int(
                        test.loc[
                            index,
                            "AwayGoals",
                        ]
                    ),

                "ActualResult":
                    actual,

                "PredictedResult":
                    prediction,

                "HomeXG":
                    home_xg,

                "AwayXG":
                    away_xg,

                "XGGap":
                    xg_gap,

                "TotalXG":
                    total_xg,

                "HomeProbability":
                    home_probability,

                "DrawProbability":
                    draw_probability,

                "AwayProbability":
                    away_probability,

                "HighestNonDrawProbability":
                    highest_non_draw_probability,

                "DrawMarginToTop":
                    draw_margin_to_top,

                "IsActualDraw":
                    actual == "D",

                "IsPredictedDraw":
                    prediction == "D",

                "ModalScore":
                    probabilities[
                        "modal_score"
                    ],
            }
        )


results = pd.DataFrame(
    rows
)


# --------------------------------------------------
# XG gap bands
# --------------------------------------------------

gap_bins = [
    0.00,
    0.10,
    0.20,
    0.30,
    0.50,
    0.75,
    1.00,
    1.50,
    np.inf,
]

gap_labels = [
    "0.00-0.10",
    "0.10-0.20",
    "0.20-0.30",
    "0.30-0.50",
    "0.50-0.75",
    "0.75-1.00",
    "1.00-1.50",
    "1.50+",
]

results[
    "XGGapBand"
] = pd.cut(
    results[
        "XGGap"
    ],
    bins=gap_bins,
    labels=gap_labels,
    right=False,
)


gap_summary = (
    results
    .groupby(
        "XGGapBand",
        observed=False,
    )
    .agg(
        Matches=(
            "ActualResult",
            "size",
        ),
        ActualDraws=(
            "IsActualDraw",
            "sum",
        ),
        PredictedDraws=(
            "IsPredictedDraw",
            "sum",
        ),
        MeanHomeXG=(
            "HomeXG",
            "mean",
        ),
        MeanAwayXG=(
            "AwayXG",
            "mean",
        ),
        MeanXGGap=(
            "XGGap",
            "mean",
        ),
        MeanTotalXG=(
            "TotalXG",
            "mean",
        ),
        MeanDrawProbability=(
            "DrawProbability",
            "mean",
        ),
        MeanDrawMarginToTop=(
            "DrawMarginToTop",
            "mean",
        ),
    )
    .reset_index()
)

gap_summary[
    "ActualDrawPct"
] = (
    gap_summary[
        "ActualDraws"
    ]
    /
    gap_summary[
        "Matches"
    ]
    * 100
)

gap_summary[
    "PredictedDrawPct"
] = (
    gap_summary[
        "PredictedDraws"
    ]
    /
    gap_summary[
        "Matches"
    ]
    * 100
)

gap_summary[
    "MeanDrawProbabilityPct"
] = (
    gap_summary[
        "MeanDrawProbability"
    ]
    * 100
)

gap_summary[
    "MeanDrawMarginToTopPct"
] = (
    gap_summary[
        "MeanDrawMarginToTop"
    ]
    * 100
)


# --------------------------------------------------
# Total xG bands
# --------------------------------------------------

total_xg_bins = [
    0.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    np.inf,
]

total_xg_labels = [
    "<1.5",
    "1.5-2.0",
    "2.0-2.5",
    "2.5-3.0",
    "3.0-3.5",
    "3.5-4.0",
    "4.0+",
]

results[
    "TotalXGBand"
] = pd.cut(
    results[
        "TotalXG"
    ],
    bins=total_xg_bins,
    labels=total_xg_labels,
    right=False,
)


total_xg_summary = (
    results
    .groupby(
        "TotalXGBand",
        observed=False,
    )
    .agg(
        Matches=(
            "ActualResult",
            "size",
        ),
        ActualDraws=(
            "IsActualDraw",
            "sum",
        ),
        PredictedDraws=(
            "IsPredictedDraw",
            "sum",
        ),
        MeanTotalXG=(
            "TotalXG",
            "mean",
        ),
        MeanXGGap=(
            "XGGap",
            "mean",
        ),
        MeanDrawProbability=(
            "DrawProbability",
            "mean",
        ),
        MeanDrawMarginToTop=(
            "DrawMarginToTop",
            "mean",
        ),
    )
    .reset_index()
)

total_xg_summary[
    "ActualDrawPct"
] = (
    total_xg_summary[
        "ActualDraws"
    ]
    /
    total_xg_summary[
        "Matches"
    ]
    * 100
)

total_xg_summary[
    "PredictedDrawPct"
] = (
    total_xg_summary[
        "PredictedDraws"
    ]
    /
    total_xg_summary[
        "Matches"
    ]
    * 100
)

total_xg_summary[
    "MeanDrawProbabilityPct"
] = (
    total_xg_summary[
        "MeanDrawProbability"
    ]
    * 100
)

total_xg_summary[
    "MeanDrawMarginToTopPct"
] = (
    total_xg_summary[
        "MeanDrawMarginToTop"
    ]
    * 100
)


# --------------------------------------------------
# Draw probability calibration bands
# --------------------------------------------------

draw_probability_bins = [
    0.00,
    0.15,
    0.18,
    0.20,
    0.22,
    0.24,
    0.26,
    0.28,
    0.30,
    0.35,
    1.01,
]

draw_probability_labels = [
    "<15%",
    "15-18%",
    "18-20%",
    "20-22%",
    "22-24%",
    "24-26%",
    "26-28%",
    "28-30%",
    "30-35%",
    "35%+",
]

results[
    "DrawProbabilityBand"
] = pd.cut(
    results[
        "DrawProbability"
    ],
    bins=draw_probability_bins,
    labels=draw_probability_labels,
    right=False,
)


draw_calibration = (
    results
    .groupby(
        "DrawProbabilityBand",
        observed=False,
    )
    .agg(
        Matches=(
            "ActualResult",
            "size",
        ),
        ActualDraws=(
            "IsActualDraw",
            "sum",
        ),
        PredictedDraws=(
            "IsPredictedDraw",
            "sum",
        ),
        MeanDrawProbability=(
            "DrawProbability",
            "mean",
        ),
        MeanHomeProbability=(
            "HomeProbability",
            "mean",
        ),
        MeanAwayProbability=(
            "AwayProbability",
            "mean",
        ),
        MeanXGGap=(
            "XGGap",
            "mean",
        ),
        MeanTotalXG=(
            "TotalXG",
            "mean",
        ),
        MeanDrawMarginToTop=(
            "DrawMarginToTop",
            "mean",
        ),
    )
    .reset_index()
)

draw_calibration[
    "ActualDrawPct"
] = (
    draw_calibration[
        "ActualDraws"
    ]
    /
    draw_calibration[
        "Matches"
    ]
    * 100
)

draw_calibration[
    "PredictedDrawPct"
] = (
    draw_calibration[
        "PredictedDraws"
    ]
    /
    draw_calibration[
        "Matches"
    ]
    * 100
)

draw_calibration[
    "MeanDrawProbabilityPct"
] = (
    draw_calibration[
        "MeanDrawProbability"
    ]
    * 100
)

draw_calibration[
    "CalibrationGapPct"
] = (
    draw_calibration[
        "ActualDrawPct"
    ]
    -
    draw_calibration[
        "MeanDrawProbabilityPct"
    ]
)

draw_calibration[
    "MeanDrawMarginToTopPct"
] = (
    draw_calibration[
        "MeanDrawMarginToTop"
    ]
    * 100
)


# --------------------------------------------------
# Overall summary
# --------------------------------------------------

matches = len(
    results
)

actual_draws = int(
    results[
        "IsActualDraw"
    ].sum()
)

predicted_draws = int(
    results[
        "IsPredictedDraw"
    ].sum()
)

mean_draw_probability = float(
    results[
        "DrawProbability"
    ].mean()
)

mean_draw_probability_actual_draws = float(
    results.loc[
        results[
            "IsActualDraw"
        ],
        "DrawProbability",
    ].mean()
)

mean_draw_probability_non_draws = float(
    results.loc[
        ~results[
            "IsActualDraw"
        ],
        "DrawProbability",
    ].mean()
)

closest_draw_margin = float(
    results[
        "DrawMarginToTop"
    ].max()
)

closest_draw_match = (
    results.loc[
        results[
            "DrawMarginToTop"
        ].idxmax()
    ]
)

modal_one_one_count = int(
    results[
        "ModalScore"
    ]
    .eq("1-1")
    .sum()
)

summary = pd.DataFrame(
    [
        {
            "Matches":
                matches,

            "ActualDraws":
                actual_draws,

            "ActualDrawPct":
                safe_percentage(
                    actual_draws,
                    matches,
                ),

            "PredictedDraws":
                predicted_draws,

            "PredictedDrawPct":
                safe_percentage(
                    predicted_draws,
                    matches,
                ),

            "MeanDrawProbabilityPct":
                (
                    mean_draw_probability
                    * 100
                ),

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

            "MaximumDrawMarginToTopPct":
                (
                    closest_draw_margin
                    * 100
                ),

            "ModalOneOneCount":
                modal_one_one_count,

            "ModalOneOnePct":
                safe_percentage(
                    modal_one_one_count,
                    matches,
                ),
        }
    ]
)


# --------------------------------------------------
# Console output
# --------------------------------------------------

print()
print("OVERALL DRAW DIAGNOSTIC")
print("=======================")

print(
    summary.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)


print()
print("CLOSEST MODEL 2 CAME TO PREDICTING DRAW")
print("========================================")

print(
    f"{closest_draw_match['Season']} | "
    f"{closest_draw_match['HomeTeam']} vs "
    f"{closest_draw_match['AwayTeam']}"
)

print(
    f"xG: "
    f"{closest_draw_match['HomeXG']:.3f} - "
    f"{closest_draw_match['AwayXG']:.3f}"
)

print(
    f"P(H): "
    f"{closest_draw_match['HomeProbability'] * 100:.2f}%"
)

print(
    f"P(D): "
    f"{closest_draw_match['DrawProbability'] * 100:.2f}%"
)

print(
    f"P(A): "
    f"{closest_draw_match['AwayProbability'] * 100:.2f}%"
)

print(
    f"Draw margin to strongest non-draw outcome: "
    f"{closest_draw_match['DrawMarginToTop'] * 100:.2f} pp"
)

print(
    f"Actual result: "
    f"{closest_draw_match['ActualResult']}"
)


print()
print("DRAW RATE BY XG GAP")
print("===================")

print(
    gap_summary[
        [
            "XGGapBand",
            "Matches",
            "ActualDrawPct",
            "PredictedDrawPct",
            "MeanDrawProbabilityPct",
            "MeanTotalXG",
            "MeanDrawMarginToTopPct",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)


print()
print("DRAW RATE BY TOTAL XG")
print("=====================")

print(
    total_xg_summary[
        [
            "TotalXGBand",
            "Matches",
            "ActualDrawPct",
            "PredictedDrawPct",
            "MeanDrawProbabilityPct",
            "MeanXGGap",
            "MeanDrawMarginToTopPct",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)


print()
print("DRAW PROBABILITY CALIBRATION")
print("============================")

print(
    draw_calibration[
        [
            "DrawProbabilityBand",
            "Matches",
            "MeanDrawProbabilityPct",
            "ActualDrawPct",
            "CalibrationGapPct",
            "PredictedDrawPct",
            "MeanXGGap",
            "MeanTotalXG",
            "MeanDrawMarginToTopPct",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)


# --------------------------------------------------
# Interpretation prompts
# --------------------------------------------------

print()
print("KEY QUESTIONS FOR NEXT MODEL")
print("============================")

print(
    "1. Does actual draw frequency rise as the "
    "home-away xG gap narrows?"
)

print(
    "2. Is Model 2 systematically under- or "
    "over-estimating draw probability in the "
    "highest draw-probability bands?"
)

print(
    "3. How far below Home/Away does Draw remain "
    "even when team xG estimates are very similar?"
)

print(
    "4. Does total expected scoring materially "
    "affect observed draw frequency?"
)

print(
    "5. Is the problem primarily calibration, "
    "xG separation, or the structure of the "
    "independent Poisson probability layer?"
)


# --------------------------------------------------
# Save outputs
# --------------------------------------------------

Path(
    "reports/model3"
).mkdir(
    parents=True,
    exist_ok=True,
)

results.to_csv(
    MATCH_RESULTS_FILE,
    index=False,
)

gap_summary.to_csv(
    XG_GAP_FILE,
    index=False,
)

total_xg_summary.to_csv(
    TOTAL_XG_FILE,
    index=False,
)

draw_calibration.to_csv(
    DRAW_CALIBRATION_FILE,
    index=False,
)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


print()
print("FILES SAVED")
print("===========")

print(
    MATCH_RESULTS_FILE
)

print(
    XG_GAP_FILE
)

print(
    TOTAL_XG_FILE
)

print(
    DRAW_CALIBRATION_FILE
)

print(
    SUMMARY_FILE
)


print()
print("DRAW / XG DIAGNOSTIC COMPLETE")
print("=============================")