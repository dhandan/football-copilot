from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


# --------------------------------------------------
# Files
# --------------------------------------------------

INPUT_FILE = (
    "data/processed/"
    "model_vs_market_results.csv"
)

DETAIL_OUTPUT = (
    "data/processed/"
    "market_model_diagnostics.csv"
)

SEASON_OUTPUT = (
    "data/processed/"
    "market_model_by_season.csv"
)

ODDS_OUTPUT = (
    "data/processed/"
    "market_model_by_odds.csv"
)

OUTCOME_OUTPUT = (
    "data/processed/"
    "market_model_by_outcome.csv"
)

CALIBRATION_OUTPUT = (
    "data/processed/"
    "market_model_calibration.csv"
)


print()
print("MARKET VS MODEL 2 DIAGNOSTICS")
print("=============================")


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(
    INPUT_FILE
)


df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce",
)


print()
print(
    f"Fixtures analysed: {len(df)}"
)


# --------------------------------------------------
# Labels
# --------------------------------------------------

LABEL_MAP = {
    "H": 0,
    "D": 1,
    "A": 2,
}


OUTCOME_NAMES = {
    "H": "Home",
    "D": "Draw",
    "A": "Away",
}


y_true = (
    df["ActualResult"]
    .map(LABEL_MAP)
)


# --------------------------------------------------
# Probability arrays
# --------------------------------------------------

model_probabilities = df[
    [
        "M2HomeProbability",
        "M2DrawProbability",
        "M2AwayProbability",
    ]
].to_numpy()


market_probabilities = df[
    [
        "MarketHomeProbability",
        "MarketDrawProbability",
        "MarketAwayProbability",
    ]
].to_numpy()


# --------------------------------------------------
# Multiclass Brier score
# --------------------------------------------------

def multiclass_brier(
    actual_results,
    probabilities,
):

    actual_matrix = np.zeros(
        (
            len(actual_results),
            3,
        )
    )


    for row_number, result in enumerate(
        actual_results
    ):

        actual_matrix[
            row_number,
            LABEL_MAP[result]
        ] = 1


    return np.mean(
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


# --------------------------------------------------
# Overall metrics
# --------------------------------------------------

model_log_loss = log_loss(
    y_true,
    model_probabilities,
    labels=[0, 1, 2],
)


market_log_loss = log_loss(
    y_true,
    market_probabilities,
    labels=[0, 1, 2],
)


model_brier = multiclass_brier(
    df["ActualResult"],
    model_probabilities,
)


market_brier = multiclass_brier(
    df["ActualResult"],
    market_probabilities,
)


# --------------------------------------------------
# Accuracy
# --------------------------------------------------

def highest_probability_result(
    probabilities,
):

    result_codes = np.array(
        ["H", "D", "A"]
    )

    return result_codes[
        np.argmax(
            probabilities,
            axis=1,
        )
    ]


model_predictions = (
    highest_probability_result(
        model_probabilities
    )
)


market_predictions = (
    highest_probability_result(
        market_probabilities
    )
)


model_accuracy = np.mean(
    model_predictions
    ==
    df["ActualResult"].to_numpy()
)


market_accuracy = np.mean(
    market_predictions
    ==
    df["ActualResult"].to_numpy()
)


# --------------------------------------------------
# Print overall comparison
# --------------------------------------------------

print()
print("OVERALL PROBABILITY PERFORMANCE")
print("===============================")


overall = pd.DataFrame(
    [
        {
            "Source": "Model 2",
            "AccuracyPct":
                model_accuracy * 100,
            "LogLoss":
                model_log_loss,
            "Brier":
                model_brier,
        },
        {
            "Source": "Market",
            "AccuracyPct":
                market_accuracy * 100,
            "LogLoss":
                market_log_loss,
            "Brier":
                market_brier,
        },
    ]
)


print(
    overall.to_string(
        index=False,
        formatters={
            "AccuracyPct":
                "{:.2f}".format,

            "LogLoss":
                "{:.4f}".format,

            "Brier":
                "{:.4f}".format,
        },
    )
)


print()
print("MARKET ADVANTAGE")
print("================")


print(
    "Accuracy difference: "
    f"{(
        market_accuracy
        -
        model_accuracy
    ) * 100:+.2f} percentage points"
)


print(
    "Log loss difference: "
    f"{(
        market_log_loss
        -
        model_log_loss
    ):+.4f}"
)


print(
    "Brier difference: "
    f"{(
        market_brier
        -
        model_brier
    ):+.4f}"
)


# --------------------------------------------------
# Generic subset metric function
# --------------------------------------------------

def calculate_subset_metrics(
    subset,
):

    if len(subset) == 0:

        return None


    actual_numeric = (
        subset[
            "ActualResult"
        ].map(
            LABEL_MAP
        )
    )


    model_probs = subset[
        [
            "M2HomeProbability",
            "M2DrawProbability",
            "M2AwayProbability",
        ]
    ].to_numpy()


    market_probs = subset[
        [
            "MarketHomeProbability",
            "MarketDrawProbability",
            "MarketAwayProbability",
        ]
    ].to_numpy()


    model_preds = (
        highest_probability_result(
            model_probs
        )
    )


    market_preds = (
        highest_probability_result(
            market_probs
        )
    )


    return {

        "Matches":
            len(subset),

        "ModelAccuracy":
            np.mean(
                model_preds
                ==
                subset[
                    "ActualResult"
                ].to_numpy()
            )
            * 100,

        "MarketAccuracy":
            np.mean(
                market_preds
                ==
                subset[
                    "ActualResult"
                ].to_numpy()
            )
            * 100,

        "ModelLogLoss":
            log_loss(
                actual_numeric,
                model_probs,
                labels=[0, 1, 2],
            ),

        "MarketLogLoss":
            log_loss(
                actual_numeric,
                market_probs,
                labels=[0, 1, 2],
            ),

        "ModelBrier":
            multiclass_brier(
                subset[
                    "ActualResult"
                ],
                model_probs,
            ),

        "MarketBrier":
            multiclass_brier(
                subset[
                    "ActualResult"
                ],
                market_probs,
            ),
    }


# --------------------------------------------------
# Performance by season
# --------------------------------------------------

season_rows = []


for season in sorted(
    df["Season"].unique()
):

    subset = df[
        df["Season"] == season
    ].copy()


    metrics = (
        calculate_subset_metrics(
            subset
        )
    )


    season_rows.append(
        {
            "Season":
                season,

            **metrics,
        }
    )


season_results = pd.DataFrame(
    season_rows
)


print()
print("PERFORMANCE BY SEASON")
print("=====================")


print(
    season_results.to_string(
        index=False,
        formatters={
            "ModelAccuracy":
                "{:.2f}".format,

            "MarketAccuracy":
                "{:.2f}".format,

            "ModelLogLoss":
                "{:.4f}".format,

            "MarketLogLoss":
                "{:.4f}".format,

            "ModelBrier":
                "{:.4f}".format,

            "MarketBrier":
                "{:.4f}".format,
        },
    )
)


# --------------------------------------------------
# Performance by actual result
# --------------------------------------------------

outcome_rows = []


for result_code in [
    "H",
    "D",
    "A",
]:

    subset = df[
        df[
            "ActualResult"
        ]
        ==
        result_code
    ].copy()


    metrics = (
        calculate_subset_metrics(
            subset
        )
    )


    outcome_rows.append(
        {
            "Outcome":
                OUTCOME_NAMES[
                    result_code
                ],

            **metrics,
        }
    )


outcome_results = pd.DataFrame(
    outcome_rows
)


print()
print("PERFORMANCE BY ACTUAL OUTCOME")
print("=============================")


print(
    outcome_results.to_string(
        index=False,
        formatters={
            "ModelAccuracy":
                "{:.2f}".format,

            "MarketAccuracy":
                "{:.2f}".format,

            "ModelLogLoss":
                "{:.4f}".format,

            "MarketLogLoss":
                "{:.4f}".format,

            "ModelBrier":
                "{:.4f}".format,

            "MarketBrier":
                "{:.4f}".format,
        },
    )
)


# --------------------------------------------------
# Selected odds
#
# For diagnostic purposes we look at the outcome
# where Model 2 had its largest positive edge.
# --------------------------------------------------

df[
    "OddsBand"
] = pd.cut(
    df[
        "SelectedOdds"
    ],

    bins=[
        1.0,
        2.0,
        3.0,
        5.0,
        np.inf,
    ],

    labels=[
        "1.01-2.00",
        "2.01-3.00",
        "3.01-5.00",
        "5.01+",
    ],

    include_lowest=True,
)


odds_rows = []


for band in df[
    "OddsBand"
].dropna().unique():

    subset = df[
        df[
            "OddsBand"
        ]
        ==
        band
    ].copy()


    selections = len(
        subset
    )


    wins = (
        subset[
            "SelectionWon"
        ].sum()
    )


    strike_rate = (
        wins
        /
        selections
        *
        100
    )


    total_profit = (
        subset[
            "Profit"
        ].sum()
    )


    roi = (
        total_profit
        /
        selections
        *
        100
    )


    odds_rows.append(
        {
            "OddsBand":
                str(band),

            "Selections":
                selections,

            "Wins":
                wins,

            "StrikeRatePct":
                strike_rate,

            "AverageOdds":
                subset[
                    "SelectedOdds"
                ].mean(),

            "AverageEdgePct":
                subset[
                    "ValueEdge"
                ].mean()
                * 100,

            "Profit":
                total_profit,

            "ROIPct":
                roi,
        }
    )


odds_results = pd.DataFrame(
    odds_rows
)


print()
print("VALUE SELECTIONS BY ODDS BAND")
print("=============================")


print(
    odds_results.to_string(
        index=False,
        formatters={
            "StrikeRatePct":
                "{:.2f}".format,

            "AverageOdds":
                "{:.2f}".format,

            "AverageEdgePct":
                "{:.2f}".format,

            "Profit":
                "{:.2f}".format,

            "ROIPct":
                "{:.2f}".format,
        },
    )
)


# --------------------------------------------------
# Model edge by selection type
# --------------------------------------------------

selection_rows = []


for result_code in [
    "H",
    "D",
    "A",
]:

    subset = df[
        df[
            "ValueSelection"
        ]
        ==
        result_code
    ].copy()


    if len(subset) == 0:
        continue


    wins = (
        subset[
            "SelectionWon"
        ].sum()
    )


    profit = (
        subset[
            "Profit"
        ].sum()
    )


    selection_rows.append(
        {
            "Selection":
                OUTCOME_NAMES[
                    result_code
                ],

            "Selections":
                len(subset),

            "Wins":
                wins,

            "StrikeRatePct":
                wins
                /
                len(subset)
                *
                100,

            "AverageEdgePct":
                subset[
                    "ValueEdge"
                ].mean()
                *
                100,

            "AverageOdds":
                subset[
                    "SelectedOdds"
                ].mean(),

            "Profit":
                profit,

            "ROIPct":
                profit
                /
                len(subset)
                *
                100,
        }
    )


selection_results = pd.DataFrame(
    selection_rows
)


print()
print("MODEL EDGE BY SELECTION TYPE")
print("============================")


print(
    selection_results.to_string(
        index=False,
        formatters={
            "StrikeRatePct":
                "{:.2f}".format,

            "AverageEdgePct":
                "{:.2f}".format,

            "AverageOdds":
                "{:.2f}".format,

            "Profit":
                "{:.2f}".format,

            "ROIPct":
                "{:.2f}".format,
        },
    )
)


# --------------------------------------------------
# Calibration
#
# Treat each H/D/A probability as a separate
# probability forecast.
# --------------------------------------------------

calibration_records = []


for _, row in df.iterrows():

    for (
        result_code,
        model_column,
        market_column,
    ) in [

        (
            "H",
            "M2HomeProbability",
            "MarketHomeProbability",
        ),

        (
            "D",
            "M2DrawProbability",
            "MarketDrawProbability",
        ),

        (
            "A",
            "M2AwayProbability",
            "MarketAwayProbability",
        ),
    ]:

        calibration_records.append(
            {
                "Outcome":
                    OUTCOME_NAMES[
                        result_code
                    ],

                "ModelProbability":
                    row[
                        model_column
                    ],

                "MarketProbability":
                    row[
                        market_column
                    ],

                "Occurred":
                    (
                        1
                        if row[
                            "ActualResult"
                        ]
                        ==
                        result_code
                        else 0
                    ),
            }
        )


calibration = pd.DataFrame(
    calibration_records
)


CALIBRATION_BINS = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]


calibration[
    "ModelBucket"
] = pd.cut(
    calibration[
        "ModelProbability"
    ],

    bins=CALIBRATION_BINS,

    include_lowest=True,
)


calibration[
    "MarketBucket"
] = pd.cut(
    calibration[
        "MarketProbability"
    ],

    bins=CALIBRATION_BINS,

    include_lowest=True,
)


# --------------------------------------------------
# Model calibration
# --------------------------------------------------

model_calibration = (
    calibration
    .groupby(
        "ModelBucket",
        observed=False,
    )
    .agg(
        Forecasts=(
            "Occurred",
            "count",
        ),

        AverageProbability=(
            "ModelProbability",
            "mean",
        ),

        ActualRate=(
            "Occurred",
            "mean",
        ),
    )
    .reset_index()
)


model_calibration[
    "Source"
] = "Model 2"


model_calibration = (
    model_calibration.rename(
        columns={
            "ModelBucket":
                "ProbabilityBucket"
        }
    )
)


# --------------------------------------------------
# Market calibration
# --------------------------------------------------

market_calibration = (
    calibration
    .groupby(
        "MarketBucket",
        observed=False,
    )
    .agg(
        Forecasts=(
            "Occurred",
            "count",
        ),

        AverageProbability=(
            "MarketProbability",
            "mean",
        ),

        ActualRate=(
            "Occurred",
            "mean",
        ),
    )
    .reset_index()
)


market_calibration[
    "Source"
] = "Market"


market_calibration = (
    market_calibration.rename(
        columns={
            "MarketBucket":
                "ProbabilityBucket"
        }
    )
)


combined_calibration = pd.concat(
    [
        model_calibration,
        market_calibration,
    ],

    ignore_index=True,
)


combined_calibration[
    "AverageProbabilityPct"
] = (
    combined_calibration[
        "AverageProbability"
    ]
    *
    100
)


combined_calibration[
    "ActualRatePct"
] = (
    combined_calibration[
        "ActualRate"
    ]
    *
    100
)


combined_calibration[
    "CalibrationErrorPct"
] = (
    combined_calibration[
        "ActualRatePct"
    ]
    -
    combined_calibration[
        "AverageProbabilityPct"
    ]
)


print()
print("CALIBRATION")
print("===========")


print(
    combined_calibration[
        [
            "Source",
            "ProbabilityBucket",
            "Forecasts",
            "AverageProbabilityPct",
            "ActualRatePct",
            "CalibrationErrorPct",
        ]
    ].to_string(
        index=False,
        formatters={
            "AverageProbabilityPct":
                "{:.2f}".format,

            "ActualRatePct":
                "{:.2f}".format,

            "CalibrationErrorPct":
                "{:+.2f}".format,
        },
    )
)


# --------------------------------------------------
# Where do model and market disagree most?
# --------------------------------------------------

df[
    "MaximumAbsoluteDisagreement"
] = (
    df[
        [
            "HomeEdge",
            "DrawEdge",
            "AwayEdge",
        ]
    ]
    .abs()
    .max(
        axis=1
    )
)


largest_disagreements = (
    df.sort_values(
        "MaximumAbsoluteDisagreement",
        ascending=False,
    )
    .head(15)
)


print()
print("LARGEST MODEL / MARKET DISAGREEMENTS")
print("====================================")


print(
    largest_disagreements[
        [
            "Season",
            "HomeTeam",
            "AwayTeam",
            "ActualResult",
            "M2HomeProbability",
            "MarketHomeProbability",
            "M2DrawProbability",
            "MarketDrawProbability",
            "M2AwayProbability",
            "MarketAwayProbability",
        ]
    ].to_string(
        index=False
    )
)


# --------------------------------------------------
# Save diagnostics
# --------------------------------------------------

Path(
    "data/processed"
).mkdir(
    parents=True,
    exist_ok=True,
)


df.to_csv(
    DETAIL_OUTPUT,
    index=False,
)


season_results.to_csv(
    SEASON_OUTPUT,
    index=False,
)


odds_results.to_csv(
    ODDS_OUTPUT,
    index=False,
)


outcome_results.to_csv(
    OUTCOME_OUTPUT,
    index=False,
)


combined_calibration.to_csv(
    CALIBRATION_OUTPUT,
    index=False,
)


print()
print("FILES SAVED")
print("===========")


print(
    DETAIL_OUTPUT
)

print(
    SEASON_OUTPUT
)

print(
    ODDS_OUTPUT
)

print(
    OUTCOME_OUTPUT
)

print(
    CALIBRATION_OUTPUT
)


print()
print("DIAGNOSTIC ANALYSIS COMPLETE")
print("============================")