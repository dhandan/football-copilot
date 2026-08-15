from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Files
# --------------------------------------------------

MODEL_FILE = (
    "data/processed/"
    "model_comparison_results.csv"
)

MARKET_FILE = (
    "data/processed/"
    "market_probabilities.csv"
)

DETAIL_OUTPUT = (
    "data/processed/"
    "model_vs_market_results.csv"
)

SUMMARY_OUTPUT = (
    "data/processed/"
    "model_vs_market_summary.csv"
)


print()
print("MODEL 2 VS MARKET")
print("=================")


# --------------------------------------------------
# Load Model 2 walk-forward predictions
# --------------------------------------------------

model = pd.read_csv(
    MODEL_FILE
)


model["Date"] = pd.to_datetime(
    model["Date"],
    errors="coerce",
)


# --------------------------------------------------
# Load historical market data
# --------------------------------------------------

market = pd.read_csv(
    MARKET_FILE
)


market["Date"] = pd.to_datetime(
    market["Date"],
    errors="coerce",
)


# --------------------------------------------------
# Join model and market
# --------------------------------------------------

data = model.merge(
    market[
        [
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",

            "HomeOdds",
            "DrawOdds",
            "AwayOdds",

            "MarketHomeProbability",
            "MarketDrawProbability",
            "MarketAwayProbability",

            "OddsSource",
        ]
    ],

    on=[
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
    ],

    how="inner",
)


print()
print(
    f"Matched fixtures: {len(data)}"
)


if len(data) == 0:

    raise ValueError(
        "No model predictions matched "
        "the bookmaker dataset."
    )


# --------------------------------------------------
# Model edges
#
# Positive edge means the model probability is
# greater than the fair-market probability.
# --------------------------------------------------

data[
    "HomeEdge"
] = (
    data[
        "M2HomeProbability"
    ]
    -
    data[
        "MarketHomeProbability"
    ]
)


data[
    "DrawEdge"
] = (
    data[
        "M2DrawProbability"
    ]
    -
    data[
        "MarketDrawProbability"
    ]
)


data[
    "AwayEdge"
] = (
    data[
        "M2AwayProbability"
    ]
    -
    data[
        "MarketAwayProbability"
    ]
)


# --------------------------------------------------
# Find largest model edge for each match
# --------------------------------------------------

def identify_best_edge(row):

    choices = {
        "H":
            row[
                "HomeEdge"
            ],

        "D":
            row[
                "DrawEdge"
            ],

        "A":
            row[
                "AwayEdge"
            ],
    }


    selection = max(
        choices,
        key=choices.get,
    )


    return pd.Series(
        {
            "ValueSelection":
                selection,

            "ValueEdge":
                choices[
                    selection
                ],
        }
    )


best_edges = data.apply(
    identify_best_edge,
    axis=1,
)


data = pd.concat(
    [
        data,
        best_edges,
    ],
    axis=1,
)


# --------------------------------------------------
# Selected decimal odds
# --------------------------------------------------

def selected_odds(row):

    if (
        row[
            "ValueSelection"
        ]
        ==
        "H"
    ):

        return row[
            "HomeOdds"
        ]


    if (
        row[
            "ValueSelection"
        ]
        ==
        "D"
    ):

        return row[
            "DrawOdds"
        ]


    return row[
        "AwayOdds"
    ]


data[
    "SelectedOdds"
] = data.apply(
    selected_odds,
    axis=1,
)


# --------------------------------------------------
# Whether selected outcome actually happened
# --------------------------------------------------

data[
    "SelectionWon"
] = (
    data[
        "ValueSelection"
    ]
    ==
    data[
        "ActualResult"
    ]
)


# --------------------------------------------------
# Hypothetical £1 return
#
# Win:
# return = decimal odds
#
# Loss:
# return = 0
#
# Profit is return minus £1 stake.
# --------------------------------------------------

data[
    "GrossReturn"
] = np.where(
    data[
        "SelectionWon"
    ],

    data[
        "SelectedOdds"
    ],

    0.0,
)


data[
    "Profit"
] = (
    data[
        "GrossReturn"
    ]
    -
    1.0
)


# --------------------------------------------------
# Test edge thresholds
# --------------------------------------------------

THRESHOLDS = [
    0.00,
    0.02,
    0.05,
    0.075,
    0.10,
]


summary_rows = []


for threshold in THRESHOLDS:

    selections = data[
        data[
            "ValueEdge"
        ]
        >= threshold
    ].copy()


    number_of_selections = len(
        selections
    )


    if number_of_selections == 0:

        continue


    winners = (
        selections[
            "SelectionWon"
        ].sum()
    )


    strike_rate = (
        winners
        /
        number_of_selections
    )


    total_staked = float(
        number_of_selections
    )


    total_return = (
        selections[
            "GrossReturn"
        ].sum()
    )


    profit = (
        total_return
        -
        total_staked
    )


    roi = (
        profit
        /
        total_staked
    )


    average_edge = (
        selections[
            "ValueEdge"
        ].mean()
    )


    average_odds = (
        selections[
            "SelectedOdds"
        ].mean()
    )


    summary_rows.append(
        {
            "MinimumEdge":
                threshold,

            "MinimumEdgePct":
                threshold * 100,

            "Selections":
                number_of_selections,

            "Winners":
                winners,

            "StrikeRatePct":
                strike_rate * 100,

            "AverageEdgePct":
                average_edge * 100,

            "AverageOdds":
                average_odds,

            "TotalStaked":
                total_staked,

            "TotalReturn":
                total_return,

            "Profit":
                profit,

            "ROI":
                roi,

            "ROIPct":
                roi * 100,
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# --------------------------------------------------
# Save
# --------------------------------------------------

Path(
    "data/processed"
).mkdir(
    parents=True,
    exist_ok=True,
)


data.to_csv(
    DETAIL_OUTPUT,
    index=False,
)


summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print()
print("EDGE BACKTEST")
print("=============")


print(
    summary[
        [
            "MinimumEdgePct",
            "Selections",
            "StrikeRatePct",
            "AverageEdgePct",
            "AverageOdds",
            "Profit",
            "ROIPct",
        ]
    ].to_string(
        index=False,
        formatters={

            "MinimumEdgePct":
                "{:.1f}".format,

            "StrikeRatePct":
                "{:.1f}".format,

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
# Edge distribution
# --------------------------------------------------

print()
print("EDGE DISTRIBUTION")
print("=================")


print(
    data[
        "ValueEdge"
    ].describe()
)


# --------------------------------------------------
# Most extreme historical model disagreements
# --------------------------------------------------

print()
print("LARGEST MODEL VS MARKET EDGES")
print("=============================")


largest_edges = (
    data.sort_values(
        "ValueEdge",
        ascending=False,
    )
    .head(10)
)


print(
    largest_edges[
        [
            "Season",
            "HomeTeam",
            "AwayTeam",
            "ValueSelection",
            "ValueEdge",
            "SelectedOdds",
            "ActualResult",
            "SelectionWon",
        ]
    ].to_string(
        index=False
    )
)


print()
print(
    f"Detailed results saved to: "
    f"{DETAIL_OUTPUT}"
)

print(
    f"Summary saved to: "
    f"{SUMMARY_OUTPUT}"
)