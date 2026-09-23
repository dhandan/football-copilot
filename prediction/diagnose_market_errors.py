from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# FOOTBALL COPILOT
# OPENING MARKET ERROR DIAGNOSTIC
#
# Diagnostic only.  This script does NOT train or
# tune a new model and does NOT touch the historical
# GW6 deployment holdout.
#
# Population: exact controlled 1,100-match OOT set
# used by Model 6.
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
    "market_error_diagnostic_detail.csv"
)

SEGMENT_OUTPUT = (
    OUTPUT_DIR
    /
    "market_error_diagnostic_segments.csv"
)


OOT_SEASONS = [
    "2023/24",
    "2024/25",
    "2025/26",
]

EXPECTED_COUNTS = {
    "2023/24": 360,
    "2024/25": 370,
    "2025/26": 370,
}

CLASS_ORDER = [
    "H",
    "D",
    "A",
]


def fail(message):
    raise RuntimeError(message)


def add_summary(
    rows,
    frame,
    dimension,
    segment,
):

    if len(frame) == 0:
        return

    rows.append(
        {
            "Dimension":
                dimension,

            "Segment":
                str(segment),

            "Matches":
                len(frame),

            "Correct":
                int(
                    frame[
                        "MarketCorrect"
                    ].sum()
                ),

            "Errors":
                int(
                    (
                        ~frame[
                            "MarketCorrect"
                        ]
                    ).sum()
                ),

            "AccuracyPct":
                100
                *
                frame[
                    "MarketCorrect"
                ].mean(),

            "DrawRatePct":
                100
                *
                (
                    frame[
                        "ActualResult"
                    ]
                    ==
                    "D"
                ).mean(),

            "MeanTopProbability":
                frame[
                    "MarketTopProbability"
                ].mean(),

            "MeanMarketMargin":
                frame[
                    "MarketMargin"
                ].mean(),

            "MeanOver25Probability":
                frame[
                    "MarketOpenOver25Prob"
                ].mean(),

            "MeanAbsXGDDifference":
                frame[
                    "AbsXGDDifference"
                ].mean(),

            "MeanAbsEWXGDDifference":
                frame[
                    "AbsEWXGDDifference"
                ].mean(),
        }
    )


print()
print("FOOTBALL COPILOT")
print("OPENING MARKET ERROR DIAGNOSTIC")
print("===============================")


df = pd.read_csv(
    INPUT_FILE
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


required = [
    "Season",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "ActualResult",
    "MarketOpenHomeProb",
    "MarketOpenDrawProb",
    "MarketOpenAwayProb",
    "MarketOpenOver25Prob",
    "MarketOpenAHLine",
    "XGDDifference",
    "EWXGDDifference",
    "TenMatchPPGDifference",
    "SOTDifference",
]

missing = [
    column
    for column in required
    if column not in df.columns
]

if missing:

    fail(
        "Missing required columns: "
        +
        ", ".join(
            missing
        )
    )


oot = (
    df[
        df[
            "Season"
        ].isin(
            OOT_SEASONS
        )
    ]
    .copy()
)

if len(
    oot
) != 1100:

    fail(
        f"Expected 1100 controlled OOT rows, "
        f"found {len(oot)}"
    )


for season, expected in (
    EXPECTED_COUNTS.items()
):

    actual_count = int(
        (
            oot[
                "Season"
            ]
            ==
            season
        ).sum()
    )

    if actual_count != expected:

        fail(
            f"{season}: expected {expected} rows, "
            f"found {actual_count}"
        )


probabilities = oot[
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
        "Opening 1X2 probabilities contain "
        "missing/non-finite values"
    )


probabilities = (
    probabilities
    /
    probabilities.sum(
        axis=1,
        keepdims=True,
    )
)


sorted_probabilities = np.sort(
    probabilities,
    axis=1,
)


top_indices = np.argmax(
    probabilities,
    axis=1,
)


oot[
    "MarketPick"
] = [
    CLASS_ORDER[index]
    for index in top_indices
]


oot[
    "MarketTopProbability"
] = probabilities[
    np.arange(
        len(oot)
    ),
    top_indices,
]


oot[
    "MarketMargin"
] = (
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


oot[
    "MarketCorrect"
] = (
    oot[
        "MarketPick"
    ]
    ==
    oot[
        "ActualResult"
    ]
)


oot[
    "AbsXGDDifference"
] = np.abs(
    oot[
        "XGDDifference"
    ]
)


oot[
    "AbsEWXGDDifference"
] = np.abs(
    oot[
        "EWXGDDifference"
    ]
)


# Football direction:
# +1 home, 0 neutral, -1 away.
# A small neutral zone avoids treating tiny xG
# differences as meaningful disagreement.

oot[
    "EWXGDirection"
] = np.select(
    [
        oot[
            "EWXGDDifference"
        ]
        >
        0.20,

        oot[
            "EWXGDDifference"
        ]
        <
        -0.20,
    ],
    [
        "H",
        "A",
    ],
    default="Neutral",
)


oot[
    "FootballVsMarket"
] = np.where(
    oot[
        "EWXGDirection"
    ]
    ==
    "Neutral",
    "Neutral",
    np.where(
        oot[
            "EWXGDirection"
        ]
        ==
        oot[
            "MarketPick"
        ],
        "Agree",
        "Disagree",
    ),
)


# ==================================================
# FIXED DIAGNOSTIC BANDS
# ==================================================

oot[
    "TopProbabilityBand"
] = pd.cut(
    oot[
        "MarketTopProbability"
    ],
    bins=[
        0.0,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.70,
        1.01,
    ],
    labels=[
        "<40%",
        "40-45%",
        "45-50%",
        "50-55%",
        "55-60%",
        "60-70%",
        "70%+",
    ],
    include_lowest=True,
    right=False,
)


oot[
    "MarginBand"
] = pd.cut(
    oot[
        "MarketMargin"
    ],
    bins=[
        0.0,
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
        1.01,
    ],
    labels=[
        "<5pp",
        "5-10pp",
        "10-15pp",
        "15-20pp",
        "20-30pp",
        "30pp+",
    ],
    include_lowest=True,
    right=False,
)


oot[
    "Over25Band"
] = pd.cut(
    oot[
        "MarketOpenOver25Prob"
    ],
    bins=[
        0.0,
        0.45,
        0.50,
        0.55,
        0.60,
        1.01,
    ],
    labels=[
        "<45%",
        "45-50%",
        "50-55%",
        "55-60%",
        "60%+",
    ],
    include_lowest=True,
    right=False,
)


oot[
    "AsianHandicapBand"
] = pd.cut(
    oot[
        "MarketOpenAHLine"
    ],
    bins=[
        -10.0,
        -1.25,
        -0.75,
        -0.25,
        0.25,
        0.75,
        1.25,
        10.0,
    ],
    labels=[
        "<=-1.25",
        "-1.0/-0.75",
        "-0.5/-0.25",
        "Level",
        "+0.25/+0.5",
        "+0.75/+1.0",
        ">=+1.25",
    ],
    include_lowest=True,
    right=False,
)


oot[
    "AbsEWXGBand"
] = pd.cut(
    oot[
        "AbsEWXGDDifference"
    ],
    bins=[
        0.0,
        0.20,
        0.40,
        0.60,
        1.00,
        np.inf,
    ],
    labels=[
        "<0.20",
        "0.20-0.40",
        "0.40-0.60",
        "0.60-1.00",
        "1.00+",
    ],
    include_lowest=True,
    right=False,
)


# ==================================================
# SUMMARIES
# ==================================================

summary_rows = []


add_summary(
    summary_rows,
    oot,
    "Overall",
    "All OOT",
)


for season in (
    OOT_SEASONS
):

    add_summary(
        summary_rows,
        oot[
            oot[
                "Season"
            ]
            ==
            season
        ],
        "Season",
        season,
    )


for result in (
    CLASS_ORDER
):

    add_summary(
        summary_rows,
        oot[
            oot[
                "ActualResult"
            ]
            ==
            result
        ],
        "ActualResult",
        result,
    )


for pick in (
    CLASS_ORDER
):

    add_summary(
        summary_rows,
        oot[
            oot[
                "MarketPick"
            ]
            ==
            pick
        ],
        "MarketPick",
        pick,
    )


for column, dimension in [
    (
        "TopProbabilityBand",
        "TopProbability",
    ),
    (
        "MarginBand",
        "MarketMargin",
    ),
    (
        "Over25Band",
        "Over25Probability",
    ),
    (
        "AsianHandicapBand",
        "AsianHandicap",
    ),
    (
        "AbsEWXGBand",
        "AbsEWXGD",
    ),
]:

    for segment in (
        oot[
            column
        ]
        .dropna()
        .unique()
    ):

        add_summary(
            summary_rows,
            oot[
                oot[
                    column
                ]
                ==
                segment
            ],
            dimension,
            segment,
        )


for segment in [
    "Agree",
    "Neutral",
    "Disagree",
]:

    add_summary(
        summary_rows,
        oot[
            oot[
                "FootballVsMarket"
            ]
            ==
            segment
        ],
        "FootballVsMarket",
        segment,
    )


segments = pd.DataFrame(
    summary_rows
)


# ==================================================
# PRINT
# ==================================================

overall_accuracy = (
    oot[
        "MarketCorrect"
    ].mean()
)


print(
    f"Controlled OOT rows: {len(oot)}"
)

print(
    f"Correct: "
    f"{int(oot['MarketCorrect'].sum())}"
)

print(
    f"Errors: "
    f"{int((~oot['MarketCorrect']).sum())}"
)

print(
    f"Accuracy: "
    f"{overall_accuracy * 100:.4f}%"
)


def print_dimension(
    dimension,
):

    view = (
        segments[
            segments[
                "Dimension"
            ]
            ==
            dimension
        ]
        .copy()
    )

    print()
    print(
        dimension.upper()
    )

    print(
        "="
        *
        len(
            dimension
        )
    )

    print(
        view[
            [
                "Segment",
                "Matches",
                "Correct",
                "Errors",
                "AccuracyPct",
                "DrawRatePct",
                "MeanTopProbability",
                "MeanMarketMargin",
            ]
        ].to_string(
            index=False,
            formatters={
                "AccuracyPct":
                    lambda x:
                        f"{x:.2f}",

                "DrawRatePct":
                    lambda x:
                        f"{x:.2f}",

                "MeanTopProbability":
                    lambda x:
                        f"{x:.3f}",

                "MeanMarketMargin":
                    lambda x:
                        f"{x:.3f}",
            },
        )
    )


for dimension in [
    "Season",
    "ActualResult",
    "MarketPick",
    "TopProbability",
    "MarketMargin",
    "Over25Probability",
    "AsianHandicap",
    "AbsEWXGD",
    "FootballVsMarket",
]:

    print_dimension(
        dimension
    )


# ==================================================
# SAVE
# ==================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


detail_columns = [
    "Season",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "ActualResult",
    "MarketPick",
    "MarketCorrect",
    "MarketOpenHomeProb",
    "MarketOpenDrawProb",
    "MarketOpenAwayProb",
    "MarketTopProbability",
    "MarketMargin",
    "TopProbabilityBand",
    "MarginBand",
    "MarketOpenOver25Prob",
    "Over25Band",
    "MarketOpenAHLine",
    "AsianHandicapBand",
    "XGDDifference",
    "EWXGDDifference",
    "AbsEWXGDDifference",
    "AbsEWXGBand",
    "EWXGDirection",
    "FootballVsMarket",
    "TenMatchPPGDifference",
    "SOTDifference",
]


oot[
    detail_columns
].to_csv(
    DETAIL_OUTPUT,
    index=False,
)


segments.to_csv(
    SEGMENT_OUTPUT,
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
    SEGMENT_OUTPUT,
)

print()
print(
    "OPENING MARKET ERROR DIAGNOSTIC COMPLETE"
)
print(
    "========================================"
)