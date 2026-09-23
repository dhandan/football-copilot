from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Files
# --------------------------------------------------

INPUT_FILE = (
    "reports/post_gw5/"
    "h3_underlying_form_predictions.csv"
)

OUTPUT_FILE = (
    "reports/post_gw5/"
    "h6_error_segments.csv"
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

LABEL_MAP = {
    "H": 0,
    "D": 1,
    "A": 2,
}


def segment_metrics(
    data,
    segment_type,
    segment,
):

    if len(data) == 0:
        return None

    probabilities = data[
        [
            "M2HomeProbability",
            "M2DrawProbability",
            "M2AwayProbability",
        ]
    ].to_numpy()

    actual_numeric = (
        data["ActualResult"]
        .map(LABEL_MAP)
        .to_numpy()
    )

    predicted = data[
        "M2PredictedResult"
    ].to_numpy()

    actual = data[
        "ActualResult"
    ].to_numpy()

    accuracy = np.mean(
        predicted == actual
    )

    selected_probabilities = (
        probabilities[
            np.arange(len(data)),
            actual_numeric,
        ]
    )

    log_loss = -np.mean(
        np.log(
            np.clip(
                selected_probabilities,
                1e-15,
                1.0,
            )
        )
    )

    actual_matrix = np.zeros(
        (
            len(data),
            3,
        )
    )

    actual_matrix[
        np.arange(len(data)),
        actual_numeric,
    ] = 1.0

    brier = np.mean(
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

    actual_draws = int(
        (
            data["ActualResult"]
            ==
            "D"
        ).sum()
    )

    predicted_draws = int(
        (
            data["M2PredictedResult"]
            ==
            "D"
        ).sum()
    )

    return {
        "SegmentType":
            segment_type,

        "Segment":
            str(segment),

        "Matches":
            len(data),

        "AccuracyPct":
            accuracy * 100,

        "LogLoss":
            log_loss,

        "Brier":
            brier,

        "ActualDraws":
            actual_draws,

        "ActualDrawPct":
            (
                actual_draws
                /
                len(data)
                *
                100
            ),

        "PredictedDraws":
            predicted_draws,

        "MeanHomeProbabilityPct":
            data[
                "M2HomeProbability"
            ].mean()
            * 100,

        "MeanDrawProbabilityPct":
            data[
                "M2DrawProbability"
            ].mean()
            * 100,

        "MeanAwayProbabilityPct":
            data[
                "M2AwayProbability"
            ].mean()
            * 100,

        "MeanHomeXG":
            data[
                "M2HomeXG"
            ].mean(),

        "MeanAwayXG":
            data[
                "M2AwayXG"
            ].mean(),
    }


def add_segment(
    rows,
    data,
    segment_type,
    column,
):

    for segment in (
        data[column]
        .dropna()
        .unique()
    ):

        subset = data[
            data[column]
            ==
            segment
        ].copy()

        metrics = segment_metrics(
            subset,
            segment_type,
            segment,
        )

        if metrics is not None:
            rows.append(
                metrics
            )


# --------------------------------------------------
# Load
# --------------------------------------------------

print()
print("FOOTBALL COPILOT")
print("H6 ERROR SEGMENT DIAGNOSTIC")
print("===========================")

df = pd.read_csv(
    INPUT_FILE
)


# --------------------------------------------------
# Derived diagnostics
# --------------------------------------------------

df["M2TopProbability"] = df[
    [
        "M2HomeProbability",
        "M2DrawProbability",
        "M2AwayProbability",
    ]
].max(
    axis=1
)


df["M2XGGap"] = (
    df["M2HomeXG"]
    -
    df["M2AwayXG"]
).abs()


df["M2TotalXG"] = (
    df["M2HomeXG"]
    +
    df["M2AwayXG"]
)


# Difference between strongest and second strongest
# 1X2 probability.

sorted_probabilities = np.sort(
    df[
        [
            "M2HomeProbability",
            "M2DrawProbability",
            "M2AwayProbability",
        ]
    ].to_numpy(),
    axis=1,
)

df["M2DecisionMargin"] = (
    sorted_probabilities[:, 2]
    -
    sorted_probabilities[:, 1]
)


# --------------------------------------------------
# Segment bands
# --------------------------------------------------

df["ConfidenceBand"] = pd.cut(
    df["M2TopProbability"],
    bins=[
        0.0,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.70,
        1.0,
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
)


df["DecisionMarginBand"] = pd.cut(
    df["M2DecisionMargin"],
    bins=[
        -0.001,
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
        1.0,
    ],
    labels=[
        "0-5pp",
        "5-10pp",
        "10-15pp",
        "15-20pp",
        "20-30pp",
        "30pp+",
    ],
)


df["XGGapBand"] = pd.cut(
    df["M2XGGap"],
    bins=[
        -0.001,
        0.10,
        0.20,
        0.30,
        0.50,
        0.75,
        1.00,
        1.50,
        np.inf,
    ],
    labels=[
        "0.00-0.10",
        "0.10-0.20",
        "0.20-0.30",
        "0.30-0.50",
        "0.50-0.75",
        "0.75-1.00",
        "1.00-1.50",
        "1.50+",
    ],
)


df["TotalXGBand"] = pd.cut(
    df["M2TotalXG"],
    bins=[
        0.0,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        np.inf,
    ],
    labels=[
        "<2.0",
        "2.0-2.5",
        "2.5-3.0",
        "3.0-3.5",
        "3.5-4.0",
        "4.0+",
    ],
    include_lowest=True,
)


# --------------------------------------------------
# Calculate segments
# --------------------------------------------------

rows = []


overall = segment_metrics(
    df,
    "Overall",
    "All matches",
)

rows.append(
    overall
)


add_segment(
    rows,
    df,
    "Season",
    "Season",
)


add_segment(
    rows,
    df,
    "PredictedResult",
    "M2PredictedResult",
)


add_segment(
    rows,
    df,
    "ActualResult",
    "ActualResult",
)


add_segment(
    rows,
    df,
    "Confidence",
    "ConfidenceBand",
)


add_segment(
    rows,
    df,
    "DecisionMargin",
    "DecisionMarginBand",
)


add_segment(
    rows,
    df,
    "XGGap",
    "XGGapBand",
)


add_segment(
    rows,
    df,
    "TotalXG",
    "TotalXGBand",
)


results = pd.DataFrame(
    rows
)


# --------------------------------------------------
# Output
# --------------------------------------------------

print()
print("OVERALL")
print("=======")

print(
    results[
        results["SegmentType"]
        ==
        "Overall"
    ].to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


for segment_type in [
    "PredictedResult",
    "Confidence",
    "DecisionMargin",
    "XGGap",
    "TotalXG",
    "Season",
]:

    print()
    print(
        segment_type.upper()
    )
    print(
        "=" * len(segment_type)
    )

    display = results[
        results["SegmentType"]
        ==
        segment_type
    ].copy()

    print(
        display[
            [
                "Segment",
                "Matches",
                "AccuracyPct",
                "LogLoss",
                "Brier",
                "ActualDrawPct",
                "PredictedDraws",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
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
    OUTPUT_FILE,
    index=False,
)


print()
print("FILE SAVED")
print("==========")

print(
    OUTPUT_FILE
)


print()
print("H6 ERROR SEGMENT DIAGNOSTIC COMPLETE")
print("====================================")