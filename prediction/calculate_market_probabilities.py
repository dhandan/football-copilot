from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Files
# --------------------------------------------------

INPUT_FILE = (
    "data/processed/"
    "market_odds.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "market_probabilities.csv"
)


print()
print("CALCULATING MARKET PROBABILITIES")
print("================================")


# --------------------------------------------------
# Load
# --------------------------------------------------

df = pd.read_csv(
    INPUT_FILE
)


# --------------------------------------------------
# Raw implied probabilities
# --------------------------------------------------

df[
    "RawHomeProbability"
] = (
    1
    /
    df["HomeOdds"]
)


df[
    "RawDrawProbability"
] = (
    1
    /
    df["DrawOdds"]
)


df[
    "RawAwayProbability"
] = (
    1
    /
    df["AwayOdds"]
)


# --------------------------------------------------
# Bookmaker overround
# --------------------------------------------------

df[
    "Overround"
] = (
    df[
        "RawHomeProbability"
    ]
    +
    df[
        "RawDrawProbability"
    ]
    +
    df[
        "RawAwayProbability"
    ]
)


df[
    "OverroundPct"
] = (
    (
        df["Overround"]
        -
        1
    )
    *
    100
)


# --------------------------------------------------
# Normalised fair-market probabilities
# --------------------------------------------------

df[
    "MarketHomeProbability"
] = (
    df[
        "RawHomeProbability"
    ]
    /
    df["Overround"]
)


df[
    "MarketDrawProbability"
] = (
    df[
        "RawDrawProbability"
    ]
    /
    df["Overround"]
)


df[
    "MarketAwayProbability"
] = (
    df[
        "RawAwayProbability"
    ]
    /
    df["Overround"]
)


# --------------------------------------------------
# Validation
# --------------------------------------------------

df[
    "FairProbabilityTotal"
] = (
    df[
        "MarketHomeProbability"
    ]
    +
    df[
        "MarketDrawProbability"
    ]
    +
    df[
        "MarketAwayProbability"
    ]
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


df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print(
    f"Matches: {len(df)}"
)


print(
    "Average bookmaker overround: "
    f"{df['OverroundPct'].mean():.2f}%"
)


print(
    "Average normalised probability total: "
    f"{df['FairProbabilityTotal'].mean():.4f}"
)


print(
    f"Saved to: {OUTPUT_FILE}"
)


print()
print("SAMPLE")
print("======")

print(
    df[
        [
            "HomeTeam",
            "AwayTeam",
            "HomeOdds",
            "DrawOdds",
            "AwayOdds",
            "MarketHomeProbability",
            "MarketDrawProbability",
            "MarketAwayProbability",
        ]
    ].head()
)