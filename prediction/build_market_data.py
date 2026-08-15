from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Files
# --------------------------------------------------

INPUT_FILE = (
    "data/processed/"
    "premier_league_matches.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "market_odds.csv"
)


print()
print("BUILDING MARKET ODDS DATA")
print("=========================")


# --------------------------------------------------
# Load original combined dataset
# --------------------------------------------------

df = pd.read_csv(
    INPUT_FILE
)


# --------------------------------------------------
# Convert date
# --------------------------------------------------

df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True,
    errors="coerce",
)


# --------------------------------------------------
# Preferred odds sources
#
# We prefer average market odds if available.
# If not, fall back to Bet365.
# --------------------------------------------------

ODDS_OPTIONS = [
    (
        "AvgH",
        "AvgD",
        "AvgA",
        "Average market odds",
    ),

    (
        "B365H",
        "B365D",
        "B365A",
        "Bet365",
    ),

    (
        "PSH",
        "PSD",
        "PSA",
        "Pinnacle",
    ),
]


selected_columns = None
selected_source = None


for (
    home_column,
    draw_column,
    away_column,
    source_name,
) in ODDS_OPTIONS:

    required = [
        home_column,
        draw_column,
        away_column,
    ]

    if all(
        column in df.columns
        for column in required
    ):

        selected_columns = required
        selected_source = source_name

        break


if selected_columns is None:

    print()
    print(
        "No recognised 1X2 bookmaker odds "
        "columns were found."
    )

    print()
    print("Available columns:")
    print(df.columns.tolist())

    raise ValueError(
        "Unable to identify bookmaker odds."
    )


print()
print(
    f"Selected odds source: "
    f"{selected_source}"
)

print(
    f"Columns: "
    f"{selected_columns}"
)


# --------------------------------------------------
# Build clean market dataset
# --------------------------------------------------

home_odds_column = (
    selected_columns[0]
)

draw_odds_column = (
    selected_columns[1]
)

away_odds_column = (
    selected_columns[2]
)


market = df[
    [
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        home_odds_column,
        draw_odds_column,
        away_odds_column,
    ]
].copy()


market = market.rename(
    columns={
        home_odds_column:
            "HomeOdds",

        draw_odds_column:
            "DrawOdds",

        away_odds_column:
            "AwayOdds",
    }
)


market[
    "OddsSource"
] = selected_source


# --------------------------------------------------
# Ensure odds are numeric
# --------------------------------------------------

for column in [
    "HomeOdds",
    "DrawOdds",
    "AwayOdds",
]:

    market[column] = (
        pd.to_numeric(
            market[column],
            errors="coerce",
        )
    )


# --------------------------------------------------
# Remove rows without complete odds
# --------------------------------------------------

before = len(
    market
)


market = market.dropna(
    subset=[
        "HomeOdds",
        "DrawOdds",
        "AwayOdds",
    ]
).copy()


market = market[
    (market["HomeOdds"] > 1)
    &
    (market["DrawOdds"] > 1)
    &
    (market["AwayOdds"] > 1)
].copy()


after = len(
    market
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


market.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print(
    f"Original matches: {before}"
)

print(
    f"Matches with usable odds: {after}"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)


print()
print("SAMPLE")
print("======")

print(
    market[
        [
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",
            "HomeOdds",
            "DrawOdds",
            "AwayOdds",
            "OddsSource",
        ]
    ].head()
)


print()
print("MARKET DATA BUILD COMPLETE")
print("==========================")