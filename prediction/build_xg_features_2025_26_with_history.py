from pathlib import Path

import pandas as pd


# ==================================================
# FILES
# ==================================================

HISTORICAL_XG_FILE = Path(
    "data/processed/"
    "premier_league_matches_xg_enriched.csv"
)

CURRENT_XG_FILE = Path(
    "data/processed/"
    "premier_league_2025_26_xg_features.csv"
)

OUTPUT_FILE = Path(
    "data/processed/"
    "premier_league_2025_26_xg_features_with_history.csv"
)


# ==================================================
# LOAD HISTORICAL XG DATA
# 2021/22 to 2024/25
# ==================================================

print()
print("FOOTBALL COPILOT")
print("BUILD 2025/26 XG FEATURES WITH HISTORY")
print("======================================")
print()

historical = pd.read_csv(
    HISTORICAL_XG_FILE
)

historical["Date"] = pd.to_datetime(
    historical["Date"],
    errors="raise",
)


historical = historical[
    historical["Season"].isin(
        [
            "2021/22",
            "2022/23",
            "2023/24",
            "2024/25",
        ]
    )
].copy()


# ==================================================
# LOAD 2025/26 BRIDGED XG DATA
# ==================================================

current = pd.read_csv(
    CURRENT_XG_FILE
)

current["Date"] = pd.to_datetime(
    current["Date"],
    format="%Y-%m-%d",
    errors="raise",
)


# ==================================================
# STANDARDISE OBSERVED XG COLUMNS
# ==================================================
#
# Historical file contains FBref-derived observed xG.
# Current file contains Understat xG transformed onto
# the historical FBref scale.
#
# We combine them only after the provider bridge has
# already been frozen using 2024/25.
# ==================================================

historical_matches = historical[
    [
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "HomeXG",
        "AwayXG",
    ]
].copy()


historical_matches = historical_matches.rename(
    columns={
        "HomeXG":
            "ObservedHomeXG",

        "AwayXG":
            "ObservedAwayXG",
    }
)


current_matches = current[
    [
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "BridgedHomeXG",
        "BridgedAwayXG",
    ]
].copy()


current_matches = current_matches.rename(
    columns={
        "BridgedHomeXG":
            "ObservedHomeXG",

        "BridgedAwayXG":
            "ObservedAwayXG",
    }
)


# ==================================================
# COMBINE HISTORY
# ==================================================

all_matches = pd.concat(
    [
        historical_matches,
        current_matches,
    ],
    ignore_index=True,
)


all_matches = (
    all_matches
    .sort_values(
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    "Historical matches:",
    len(historical_matches),
)

print(
    "2025/26 matches:",
    len(current_matches),
)

print(
    "Combined matches:",
    len(all_matches),
)


# ==================================================
# CREATE TEAM-PERSPECTIVE HISTORY
# ==================================================

home_rows = pd.DataFrame(
    {
        "Season":
            all_matches["Season"],

        "Date":
            all_matches["Date"],

        "Team":
            all_matches["HomeTeam"],

        "Opponent":
            all_matches["AwayTeam"],

        "Venue":
            "Home",

        "XGFor":
            all_matches["ObservedHomeXG"],

        "XGAgainst":
            all_matches["ObservedAwayXG"],

        "GoalsFor":
            all_matches["FTHG"],

        "GoalsAgainst":
            all_matches["FTAG"],
    }
)


away_rows = pd.DataFrame(
    {
        "Season":
            all_matches["Season"],

        "Date":
            all_matches["Date"],

        "Team":
            all_matches["AwayTeam"],

        "Opponent":
            all_matches["HomeTeam"],

        "Venue":
            "Away",

        "XGFor":
            all_matches["ObservedAwayXG"],

        "XGAgainst":
            all_matches["ObservedHomeXG"],

        "GoalsFor":
            all_matches["FTAG"],

        "GoalsAgainst":
            all_matches["FTHG"],
    }
)


team_history = pd.concat(
    [
        home_rows,
        away_rows,
    ],
    ignore_index=True,
)


team_history = (
    team_history
    .sort_values(
        [
            "Team",
            "Date",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ==================================================
# ROLLING FEATURE FUNCTION
# STRICTLY PRIOR MATCHES ONLY
# ==================================================

def rolling_prior_mean(
    series,
    window,
):
    return (
        series
        .shift(1)
        .rolling(
            window=window,
            min_periods=1,
        )
        .mean()
    )


# ==================================================
# BUILD ROLLING FEATURES
# ==================================================

for window in [
    5,
    10,
]:

    team_history[
        f"XGForAvg{window}"
    ] = (
        team_history
        .groupby(
            "Team"
        )[
            "XGFor"
        ]
        .transform(
            lambda series:
                rolling_prior_mean(
                    series,
                    window,
                )
        )
    )


    team_history[
        f"XGAgainstAvg{window}"
    ] = (
        team_history
        .groupby(
            "Team"
        )[
            "XGAgainst"
        ]
        .transform(
            lambda series:
                rolling_prior_mean(
                    series,
                    window,
                )
        )
    )


    team_history[
        f"GoalsForAvg{window}"
    ] = (
        team_history
        .groupby(
            "Team"
        )[
            "GoalsFor"
        ]
        .transform(
            lambda series:
                rolling_prior_mean(
                    series,
                    window,
                )
        )
    )


    team_history[
        f"GoalsAgainstAvg{window}"
    ] = (
        team_history
        .groupby(
            "Team"
        )[
            "GoalsAgainst"
        ]
        .transform(
            lambda series:
                rolling_prior_mean(
                    series,
                    window,
                )
        )
    )


    team_history[
        f"XGDifferenceAvg{window}"
    ] = (
        team_history[
            f"XGForAvg{window}"
        ]
        -
        team_history[
            f"XGAgainstAvg{window}"
        ]
    )


    team_history[
        f"GoalsMinusXGAvg{window}"
    ] = (
        team_history[
            f"GoalsForAvg{window}"
        ]
        -
        team_history[
            f"XGForAvg{window}"
        ]
    )


    team_history[
        f"GoalsAgainstMinusXGAAvg{window}"
    ] = (
        team_history[
            f"GoalsAgainstAvg{window}"
        ]
        -
        team_history[
            f"XGAgainstAvg{window}"
        ]
    )


# ==================================================
# TRENDS
# ==================================================

team_history[
    "XGForTrend"
] = (
    team_history[
        "XGForAvg5"
    ]
    -
    team_history[
        "XGForAvg10"
    ]
)


team_history[
    "XGAgainstTrend"
] = (
    team_history[
        "XGAgainstAvg5"
    ]
    -
    team_history[
        "XGAgainstAvg10"
    ]
)


# ==================================================
# FEATURE TABLES
# ==================================================

feature_columns = [
    "Season",
    "Date",
    "Team",
    "XGForAvg5",
    "XGAgainstAvg5",
    "XGForAvg10",
    "XGAgainstAvg10",
    "XGDifferenceAvg5",
    "XGDifferenceAvg10",
    "GoalsMinusXGAvg5",
    "GoalsMinusXGAvg10",
    "GoalsAgainstMinusXGAAvg5",
    "GoalsAgainstMinusXGAAvg10",
    "XGForTrend",
    "XGAgainstTrend",
]


home_features = (
    team_history[
        feature_columns
    ]
    .copy()
    .rename(
        columns={
            "Team":
                "HomeTeam",
        }
    )
)


home_features = home_features.rename(
    columns={
        column:
            "Home" + column
        for column in feature_columns
        if column
        not in [
            "Season",
            "Date",
            "Team",
        ]
    }
)


away_features = (
    team_history[
        feature_columns
    ]
    .copy()
    .rename(
        columns={
            "Team":
                "AwayTeam",
        }
    )
)


away_features = away_features.rename(
    columns={
        column:
            "Away" + column
        for column in feature_columns
        if column
        not in [
            "Season",
            "Date",
            "Team",
        ]
    }
)


# ==================================================
# EXTRACT 2025/26 MATCHES
# ==================================================

output = all_matches[
    all_matches["Season"]
    ==
    "2025/26"
].copy()


output = output.merge(
    home_features,
    on=[
        "Season",
        "Date",
        "HomeTeam",
    ],
    how="left",
    validate="one_to_one",
)


output = output.merge(
    away_features,
    on=[
        "Season",
        "Date",
        "AwayTeam",
    ],
    how="left",
    validate="one_to_one",
)


# ==================================================
# MATCH-LEVEL DIFFERENCE FEATURES
# ==================================================

output[
    "XGForDifference5"
] = (
    output["HomeXGForAvg5"]
    -
    output["AwayXGForAvg5"]
)


output[
    "XGForDifference10"
] = (
    output["HomeXGForAvg10"]
    -
    output["AwayXGForAvg10"]
)


output[
    "XGAgainstDifference5"
] = (
    output["HomeXGAgainstAvg5"]
    -
    output["AwayXGAgainstAvg5"]
)


output[
    "XGAgainstDifference10"
] = (
    output["HomeXGAgainstAvg10"]
    -
    output["AwayXGAgainstAvg10"]
)


output[
    "XGDifferenceDifference5"
] = (
    output["HomeXGDifferenceAvg5"]
    -
    output["AwayXGDifferenceAvg5"]
)


output[
    "XGDifferenceDifference10"
] = (
    output["HomeXGDifferenceAvg10"]
    -
    output["AwayXGDifferenceAvg10"]
)


# ==================================================
# SORT
# ==================================================

output = (
    output
    .sort_values(
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ==================================================
# SAVE
# ==================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


output.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ==================================================
# COMPLETENESS CHECK
# ==================================================

MODEL5_XG_FEATURES = [
    "HomeXGForAvg5",
    "HomeXGAgainstAvg5",
    "AwayXGForAvg5",
    "AwayXGAgainstAvg5",
    "HomeXGForAvg10",
    "HomeXGAgainstAvg10",
    "AwayXGForAvg10",
    "AwayXGAgainstAvg10",
    "HomeXGDifferenceAvg5",
    "AwayXGDifferenceAvg5",
    "HomeXGForTrend",
    "AwayXGForTrend",
    "HomeXGAgainstTrend",
    "AwayXGAgainstTrend",
]


print()
print("FEATURE COMPLETENESS")
print("====================")

for feature in MODEL5_XG_FEATURES:

    missing = int(
        output[
            feature
        ].isna().sum()
    )

    print(
        f"{feature:<30} "
        f"missing={missing:>3}"
    )


print()
print(
    "Output:",
    OUTPUT_FILE
)

print(
    "2025/26 rows:",
    len(output)
)

print()
print("BUILD COMPLETE")
print("==============")