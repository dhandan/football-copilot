from pathlib import Path

import pandas as pd


# ==================================================
# FILES
# ==================================================

FOOTBALL_COPILOT_FILE = Path(
    "data/processed/matches_clean.csv"
)

FBREF_FILE = Path(
    "data/external/fbref_xg/final_matches.csv"
)


# ==================================================
# SEASON MAPPING
# ==================================================

SEASON_MAP = {
    2022: "2021/22",
    2023: "2022/23",
    2024: "2023/24",
    2025: "2024/25",
}


# ==================================================
# TEAM NAME NORMALISATION
# ==================================================

TEAM_MAP = {
    # Brighton
    "Brighton And Hove Albion": "Brighton",
    "Brighton": "Brighton",

    # Ipswich
    "Ipswich Town": "Ipswich",

    # Leeds
    "Leeds United": "Leeds",

    # Leicester
    "Leicester City": "Leicester",

    # Luton
    "Luton Town": "Luton",

    # Manchester City
    "Manchester City": "Man City",

    # Manchester United
    "Manchester United": "Man United",
    "Manchester Utd": "Man United",

    # Newcastle
    "Newcastle United": "Newcastle",
    "Newcastle Utd": "Newcastle",

    # Norwich
    "Norwich City": "Norwich",

    # Nottingham Forest
    "Nottingham Forest": "Nott'm Forest",
    "Nott'ham Forest": "Nott'm Forest",

    # Sheffield United
    "Sheffield United": "Sheffield United",
    "Sheffield Utd": "Sheffield United",

    # Tottenham
    "Tottenham Hotspur": "Tottenham",
    "Tottenham": "Tottenham",

    # West Brom
    "West Bromwich Albion": "West Brom",
    "West Brom": "West Brom",

    # West Ham
    "West Ham United": "West Ham",
    "West Ham": "West Ham",

    # Wolves
    "Wolverhampton Wanderers": "Wolves",
    "Wolves": "Wolves",
}

def normalise_team(team):
    return TEAM_MAP.get(
        team,
        team,
    )


# ==================================================
# BUILD MATCH-LEVEL FBREF DATA
# ==================================================

def build_fbref_matches(df):
    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = df[
        df["season"].isin(SEASON_MAP)
    ].copy()

    df["Season"] = df["season"].map(
        SEASON_MAP
    )

    # Each fixture exists twice in the source:
    # one row from each team's perspective.
    #
    # Using the Home row gives us:
    #
    # team     = HomeTeam
    # opponent = AwayTeam
    # xg       = Home xG
    # xga      = Away xG
    # sh       = Home shots
    # sot      = Home shots on target
    #
    # We then find the corresponding Away row
    # to obtain the Away attacking statistics.

    home = df[
        df["venue"] == "Home"
    ].copy()

    away = df[
        df["venue"] == "Away"
    ].copy()

    home["HomeTeam"] = (
        home["team"]
        .map(normalise_team)
    )

    home["AwayTeam"] = (
        home["opponent"]
        .map(normalise_team)
    )

    away["AwayTeam"] = (
        away["team"]
        .map(normalise_team)
    )

    away["HomeTeam"] = (
        away["opponent"]
        .map(normalise_team)
    )

    home = home.rename(
        columns={
            "date": "Date",
            "xg": "HomeXG",
            "xga": "AwayXGFromHomeRow",
            "sh": "HomeShots",
            "sot": "HomeShotsOnTarget",
            "poss": "HomePossession",
            "dist": "HomeShotDistance",
        }
    )

    away = away.rename(
        columns={
            "date": "Date",
            "xg": "AwayXG",
            "xga": "HomeXGFromAwayRow",
            "sh": "AwayShots",
            "sot": "AwayShotsOnTarget",
            "poss": "AwayPossession",
            "dist": "AwayShotDistance",
        }
    )

    home_columns = [
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "HomeXG",
        "AwayXGFromHomeRow",
        "HomeShots",
        "HomeShotsOnTarget",
        "HomePossession",
        "HomeShotDistance",
    ]

    away_columns = [
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "AwayXG",
        "HomeXGFromAwayRow",
        "AwayShots",
        "AwayShotsOnTarget",
        "AwayPossession",
        "AwayShotDistance",
    ]

    matches = home[
        home_columns
    ].merge(
        away[away_columns],
        on=[
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",
        ],
        how="outer",
        indicator=True,
    )

    return matches


# ==================================================
# MAIN
# ==================================================

def main():
    print()
    print("FOOTBALL COPILOT")
    print("FBREF XG JOIN VALIDATION")
    print("========================")
    print()

    fc = pd.read_csv(
        FOOTBALL_COPILOT_FILE
    )

    fbref_raw = pd.read_csv(
        FBREF_FILE
    )

    fc["Date"] = pd.to_datetime(
        fc["Date"],
        errors="coerce",
    )

    overlapping_seasons = list(
        SEASON_MAP.values()
    )

    fc = fc[
        fc["Season"].isin(
            overlapping_seasons
        )
    ].copy()

    fbref = build_fbref_matches(
        fbref_raw
    )

    print("Football Copilot matches:")
    print(
        fc.groupby("Season")
        .size()
        .to_string()
    )

    print()

    print("FBref reconstructed matches:")
    print(
        fbref.groupby("Season")
        .size()
        .to_string()
    )

    print()

    print("FBref home/away reconstruction:")
    print(
        fbref["_merge"]
        .value_counts()
        .to_string()
    )

    # ----------------------------------------------
    # Internal xG consistency
    # ----------------------------------------------

    complete_fbref = fbref[
        fbref["_merge"] == "both"
    ].copy()

    complete_fbref[
        "HomeXGDifference"
    ] = (
        complete_fbref["HomeXG"]
        -
        complete_fbref[
            "HomeXGFromAwayRow"
        ]
    ).abs()

    complete_fbref[
        "AwayXGDifference"
    ] = (
        complete_fbref["AwayXG"]
        -
        complete_fbref[
            "AwayXGFromHomeRow"
        ]
    ).abs()

    print()
    print("XG INTERNAL CONSISTENCY")
    print("=======================")

    print(
        "Maximum Home xG discrepancy:",
        complete_fbref[
            "HomeXGDifference"
        ].max(),
    )

    print(
        "Maximum Away xG discrepancy:",
        complete_fbref[
            "AwayXGDifference"
        ].max(),
    )

    # ----------------------------------------------
    # Join to Football Copilot
    # ----------------------------------------------

    fbref_for_join = complete_fbref[
        [
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",
            "HomeXG",
            "AwayXG",
            "HomeShots",
            "AwayShots",
            "HomeShotsOnTarget",
            "AwayShotsOnTarget",
            "HomePossession",
            "AwayPossession",
            "HomeShotDistance",
            "AwayShotDistance",
        ]
    ].copy()

    joined = fc.merge(
        fbref_for_join,
        on=[
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",
        ],
        how="outer",
        indicator=True,
    )

    print()
    print("JOIN RESULT")
    print("===========")

    print(
        joined["_merge"]
        .value_counts()
        .to_string()
    )

    print()

    print("JOIN RATE BY SEASON")
    print("===================")

    for season in overlapping_seasons:
        season_df = joined[
            joined["Season"] == season
        ]

        both = (
            season_df["_merge"]
            ==
            "both"
        ).sum()

        left_only = (
            season_df["_merge"]
            ==
            "left_only"
        ).sum()

        right_only = (
            season_df["_merge"]
            ==
            "right_only"
        ).sum()

        expected = len(
            fc[
                fc["Season"]
                ==
                season
            ]
        )

        rate = (
            both / expected * 100
            if expected
            else 0
        )

        print(
            f"{season}: "
            f"{both}/{expected} "
            f"({rate:.2f}%) | "
            f"FC only={left_only} | "
            f"FBref only={right_only}"
        )

    # ----------------------------------------------
    # Show mismatches
    # ----------------------------------------------

    mismatches = joined[
        joined["_merge"] != "both"
    ].copy()

    print()
    print("MISMATCHES")
    print("==========")

    if mismatches.empty:
        print(
            "None. Perfect fixture reconciliation."
        )

    else:
        columns = [
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",
            "_merge",
        ]

        print(
            mismatches[
                columns
            ]
            .sort_values(
                [
                    "Season",
                    "Date",
                ]
            )
            .to_string(
                index=False
            )
        )

    print()
    print("VALIDATION COMPLETE")
    print("===================")


if __name__ == "__main__":
    main()