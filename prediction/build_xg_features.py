from pathlib import Path

import numpy as np
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

OUTPUT_FILE = Path(
    "data/processed/premier_league_matches_xg_enriched.csv"
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
# TEAM NORMALISATION
# ==================================================

TEAM_MAP = {
    "Brighton And Hove Albion": "Brighton",
    "Brighton": "Brighton",

    "Ipswich Town": "Ipswich",

    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Luton Town": "Luton",

    "Manchester City": "Man City",

    "Manchester United": "Man United",
    "Manchester Utd": "Man United",

    "Newcastle United": "Newcastle",
    "Newcastle Utd": "Newcastle",

    "Norwich City": "Norwich",

    "Nottingham Forest": "Nott'm Forest",
    "Nott'ham Forest": "Nott'm Forest",

    "Sheffield United": "Sheffield United",
    "Sheffield Utd": "Sheffield United",

    "Tottenham Hotspur": "Tottenham",
    "Tottenham": "Tottenham",

    "West Bromwich Albion": "West Brom",
    "West Brom": "West Brom",

    "West Ham United": "West Ham",
    "West Ham": "West Ham",

    "Wolverhampton Wanderers": "Wolves",
    "Wolves": "Wolves",
}


def normalise_team(team):
    return TEAM_MAP.get(team, team)


# ==================================================
# RECONSTRUCT ONE ROW PER MATCH
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

    home = df[
        df["venue"] == "Home"
    ].copy()

    away = df[
        df["venue"] == "Away"
    ].copy()

    home["HomeTeam"] = home[
        "team"
    ].map(normalise_team)

    home["AwayTeam"] = home[
        "opponent"
    ].map(normalise_team)

    away["AwayTeam"] = away[
        "team"
    ].map(normalise_team)

    away["HomeTeam"] = away[
        "opponent"
    ].map(normalise_team)

    home = home.rename(
        columns={
            "date": "Date",
            "xg": "HomeXG",
            "xga": "AwayXGCheck",
            "gf": "HomeGoals",
            "ga": "AwayGoalsCheck",
            "sh": "HomeShots",
            "sot": "HomeShotsOnTarget",
            "poss": "HomePossession",
        }
    )

    away = away.rename(
        columns={
            "date": "Date",
            "xg": "AwayXG",
            "xga": "HomeXGCheck",
            "gf": "AwayGoals",
            "ga": "HomeGoalsCheck",
            "sh": "AwayShots",
            "sot": "AwayShotsOnTarget",
            "poss": "AwayPossession",
        }
    )

    home_cols = [
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "HomeXG",
        "AwayXGCheck",
        "HomeGoals",
        "AwayGoalsCheck",
        "HomeShots",
        "HomeShotsOnTarget",
        "HomePossession",
    ]

    away_cols = [
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "AwayXG",
        "HomeXGCheck",
        "AwayGoals",
        "HomeGoalsCheck",
        "AwayShots",
        "AwayShotsOnTarget",
        "AwayPossession",
    ]

    matches = home[
        home_cols
    ].merge(
        away[away_cols],
        on=[
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",
        ],
        how="inner",
    )

    return matches


# ==================================================
# BUILD TEAM-PERSPECTIVE HISTORY
# ==================================================

def build_team_history(matches):
    home = pd.DataFrame({
        "Season": matches["Season"],
        "Date": matches["Date"],
        "Team": matches["HomeTeam"],
        "Opponent": matches["AwayTeam"],
        "Venue": "Home",
        "XGFor": matches["HomeXG"],
        "XGAgainst": matches["AwayXG"],
        "GoalsFor": matches["HomeGoals"],
        "GoalsAgainst": matches["AwayGoals"],
        "Shots": matches["HomeShots"],
        "ShotsOnTarget": matches["HomeShotsOnTarget"],
        "Possession": matches["HomePossession"],
    })

    away = pd.DataFrame({
        "Season": matches["Season"],
        "Date": matches["Date"],
        "Team": matches["AwayTeam"],
        "Opponent": matches["HomeTeam"],
        "Venue": "Away",
        "XGFor": matches["AwayXG"],
        "XGAgainst": matches["HomeXG"],
        "GoalsFor": matches["AwayGoals"],
        "GoalsAgainst": matches["HomeGoals"],
        "Shots": matches["AwayShots"],
        "ShotsOnTarget": matches["AwayShotsOnTarget"],
        "Possession": matches["AwayPossession"],
    })

    history = pd.concat(
        [home, away],
        ignore_index=True,
    )

    history = history.sort_values(
        [
            "Team",
            "Date",
        ]
    ).reset_index(drop=True)

    return history


# ==================================================
# LEAKAGE-SAFE ROLLING FEATURES
# ==================================================

def add_rolling_features(history):
    history = history.copy()

    metrics = [
        "XGFor",
        "XGAgainst",
        "GoalsFor",
        "GoalsAgainst",
        "Shots",
        "ShotsOnTarget",
        "Possession",
    ]

    windows = [
        5,
        10,
    ]

    for metric in metrics:
        for window in windows:

            col = (
                f"{metric}Avg{window}"
            )

            history[col] = (
                history
                .groupby("Team")[metric]
                .transform(
                    lambda s: (
                        s.shift(1)
                        .rolling(
                            window=window,
                            min_periods=1,
                        )
                        .mean()
                    )
                )
            )

    # ----------------------------------------------
    # Derived xG features
    # ----------------------------------------------

    for window in windows:

        history[
            f"XGDifferenceAvg{window}"
        ] = (
            history[
                f"XGForAvg{window}"
            ]
            -
            history[
                f"XGAgainstAvg{window}"
            ]
        )

        history[
            f"GoalsMinusXGAvg{window}"
        ] = (
            history[
                f"GoalsForAvg{window}"
            ]
            -
            history[
                f"XGForAvg{window}"
            ]
        )

        history[
            f"GoalsAgainstMinusXGAAvg{window}"
        ] = (
            history[
                f"GoalsAgainstAvg{window}"
            ]
            -
            history[
                f"XGAgainstAvg{window}"
            ]
        )

    # ----------------------------------------------
    # Short-term xG trend
    # ----------------------------------------------

    history[
        "XGForTrend"
    ] = (
        history["XGForAvg5"]
        -
        history["XGForAvg10"]
    )

    history[
        "XGAgainstTrend"
    ] = (
        history["XGAgainstAvg5"]
        -
        history["XGAgainstAvg10"]
    )

    return history


# ==================================================
# CONVERT TEAM FEATURES BACK TO MATCH LEVEL
# ==================================================

def build_match_features(
    matches,
    history,
):
    feature_cols = [
        "XGForAvg5",
        "XGAgainstAvg5",
        "XGDifferenceAvg5",
        "GoalsMinusXGAvg5",
        "GoalsAgainstMinusXGAAvg5",

        "XGForAvg10",
        "XGAgainstAvg10",
        "XGDifferenceAvg10",
        "GoalsMinusXGAvg10",
        "GoalsAgainstMinusXGAAvg10",

        "ShotsAvg5",
        "ShotsOnTargetAvg5",
        "PossessionAvg5",

        "ShotsAvg10",
        "ShotsOnTargetAvg10",
        "PossessionAvg10",

        "XGForTrend",
        "XGAgainstTrend",
    ]

    home_history = history[
        history["Venue"] == "Home"
    ][
        [
            "Season",
            "Date",
            "Team",
        ]
        +
        feature_cols
    ].copy()

    away_history = history[
        history["Venue"] == "Away"
    ][
        [
            "Season",
            "Date",
            "Team",
        ]
        +
        feature_cols
    ].copy()

    home_history = home_history.rename(
        columns={
            "Team": "HomeTeam",
            **{
                col: f"Home{col}"
                for col in feature_cols
            },
        }
    )

    away_history = away_history.rename(
        columns={
            "Team": "AwayTeam",
            **{
                col: f"Away{col}"
                for col in feature_cols
            },
        }
    )

    enriched = matches.merge(
        home_history,
        on=[
            "Season",
            "Date",
            "HomeTeam",
        ],
        how="left",
    )

    enriched = enriched.merge(
        away_history,
        on=[
            "Season",
            "Date",
            "AwayTeam",
        ],
        how="left",
    )

    return enriched


# ==================================================
# MAIN
# ==================================================

def main():
    print()
    print("FOOTBALL COPILOT")
    print("BUILD XG ENRICHED FEATURES")
    print("==========================")
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

    fbref_matches = build_fbref_matches(
        fbref_raw
    )

    print(
        "Reconstructed FBref matches:",
        len(fbref_matches),
    )

    # ----------------------------------------------
    # Join external data to canonical FC matches
    # ----------------------------------------------

    external_cols = [
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "HomeXG",
        "AwayXG",
        "HomeGoals",
        "AwayGoals",
        "HomeShots",
        "AwayShots",
        "HomeShotsOnTarget",
        "AwayShotsOnTarget",
        "HomePossession",
        "AwayPossession",
    ]

    enriched_matches = fc.merge(
        fbref_matches[
            external_cols
        ],
        on=[
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",
        ],
        how="inner",
    )

    print(
        "Joined canonical matches:",
        len(enriched_matches),
    )

    # ----------------------------------------------
    # Team history
    # ----------------------------------------------

    history = build_team_history(
        enriched_matches
    )

    history = add_rolling_features(
        history
    )

    # ----------------------------------------------
    # Match-level pre-match features
    # ----------------------------------------------

    final = build_match_features(
        enriched_matches,
        history,
    )

    # ----------------------------------------------
    # Difference features
    # ----------------------------------------------

    for window in [5, 10]:

        final[
            f"XGForDifference{window}"
        ] = (
            final[
                f"HomeXGForAvg{window}"
            ]
            -
            final[
                f"AwayXGForAvg{window}"
            ]
        )

        final[
            f"XGAgainstDifference{window}"
        ] = (
            final[
                f"HomeXGAgainstAvg{window}"
            ]
            -
            final[
                f"AwayXGAgainstAvg{window}"
            ]
        )

        final[
            f"XGDifferenceDifference{window}"
        ] = (
            final[
                f"HomeXGDifferenceAvg{window}"
            ]
            -
            final[
                f"AwayXGDifferenceAvg{window}"
            ]
        )

    # ----------------------------------------------
    # Save
    # ----------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "Output:",
        OUTPUT_FILE,
    )

    print(
        "Rows:",
        len(final),
    )

    print(
        "Columns:",
        len(final.columns),
    )

    print()
    print("FEATURE COMPLETENESS")
    print("====================")

    feature_test = [
        "HomeXGForAvg5",
        "AwayXGForAvg5",
        "HomeXGAgainstAvg5",
        "AwayXGAgainstAvg5",
        "HomeXGForAvg10",
        "AwayXGForAvg10",
        "XGForDifference5",
        "XGDifferenceDifference5",
    ]

    for col in feature_test:

        missing = final[
            col
        ].isna().sum()

        pct = (
            missing
            /
            len(final)
            *
            100
        )

        print(
            f"{col:30} "
            f"missing={missing:4} "
            f"({pct:.2f}%)"
        )

    print()
    print("BUILD COMPLETE")
    print("==============")


if __name__ == "__main__":
    main()