from pathlib import Path
import pandas as pd


# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


CHAMPIONSHIP_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "championship_team_seasons.csv"
)


PREMIER_LEAGUE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "matches_clean.csv"
)


OUTPUT_SAMPLE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "promotion_translation_sample.csv"
)


OUTPUT_FACTORS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "promotion_translation_factors.csv"
)


# ==================================================
# SEASON ORDER
# ==================================================

SEASONS = [
    "2021/22",
    "2022/23",
    "2023/24",
    "2024/25",
    "2025/26",
]


# ==================================================
# TEAM NAME NORMALISATION
# ==================================================

TEAM_NAME_MAP = {

    "Ipswich Town":
        "Ipswich",

    "Leicester City":
        "Leicester",

    "Norwich City":
        "Norwich",

    "Sheffield United":
        "Sheffield United",

    "Luton Town":
        "Luton",

    "Southampton":
        "Southampton",

    "Burnley":
        "Burnley",

    "Leeds United":
        "Leeds",

    "Fulham":
        "Fulham",

    "Bournemouth":
        "Bournemouth",

    "Nott'm Forest":
        "Nott'm Forest",
}


def normalise_team_name(
    team
):

    return TEAM_NAME_MAP.get(
        team,
        team
    )


# ==================================================
# LOAD DATA
# ==================================================

championship = pd.read_csv(
    CHAMPIONSHIP_FILE
)


premier = pd.read_csv(
    PREMIER_LEAGUE_FILE
)


# ==================================================
# BUILD PREMIER LEAGUE TEAM-SEASON STATS
# ==================================================

def calculate_pl_team_stats(
    df,
    season,
    team,
):

    season_df = df[
        df["Season"]
        ==
        season
    ].copy()


    home = season_df[
        season_df["HomeTeam"]
        ==
        team
    ].copy()


    away = season_df[
        season_df["AwayTeam"]
        ==
        team
    ].copy()


    played = (
        len(home)
        +
        len(away)
    )


    if played == 0:

        return None


    goals_for = (
        home["FTHG"].sum()
        +
        away["FTAG"].sum()
    )


    goals_against = (
        home["FTAG"].sum()
        +
        away["FTHG"].sum()
    )


    points = 0


    points += (
        (home["FTR"] == "H").sum()
        * 3
    )

    points += (
        (home["FTR"] == "D").sum()
    )


    points += (
        (away["FTR"] == "A").sum()
        * 3
    )

    points += (
        (away["FTR"] == "D").sum()
    )


    return {

        "Played":
            played,

        "Points":
            points,

        "PPG":
            points / played,

        "GoalsForPG":
            goals_for / played,

        "GoalsAgainstPG":
            goals_against / played,

        "GoalDifferencePG":
            (
                goals_for
                -
                goals_against
            )
            /
            played,
    }


# ==================================================
# BUILD PREMIER LEAGUE TEAM LIST BY SEASON
# ==================================================

pl_teams_by_season = {}


for season in SEASONS:

    season_df = premier[
        premier["Season"]
        ==
        season
    ]


    teams = set(
        season_df[
            "HomeTeam"
        ].dropna()
    ) | set(
        season_df[
            "AwayTeam"
        ].dropna()
    )


    pl_teams_by_season[
        season
    ] = teams


# ==================================================
# IDENTIFY PROMOTED TEAMS AUTOMATICALLY
# ==================================================

translation_rows = []


for index in range(
    len(SEASONS) - 1
):

    championship_season = (
        SEASONS[
            index
        ]
    )


    premier_season = (
        SEASONS[
            index + 1
        ]
    )


    championship_teams = (
        championship[
            championship[
                "Season"
            ]
            ==
            championship_season
        ]
        .copy()
    )


    next_pl_teams = (
        pl_teams_by_season[
            premier_season
        ]
    )


    for _, row in (
        championship_teams
        .iterrows()
    ):

        championship_team = (
            row["Team"]
        )


        pl_team_name = normalise_team_name(
            championship_team
        )


        if (
            pl_team_name
            not in next_pl_teams
        ):

            continue


        pl_stats = (
            calculate_pl_team_stats(
                premier,
                premier_season,
                pl_team_name,
            )
        )


        if pl_stats is None:

            continue


        translation_rows.append(
            {

                "ChampionshipSeason":
                    championship_season,

                "PremierLeagueSeason":
                    premier_season,

                "Team":
                    championship_team,

                "PremierLeagueTeamName":
                    pl_team_name,

                "ChampionshipPPG":
                    row["PPG"],

                "PremierLeaguePPG":
                    pl_stats["PPG"],

                "ChampionshipGoalsForPG":
                    row["GoalsForPG"],

                "PremierLeagueGoalsForPG":
                    pl_stats["GoalsForPG"],

                "ChampionshipGoalsAgainstPG":
                    row["GoalsAgainstPG"],

                "PremierLeagueGoalsAgainstPG":
                    pl_stats["GoalsAgainstPG"],

                "PPGRatio":
                    (
                        pl_stats["PPG"]
                        /
                        row["PPG"]
                    )
                    if row["PPG"] > 0
                    else None,

                "GoalsForRatio":
                    (
                        pl_stats[
                            "GoalsForPG"
                        ]
                        /
                        row[
                            "GoalsForPG"
                        ]
                    )
                    if row[
                        "GoalsForPG"
                    ] > 0
                    else None,

                "GoalsAgainstRatio":
                    (
                        pl_stats[
                            "GoalsAgainstPG"
                        ]
                        /
                        row[
                            "GoalsAgainstPG"
                        ]
                    )
                    if row[
                        "GoalsAgainstPG"
                    ] > 0
                    else None,
            }
        )


# ==================================================
# CREATE SAMPLE
# ==================================================

sample = pd.DataFrame(
    translation_rows
)


print()
print("FOOTBALL COPILOT")
print("PROMOTION TRANSLATION SAMPLE")
print("============================")
print()


if sample.empty:

    print(
        "No promoted teams were identified."
    )

    raise SystemExit


print(
    sample.to_string(
        index=False,
        formatters={
            "ChampionshipPPG":
                "{:.3f}".format,

            "PremierLeaguePPG":
                "{:.3f}".format,

            "ChampionshipGoalsForPG":
                "{:.3f}".format,

            "PremierLeagueGoalsForPG":
                "{:.3f}".format,

            "ChampionshipGoalsAgainstPG":
                "{:.3f}".format,

            "PremierLeagueGoalsAgainstPG":
                "{:.3f}".format,

            "PPGRatio":
                "{:.3f}".format,

            "GoalsForRatio":
                "{:.3f}".format,

            "GoalsAgainstRatio":
                "{:.3f}".format,
        },
    )
)


# ==================================================
# SAVE SAMPLE
# ==================================================

sample.to_csv(
    OUTPUT_SAMPLE_FILE,
    index=False,
)


# ==================================================
# CALCULATE TRANSLATION FACTORS
# ==================================================

factors = pd.DataFrame(
    [
        {

            "Metric":
                "PPG",

            "MeanRatio":
                sample[
                    "PPGRatio"
                ].mean(),

            "MedianRatio":
                sample[
                    "PPGRatio"
                ].median(),

            "Teams":
                sample[
                    "PPGRatio"
                ].notna().sum(),
        },

        {

            "Metric":
                "GoalsForPG",

            "MeanRatio":
                sample[
                    "GoalsForRatio"
                ].mean(),

            "MedianRatio":
                sample[
                    "GoalsForRatio"
                ].median(),

            "Teams":
                sample[
                    "GoalsForRatio"
                ].notna().sum(),
        },

        {

            "Metric":
                "GoalsAgainstPG",

            "MeanRatio":
                sample[
                    "GoalsAgainstRatio"
                ].mean(),

            "MedianRatio":
                sample[
                    "GoalsAgainstRatio"
                ].median(),

            "Teams":
                sample[
                    "GoalsAgainstRatio"
                ].notna().sum(),
        },
    ]
)


print()
print("TRANSLATION FACTORS")
print("===================")
print()


print(
    factors.to_string(
        index=False,
        formatters={
            "MeanRatio":
                "{:.3f}".format,

            "MedianRatio":
                "{:.3f}".format,
        },
    )
)


# ==================================================
# SAVE FACTORS
# ==================================================

factors.to_csv(
    OUTPUT_FACTORS_FILE,
    index=False,
)


print()
print("FILES SAVED")
print("===========")

print(
    OUTPUT_SAMPLE_FILE
)

print(
    OUTPUT_FACTORS_FILE
)