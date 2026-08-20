from pathlib import Path

import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


CHAMPIONSHIP_DIR = (
    PROJECT_ROOT
    / "data"
    / "championship"
)


OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "championship_team_seasons.csv"
)


SEASON_FILES = {
    "2021/22":
        "championship_2021_22.csv",

    "2022/23":
        "championship_2022_23.csv",

    "2023/24":
        "championship_2023_24.csv",

    "2024/25":
        "championship_2024_25.csv",

    "2025/26":
        "championship_2025_26.csv",
}


def calculate_team_stats(
    df,
    team,
):

    home = df[
        df["HomeTeam"]
        ==
        team
    ].copy()

    away = df[
        df["AwayTeam"]
        ==
        team
    ].copy()


    played = (
        len(home)
        +
        len(away)
    )


    home_goals_for = (
        home["FTHG"].sum()
    )

    home_goals_against = (
        home["FTAG"].sum()
    )


    away_goals_for = (
        away["FTAG"].sum()
    )

    away_goals_against = (
        away["FTHG"].sum()
    )


    goals_for = (
        home_goals_for
        +
        away_goals_for
    )

    goals_against = (
        home_goals_against
        +
        away_goals_against
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

        "Team":
            team,

        "Played":
            played,

        "Points":
            points,

        "PPG":
            points / played,

        "GoalsFor":
            goals_for,

        "GoalsAgainst":
            goals_against,

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


rows = []


for season, filename in SEASON_FILES.items():

    file_path = (
        CHAMPIONSHIP_DIR
        /
        filename
    )


    df = pd.read_csv(
        file_path
    )


    teams = sorted(
        set(
            df["HomeTeam"]
        )
        |
        set(
            df["AwayTeam"]
        )
    )


    for team in teams:

        stats = calculate_team_stats(
            df,
            team,
        )


        stats[
            "Season"
        ] = season


        rows.append(
            stats
        )


output = pd.DataFrame(
    rows
)


output = output[
    [
        "Season",
        "Team",
        "Played",
        "Points",
        "PPG",
        "GoalsFor",
        "GoalsAgainst",
        "GoalsForPG",
        "GoalsAgainstPG",
        "GoalDifferencePG",
    ]
]


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


output.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("CHAMPIONSHIP TEAM SEASONS")
print("=========================")

print(
    output.tail(
        30
    ).to_string(
        index=False
    )
)

print()
print(
    f"Saved: {OUTPUT_FILE}"
)