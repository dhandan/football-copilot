from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Files
# --------------------------------------------------

INPUT_FILE = "data/processed/matches_clean.csv"

OUTPUT_FILE = (
    "data/processed/"
    "prediction_features_v2.csv"
)


print()
print("BUILDING MODEL 2 FEATURES")
print("=========================")


# --------------------------------------------------
# Load match data
# --------------------------------------------------

matches = pd.read_csv(
    INPUT_FILE
)

matches["Date"] = pd.to_datetime(
    matches["Date"],
    errors="coerce",
)

matches = matches.sort_values(
    ["Date", "HomeTeam", "AwayTeam"]
).reset_index(drop=True)


# --------------------------------------------------
# Get previous matches for a team
# --------------------------------------------------

def get_previous_matches(
    data,
    team,
    current_date,
    n=None,
    season=None,
    venue=None,
):
    """
    Return matches played by a team before
    current_date.

    Optional filters:
        n       = most recent N matches
        season  = only one season
        venue   = Home or Away
    """

    previous = data[
        (
            (data["HomeTeam"] == team)
            |
            (data["AwayTeam"] == team)
        )
        &
        (
            data["Date"] < current_date
        )
    ].copy()


    if season is not None:

        previous = previous[
            previous["Season"] == season
        ]


    if venue == "Home":

        previous = previous[
            previous["HomeTeam"] == team
        ]


    elif venue == "Away":

        previous = previous[
            previous["AwayTeam"] == team
        ]


    previous = previous.sort_values(
        "Date"
    )


    if n is not None:

        previous = previous.tail(
            n
        )


    return previous


# --------------------------------------------------
# Calculate team statistics
# --------------------------------------------------

def calculate_team_stats(
    previous_matches,
    team,
):
    """
    Calculate statistics from the team's
    perspective.
    """

    games = len(
        previous_matches
    )


    if games == 0:

        return {
            "games": 0,
            "goals_for_pg": 0.0,
            "goals_against_pg": 0.0,
            "goal_difference_pg": 0.0,
            "ppg": 0.0,
        }


    goals_for = 0
    goals_against = 0
    points = 0


    for _, match in previous_matches.iterrows():

        if match["HomeTeam"] == team:

            gf = match["FTHG"]
            ga = match["FTAG"]


        else:

            gf = match["FTAG"]
            ga = match["FTHG"]


        goals_for += gf
        goals_against += ga


        if gf > ga:

            points += 3


        elif gf == ga:

            points += 1


    return {
        "games": games,

        "goals_for_pg":
            goals_for / games,

        "goals_against_pg":
            goals_against / games,

        "goal_difference_pg":
            (
                goals_for
                -
                goals_against
            )
            / games,

        "ppg":
            points / games,
    }


# --------------------------------------------------
# Build features
# --------------------------------------------------

feature_rows = []


for index, match in matches.iterrows():

    current_date = match[
        "Date"
    ]

    season = match[
        "Season"
    ]

    home_team = match[
        "HomeTeam"
    ]

    away_team = match[
        "AwayTeam"
    ]


    # ----------------------------------------------
    # Last 5 overall matches
    # ----------------------------------------------

    home_last_5 = (
        get_previous_matches(
            matches,
            home_team,
            current_date,
            n=5,
        )
    )

    away_last_5 = (
        get_previous_matches(
            matches,
            away_team,
            current_date,
            n=5,
        )
    )


    home_5_stats = (
        calculate_team_stats(
            home_last_5,
            home_team,
        )
    )

    away_5_stats = (
        calculate_team_stats(
            away_last_5,
            away_team,
        )
    )


    # ----------------------------------------------
    # Last 10 overall matches
    # ----------------------------------------------

    home_last_10 = (
        get_previous_matches(
            matches,
            home_team,
            current_date,
            n=10,
        )
    )

    away_last_10 = (
        get_previous_matches(
            matches,
            away_team,
            current_date,
            n=10,
        )
    )


    home_10_stats = (
        calculate_team_stats(
            home_last_10,
            home_team,
        )
    )

    away_10_stats = (
        calculate_team_stats(
            away_last_10,
            away_team,
        )
    )


    # ----------------------------------------------
    # Recent venue performance
    # ----------------------------------------------

    home_recent_home = (
        get_previous_matches(
            matches,
            home_team,
            current_date,
            n=5,
            venue="Home",
        )
    )

    away_recent_away = (
        get_previous_matches(
            matches,
            away_team,
            current_date,
            n=5,
            venue="Away",
        )
    )


    home_venue_stats = (
        calculate_team_stats(
            home_recent_home,
            home_team,
        )
    )

    away_venue_stats = (
        calculate_team_stats(
            away_recent_away,
            away_team,
        )
    )


    # ----------------------------------------------
    # Season-to-date performance
    # ----------------------------------------------

    home_season_matches = (
        get_previous_matches(
            matches,
            home_team,
            current_date,
            season=season,
        )
    )

    away_season_matches = (
        get_previous_matches(
            matches,
            away_team,
            current_date,
            season=season,
        )
    )


    home_season_stats = (
        calculate_team_stats(
            home_season_matches,
            home_team,
        )
    )

    away_season_stats = (
        calculate_team_stats(
            away_season_matches,
            away_team,
        )
    )


    # ----------------------------------------------
    # Create feature row
    # ----------------------------------------------

    row = {

        # Fixture information

        "Season":
            season,

        "Date":
            current_date,

        "HomeTeam":
            home_team,

        "AwayTeam":
            away_team,

        "HomeGoals":
            match["FTHG"],

        "AwayGoals":
            match["FTAG"],


        # ------------------------------------------
        # MODEL 1 FEATURES
        # ------------------------------------------

        "HomeRecentGoalsFor":
            home_5_stats[
                "goals_for_pg"
            ],

        "HomeRecentGoalsAgainst":
            home_5_stats[
                "goals_against_pg"
            ],

        "HomeRecentPPG":
            home_5_stats[
                "ppg"
            ],

        "AwayRecentGoalsFor":
            away_5_stats[
                "goals_for_pg"
            ],

        "AwayRecentGoalsAgainst":
            away_5_stats[
                "goals_against_pg"
            ],

        "AwayRecentPPG":
            away_5_stats[
                "ppg"
            ],


        # ------------------------------------------
        # MODEL 2 ADDITIONAL FEATURES
        # ------------------------------------------

        # 10-match form

        "Home10GoalsFor":
            home_10_stats[
                "goals_for_pg"
            ],

        "Home10GoalsAgainst":
            home_10_stats[
                "goals_against_pg"
            ],

        "Home10PPG":
            home_10_stats[
                "ppg"
            ],

        "Away10GoalsFor":
            away_10_stats[
                "goals_for_pg"
            ],

        "Away10GoalsAgainst":
            away_10_stats[
                "goals_against_pg"
            ],

        "Away10PPG":
            away_10_stats[
                "ppg"
            ],


        # Venue-specific form

        "HomeVenuePPG":
            home_venue_stats[
                "ppg"
            ],

        "HomeVenueGoalsFor":
            home_venue_stats[
                "goals_for_pg"
            ],

        "AwayVenuePPG":
            away_venue_stats[
                "ppg"
            ],

        "AwayVenueGoalsFor":
            away_venue_stats[
                "goals_for_pg"
            ],


        # Season-to-date strength

        "HomeSeasonPPG":
            home_season_stats[
                "ppg"
            ],

        "HomeSeasonGoalDifferencePG":
            home_season_stats[
                "goal_difference_pg"
            ],

        "AwaySeasonPPG":
            away_season_stats[
                "ppg"
            ],

        "AwaySeasonGoalDifferencePG":
            away_season_stats[
                "goal_difference_pg"
            ],


        # Relative strength

        "RecentPPGDifference":
            (
                home_5_stats["ppg"]
                -
                away_5_stats["ppg"]
            ),

        "TenMatchPPGDifference":
            (
                home_10_stats["ppg"]
                -
                away_10_stats["ppg"]
            ),

        "SeasonPPGDifference":
            (
                home_season_stats["ppg"]
                -
                away_season_stats["ppg"]
            ),

        "AttackVsDefenceHome":
            (
                home_10_stats[
                    "goals_for_pg"
                ]
                -
                away_10_stats[
                    "goals_against_pg"
                ]
            ),

        "AttackVsDefenceAway":
            (
                away_10_stats[
                    "goals_for_pg"
                ]
                -
                home_10_stats[
                    "goals_against_pg"
                ]
            ),


        # History counts used for filtering

        "HomeHistoryGames5":
            home_5_stats[
                "games"
            ],

        "AwayHistoryGames5":
            away_5_stats[
                "games"
            ],

        "HomeHistoryGames10":
            home_10_stats[
                "games"
            ],

        "AwayHistoryGames10":
            away_10_stats[
                "games"
            ],
    }


    feature_rows.append(
        row
    )


# --------------------------------------------------
# Create dataframe
# --------------------------------------------------

features = pd.DataFrame(
    feature_rows
)


# --------------------------------------------------
# Require 10 matches of history for both teams
#
# This ensures Model 1 and Model 2 will be evaluated
# using exactly the same fixture sample.
# --------------------------------------------------

features = features[
    (
        features[
            "HomeHistoryGames10"
        ]
        >= 10
    )
    &
    (
        features[
            "AwayHistoryGames10"
        ]
        >= 10
    )
].copy()


features = features.sort_values(
    "Date"
).reset_index(
    drop=True
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


features.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print(
    f"Rows created: {len(features)}"
)

print(
    f"Columns created: "
    f"{len(features.columns)}"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)


print()
print("ROWS BY SEASON")
print("==============")

print(
    features[
        "Season"
    ]
    .value_counts()
    .sort_index()
)


print()
print("MODEL 2 FEATURE BUILD COMPLETE")
print("==============================")