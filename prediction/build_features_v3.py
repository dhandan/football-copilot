from pathlib import Path
import math

import pandas as pd


# --------------------------------------------------
# Files
# --------------------------------------------------

INPUT_FILE = "data/processed/matches_clean.csv"

OUTPUT_FILE = (
    "data/processed/"
    "prediction_features_v3.csv"
)


print()
print("BUILDING MODEL 3 FEATURES")
print("=========================")


# --------------------------------------------------
# Load matches
# --------------------------------------------------

matches = pd.read_csv(INPUT_FILE)

matches["Date"] = pd.to_datetime(
    matches["Date"],
    errors="coerce",
)

matches = matches.sort_values(
    ["Date", "HomeTeam", "AwayTeam"]
).reset_index(drop=True)


# --------------------------------------------------
# ELO SETTINGS
# --------------------------------------------------

STARTING_ELO = 1500.0
K_FACTOR = 20.0
HOME_ELO_ADVANTAGE = 65.0


# --------------------------------------------------
# Elo helper
# --------------------------------------------------

def expected_score(
    rating_a,
    rating_b,
):
    return 1 / (
        1
        +
        10 ** (
            (rating_b - rating_a)
            / 400
        )
    )


def update_elo(
    home_elo,
    away_elo,
    home_goals,
    away_goals,
):
    """
    Update home and away Elo ratings
    after one match.
    """

    adjusted_home_elo = (
        home_elo
        +
        HOME_ELO_ADVANTAGE
    )

    expected_home = expected_score(
        adjusted_home_elo,
        away_elo,
    )

    expected_away = (
        1
        -
        expected_home
    )


    if home_goals > away_goals:

        actual_home = 1.0
        actual_away = 0.0


    elif home_goals == away_goals:

        actual_home = 0.5
        actual_away = 0.5


    else:

        actual_home = 0.0
        actual_away = 1.0


    new_home_elo = (
        home_elo
        +
        K_FACTOR
        *
        (
            actual_home
            -
            expected_home
        )
    )


    new_away_elo = (
        away_elo
        +
        K_FACTOR
        *
        (
            actual_away
            -
            expected_away
        )
    )


    return (
        new_home_elo,
        new_away_elo,
    )


# --------------------------------------------------
# Previous matches helper
# --------------------------------------------------

def get_previous_matches(
    data,
    team,
    current_date,
    n=None,
    season=None,
    venue=None,
):
    previous = data[
        (
            (
                data["HomeTeam"]
                ==
                team
            )
            |
            (
                data["AwayTeam"]
                ==
                team
            )
        )
        &
        (
            data["Date"]
            <
            current_date
        )
    ].copy()


    if season is not None:

        previous = previous[
            previous["Season"]
            ==
            season
        ]


    if venue == "Home":

        previous = previous[
            previous["HomeTeam"]
            ==
            team
        ]


    elif venue == "Away":

        previous = previous[
            previous["AwayTeam"]
            ==
            team
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
# Standard stats
# --------------------------------------------------

def calculate_team_stats(
    previous_matches,
    team,
):
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

        if (
            match["HomeTeam"]
            ==
            team
        ):

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
        "games":
            games,

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
# Exponentially weighted stats
# --------------------------------------------------

def calculate_weighted_stats(
    previous_matches,
    team,
    decay=0.85,
):
    """
    Older matches receive progressively
    smaller weights.

    Most recent match weight = 1
    Previous match = decay
    Then decay^2, etc.
    """

    if len(
        previous_matches
    ) == 0:

        return {
            "weighted_goals_for": 0.0,
            "weighted_goals_against": 0.0,
            "weighted_ppg": 0.0,
        }


    previous_matches = (
        previous_matches
        .sort_values("Date")
        .copy()
    )


    weighted_gf = 0.0
    weighted_ga = 0.0
    weighted_points = 0.0
    total_weight = 0.0


    reverse_matches = list(
        previous_matches
        .iterrows()
    )[::-1]


    for position, (
        _,
        match,
    ) in enumerate(
        reverse_matches
    ):

        weight = (
            decay
            **
            position
        )


        if (
            match["HomeTeam"]
            ==
            team
        ):

            gf = match["FTHG"]
            ga = match["FTAG"]


        else:

            gf = match["FTAG"]
            ga = match["FTHG"]


        if gf > ga:
            points = 3

        elif gf == ga:
            points = 1

        else:
            points = 0


        weighted_gf += (
            gf
            *
            weight
        )

        weighted_ga += (
            ga
            *
            weight
        )

        weighted_points += (
            points
            *
            weight
        )

        total_weight += weight


    return {
        "weighted_goals_for":
            weighted_gf
            /
            total_weight,

        "weighted_goals_against":
            weighted_ga
            /
            total_weight,

        "weighted_ppg":
            weighted_points
            /
            total_weight,
    }


# --------------------------------------------------
# Pre-calculate Elo before every fixture
# --------------------------------------------------

elo_ratings = {}

elo_rows = []


for index, match in matches.iterrows():

    home_team = match[
        "HomeTeam"
    ]

    away_team = match[
        "AwayTeam"
    ]


    if (
        home_team
        not in elo_ratings
    ):

        elo_ratings[
            home_team
        ] = STARTING_ELO


    if (
        away_team
        not in elo_ratings
    ):

        elo_ratings[
            away_team
        ] = STARTING_ELO


    home_elo_before = (
        elo_ratings[
            home_team
        ]
    )

    away_elo_before = (
        elo_ratings[
            away_team
        ]
    )


    elo_rows.append(
        {
            "HomeElo":
                home_elo_before,

            "AwayElo":
                away_elo_before,

            "EloDifference":
                (
                    home_elo_before
                    +
                    HOME_ELO_ADVANTAGE
                    -
                    away_elo_before
                ),
        }
    )


    (
        new_home_elo,
        new_away_elo,
    ) = update_elo(
        home_elo_before,
        away_elo_before,
        match["FTHG"],
        match["FTAG"],
    )


    elo_ratings[
        home_team
    ] = new_home_elo

    elo_ratings[
        away_team
    ] = new_away_elo


elo_df = pd.DataFrame(
    elo_rows
)


matches = pd.concat(
    [
        matches,
        elo_df,
    ],
    axis=1,
)


# --------------------------------------------------
# Build Model 3 features
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
    # Last 5
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
    # Last 10
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
    # Venue stats
    # ----------------------------------------------

    home_recent_home = (
        get_previous_matches(
            matches,
            home_team,
            current_date,
            n=8,
            venue="Home",
        )
    )

    away_recent_away = (
        get_previous_matches(
            matches,
            away_team,
            current_date,
            n=8,
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
    # Season-to-date
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
    # Weighted long-term stats
    #
    # Use last 20 matches with exponential decay.
    # ----------------------------------------------

    home_last_20 = (
        get_previous_matches(
            matches,
            home_team,
            current_date,
            n=20,
        )
    )

    away_last_20 = (
        get_previous_matches(
            matches,
            away_team,
            current_date,
            n=20,
        )
    )


    home_weighted = (
        calculate_weighted_stats(
            home_last_20,
            home_team,
            decay=0.88,
        )
    )

    away_weighted = (
        calculate_weighted_stats(
            away_last_20,
            away_team,
            decay=0.88,
        )
    )


    # ----------------------------------------------
    # Weighted venue stats
    # ----------------------------------------------

    home_weighted_venue = (
        calculate_weighted_stats(
            home_recent_home,
            home_team,
            decay=0.88,
        )
    )

    away_weighted_venue = (
        calculate_weighted_stats(
            away_recent_away,
            away_team,
            decay=0.88,
        )
    )


    row = {

        # ------------------------------------------
        # Match identity
        # ------------------------------------------

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
        # Model 1 features
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
        # Model 2 features
        # ------------------------------------------

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


        # ------------------------------------------
        # MODEL 3: ELO
        # ------------------------------------------

        "HomeElo":
            match[
                "HomeElo"
            ],

        "AwayElo":
            match[
                "AwayElo"
            ],

        "EloDifference":
            match[
                "EloDifference"
            ],


        # ------------------------------------------
        # MODEL 3: weighted strength
        # ------------------------------------------

        "HomeWeightedGoalsFor":
            home_weighted[
                "weighted_goals_for"
            ],

        "HomeWeightedGoalsAgainst":
            home_weighted[
                "weighted_goals_against"
            ],

        "HomeWeightedPPG":
            home_weighted[
                "weighted_ppg"
            ],

        "AwayWeightedGoalsFor":
            away_weighted[
                "weighted_goals_for"
            ],

        "AwayWeightedGoalsAgainst":
            away_weighted[
                "weighted_goals_against"
            ],

        "AwayWeightedPPG":
            away_weighted[
                "weighted_ppg"
            ],


        # ------------------------------------------
        # MODEL 3: weighted venue
        # ------------------------------------------

        "HomeWeightedVenueGoalsFor":
            home_weighted_venue[
                "weighted_goals_for"
            ],

        "HomeWeightedVenueGoalsAgainst":
            home_weighted_venue[
                "weighted_goals_against"
            ],

        "HomeWeightedVenuePPG":
            home_weighted_venue[
                "weighted_ppg"
            ],

        "AwayWeightedVenueGoalsFor":
            away_weighted_venue[
                "weighted_goals_for"
            ],

        "AwayWeightedVenueGoalsAgainst":
            away_weighted_venue[
                "weighted_goals_against"
            ],

        "AwayWeightedVenuePPG":
            away_weighted_venue[
                "weighted_ppg"
            ],


        # ------------------------------------------
        # MODEL 3: relative weighted strength
        # ------------------------------------------

        "WeightedPPGDifference":
            (
                home_weighted[
                    "weighted_ppg"
                ]
                -
                away_weighted[
                    "weighted_ppg"
                ]
            ),

        "WeightedAttackDifference":
            (
                home_weighted[
                    "weighted_goals_for"
                ]
                -
                away_weighted[
                    "weighted_goals_for"
                ]
            ),

        "WeightedDefenceDifference":
            (
                away_weighted[
                    "weighted_goals_against"
                ]
                -
                home_weighted[
                    "weighted_goals_against"
                ]
            ),

        "WeightedHomeAttackVsAwayDefence":
            (
                home_weighted_venue[
                    "weighted_goals_for"
                ]
                -
                away_weighted_venue[
                    "weighted_goals_against"
                ]
            ),

        "WeightedAwayAttackVsHomeDefence":
            (
                away_weighted_venue[
                    "weighted_goals_for"
                ]
                -
                home_weighted_venue[
                    "weighted_goals_against"
                ]
            ),


        # ------------------------------------------
        # History
        # ------------------------------------------

        "HomeHistoryGames10":
            home_10_stats[
                "games"
            ],

        "AwayHistoryGames10":
            away_10_stats[
                "games"
            ],

        "HomeHistoryGames20":
            len(
                home_last_20
            ),

        "AwayHistoryGames20":
            len(
                away_last_20
            ),
    }


    feature_rows.append(
        row
    )


# --------------------------------------------------
# Create feature dataframe
# --------------------------------------------------

features = pd.DataFrame(
    feature_rows
)


# --------------------------------------------------
# Fair common sample
#
# Require at least 20 historical matches for
# both teams.
# --------------------------------------------------

features = features[
    (
        features[
            "HomeHistoryGames20"
        ]
        >= 20
    )
    &
    (
        features[
            "AwayHistoryGames20"
        ]
        >= 20
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
    f"Rows created: "
    f"{len(features)}"
)

print(
    f"Columns created: "
    f"{len(features.columns)}"
)

print(
    f"Saved to: "
    f"{OUTPUT_FILE}"
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
print("MODEL 3 FEATURE BUILD COMPLETE")
print("==============================")