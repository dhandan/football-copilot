import pickle

import pandas as pd
from scipy.stats import poisson


# --------------------------------------------------
# Files
# --------------------------------------------------

MATCH_FILE = (
    "data/processed/"
    "matches_clean.csv"
)

MODEL_FILE = (
    "models/"
    "production_model_v2.pkl"
)


# --------------------------------------------------
# Load production model
# --------------------------------------------------

with open(
    MODEL_FILE,
    "rb",
) as file:

    MODEL_BUNDLE = pickle.load(
        file
    )


MODEL_NAME = (
    MODEL_BUNDLE[
        "model_name"
    ]
)

MODEL_TYPE = (
    MODEL_BUNDLE[
        "model_type"
    ]
)

MODEL_ALPHA = (
    MODEL_BUNDLE[
        "model_alpha"
    ]
)

FEATURE_COLUMNS = (
    MODEL_BUNDLE[
        "features"
    ]
)

HOME_MODEL = (
    MODEL_BUNDLE[
        "home_model"
    ]
)

AWAY_MODEL = (
    MODEL_BUNDLE[
        "away_model"
    ]
)

VALIDATION = (
    MODEL_BUNDLE[
        "validation"
    ]
)


# --------------------------------------------------
# Load historical matches
# --------------------------------------------------

matches = pd.read_csv(
    MATCH_FILE
)


matches["Date"] = pd.to_datetime(
    matches["Date"],
    errors="coerce",
)


matches = matches.sort_values(
    "Date"
).reset_index(
    drop=True
)


# --------------------------------------------------
# Team validation
# --------------------------------------------------

def validate_team(
    team
):

    teams = set(
        matches[
            "HomeTeam"
        ]
    ) | set(
        matches[
            "AwayTeam"
        ]
    )


    if team not in teams:

        raise ValueError(
            f"Team '{team}' is not available "
            "in the current dataset."
        )


# --------------------------------------------------
# Historical match retrieval
# --------------------------------------------------

def get_previous_matches(
    team,
    n=None,
    venue=None,
    season=None,
):

    previous = matches[
        (
            matches[
                "HomeTeam"
            ]
            ==
            team
        )
        |
        (
            matches[
                "AwayTeam"
            ]
            ==
            team
        )
    ].copy()


    if venue == "Home":

        previous = previous[
            previous[
                "HomeTeam"
            ]
            ==
            team
        ]


    elif venue == "Away":

        previous = previous[
            previous[
                "AwayTeam"
            ]
            ==
            team
        ]


    if season is not None:

        previous = previous[
            previous[
                "Season"
            ]
            ==
            season
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
# Team stats
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


    for _, match in (
        previous_matches.iterrows()
    ):

        if (
            match[
                "HomeTeam"
            ]
            ==
            team
        ):

            gf = match[
                "FTHG"
            ]

            ga = match[
                "FTAG"
            ]


        else:

            gf = match[
                "FTAG"
            ]

            ga = match[
                "FTHG"
            ]


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
            /
            games,

        "ppg":
            points / games,
    }


# --------------------------------------------------
# Build production features
# --------------------------------------------------

def build_fixture_features(
    home_team,
    away_team,
):

    validate_team(
        home_team
    )

    validate_team(
        away_team
    )


    latest_season = (
        matches[
            "Season"
        ]
        .dropna()
        .iloc[-1]
    )


    # ----------------------------------------------
    # Last 5
    # ----------------------------------------------

    home_5 = calculate_team_stats(
        get_previous_matches(
            home_team,
            n=5,
        ),
        home_team,
    )


    away_5 = calculate_team_stats(
        get_previous_matches(
            away_team,
            n=5,
        ),
        away_team,
    )


    # ----------------------------------------------
    # Last 10
    # ----------------------------------------------

    home_10 = calculate_team_stats(
        get_previous_matches(
            home_team,
            n=10,
        ),
        home_team,
    )


    away_10 = calculate_team_stats(
        get_previous_matches(
            away_team,
            n=10,
        ),
        away_team,
    )


    # ----------------------------------------------
    # Venue-specific
    # ----------------------------------------------

    home_venue = calculate_team_stats(
        get_previous_matches(
            home_team,
            n=5,
            venue="Home",
        ),
        home_team,
    )


    away_venue = calculate_team_stats(
        get_previous_matches(
            away_team,
            n=5,
            venue="Away",
        ),
        away_team,
    )


    # ----------------------------------------------
    # Season-to-date
    # ----------------------------------------------

    home_season = calculate_team_stats(
        get_previous_matches(
            home_team,
            season=latest_season,
        ),
        home_team,
    )


    away_season = calculate_team_stats(
        get_previous_matches(
            away_team,
            season=latest_season,
        ),
        away_team,
    )


    # ----------------------------------------------
    # Frozen Model 2 feature vector
    # ----------------------------------------------

    features = {

        "HomeRecentGoalsFor":
            home_5[
                "goals_for_pg"
            ],

        "HomeRecentGoalsAgainst":
            home_5[
                "goals_against_pg"
            ],

        "HomeRecentPPG":
            home_5[
                "ppg"
            ],

        "AwayRecentGoalsFor":
            away_5[
                "goals_for_pg"
            ],

        "AwayRecentGoalsAgainst":
            away_5[
                "goals_against_pg"
            ],

        "AwayRecentPPG":
            away_5[
                "ppg"
            ],


        "Home10GoalsFor":
            home_10[
                "goals_for_pg"
            ],

        "Home10GoalsAgainst":
            home_10[
                "goals_against_pg"
            ],

        "Home10PPG":
            home_10[
                "ppg"
            ],

        "Away10GoalsFor":
            away_10[
                "goals_for_pg"
            ],

        "Away10GoalsAgainst":
            away_10[
                "goals_against_pg"
            ],

        "Away10PPG":
            away_10[
                "ppg"
            ],


        "HomeVenuePPG":
            home_venue[
                "ppg"
            ],

        "HomeVenueGoalsFor":
            home_venue[
                "goals_for_pg"
            ],

        "AwayVenuePPG":
            away_venue[
                "ppg"
            ],

        "AwayVenueGoalsFor":
            away_venue[
                "goals_for_pg"
            ],


        "HomeSeasonPPG":
            home_season[
                "ppg"
            ],

        "HomeSeasonGoalDifferencePG":
            home_season[
                "goal_difference_pg"
            ],

        "AwaySeasonPPG":
            away_season[
                "ppg"
            ],

        "AwaySeasonGoalDifferencePG":
            away_season[
                "goal_difference_pg"
            ],


        "RecentPPGDifference":
            (
                home_5[
                    "ppg"
                ]
                -
                away_5[
                    "ppg"
                ]
            ),

        "TenMatchPPGDifference":
            (
                home_10[
                    "ppg"
                ]
                -
                away_10[
                    "ppg"
                ]
            ),

        "SeasonPPGDifference":
            (
                home_season[
                    "ppg"
                ]
                -
                away_season[
                    "ppg"
                ]
            ),

        "AttackVsDefenceHome":
            (
                home_10[
                    "goals_for_pg"
                ]
                -
                away_10[
                    "goals_against_pg"
                ]
            ),

        "AttackVsDefenceAway":
            (
                away_10[
                    "goals_for_pg"
                ]
                -
                home_10[
                    "goals_against_pg"
                ]
            ),
    }


    return (
        features,
        latest_season,
    )


# --------------------------------------------------
# Poisson score matrix
# --------------------------------------------------

def calculate_probabilities(
    home_expected_goals,
    away_expected_goals,
    max_goals=8,
):

    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    scorelines = []


    for home_goals in range(
        max_goals + 1
    ):

        for away_goals in range(
            max_goals + 1
        ):

            probability = (
                poisson.pmf(
                    home_goals,
                    home_expected_goals,
                )
                *
                poisson.pmf(
                    away_goals,
                    away_expected_goals,
                )
            )


            scorelines.append(
                {
                    "score":
                        (
                            f"{home_goals}-"
                            f"{away_goals}"
                        ),

                    "probability":
                        float(
                            probability
                        ),
                }
            )


            if (
                home_goals
                >
                away_goals
            ):

                home_win += (
                    probability
                )


            elif (
                home_goals
                ==
                away_goals
            ):

                draw += probability


            else:

                away_win += (
                    probability
                )


    total = (
        home_win
        +
        draw
        +
        away_win
    )


    scorelines = sorted(
        scorelines,
        key=lambda item:
            item[
                "probability"
            ],
        reverse=True,
    )


    return {

        "home_win":
            float(
                home_win
                /
                total
            ),

        "draw":
            float(
                draw
                /
                total
            ),

        "away_win":
            float(
                away_win
                /
                total
            ),

        "scorelines":
            scorelines[
                :5
            ],
    }


# --------------------------------------------------
# Main production predictor
# --------------------------------------------------

def predict_fixture(
    home_team,
    away_team,
):

    (
        features,
        feature_season,
    ) = build_fixture_features(
        home_team,
        away_team,
    )


    feature_df = pd.DataFrame(
        [
            features
        ]
    )[
        FEATURE_COLUMNS
    ]


    expected_home_goals = float(
        HOME_MODEL.predict(
            feature_df
        )[0]
    )


    expected_away_goals = float(
        AWAY_MODEL.predict(
            feature_df
        )[0]
    )


    probabilities = (
        calculate_probabilities(
            expected_home_goals,
            expected_away_goals,
        )
    )


    most_likely_scores = [

        {
            "score":
                item[
                    "score"
                ],

            "probability_pct":
                round(
                    item[
                        "probability"
                    ]
                    *
                    100,
                    1,
                ),
        }

        for item in probabilities[
            "scorelines"
        ]
    ]


    return {

        "type":
            "fixture_prediction",

        "home_team":
            home_team,

        "away_team":
            away_team,

        "feature_season":
            feature_season,

        "expected_home_goals":
            round(
                expected_home_goals,
                2,
            ),

        "expected_away_goals":
            round(
                expected_away_goals,
                2,
            ),

        "home_win_probability":
            round(
                probabilities[
                    "home_win"
                ]
                *
                100,
                1,
            ),

        "draw_probability":
            round(
                probabilities[
                    "draw"
                ]
                *
                100,
                1,
            ),

        "away_win_probability":
            round(
                probabilities[
                    "away_win"
                ]
                *
                100,
                1,
            ),

        "most_likely_scores":
            most_likely_scores,

        "model": {

            "name":
                MODEL_NAME,

            "type":
                MODEL_TYPE,

            "alpha":
                MODEL_ALPHA,

            "validation_matches":
                VALIDATION[
                    "validation_matches"
                ],

            "accuracy_pct":
                VALIDATION[
                    "model_accuracy_pct"
                ],

            "log_loss":
                VALIDATION[
                    "model_log_loss"
                ],

            "brier":
                VALIDATION[
                    "model_brier"
                ],

            "market_accuracy_pct":
                VALIDATION[
                    "market_accuracy_pct"
                ],

            "market_log_loss":
                VALIDATION[
                    "market_log_loss"
                ],

            "market_brier":
                VALIDATION[
                    "market_brier"
                ],
        },

        "market_value_note":
            (
                "Historical model-market "
                "disagreements did not produce "
                "a persistent positive-return "
                "signal in final validation."
            ),
    }