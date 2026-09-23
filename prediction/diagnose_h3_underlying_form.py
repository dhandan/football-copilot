from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import poisson

from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import log_loss


# --------------------------------------------------
# Files
# --------------------------------------------------

INPUT_FILE = "data/processed/matches_clean.csv"

DETAIL_OUTPUT = (
    "reports/post_gw5/"
    "h3_underlying_form_predictions.csv"
)

SUMMARY_OUTPUT = (
    "reports/post_gw5/"
    "h3_underlying_form_summary.csv"
)

SEASON_OUTPUT = (
    "reports/post_gw5/"
    "h3_underlying_form_by_season.csv"
)


# --------------------------------------------------
# Model 2 features
# --------------------------------------------------

MODEL_2_FEATURES = [

    "HomeRecentGoalsFor",
    "HomeRecentGoalsAgainst",
    "HomeRecentPPG",

    "AwayRecentGoalsFor",
    "AwayRecentGoalsAgainst",
    "AwayRecentPPG",

    "Home10GoalsFor",
    "Home10GoalsAgainst",
    "Home10PPG",

    "Away10GoalsFor",
    "Away10GoalsAgainst",
    "Away10PPG",

    "HomeVenuePPG",
    "HomeVenueGoalsFor",

    "AwayVenuePPG",
    "AwayVenueGoalsFor",

    "HomeSeasonPPG",
    "HomeSeasonGoalDifferencePG",

    "AwaySeasonPPG",
    "AwaySeasonGoalDifferencePG",

    "RecentPPGDifference",
    "TenMatchPPGDifference",
    "SeasonPPGDifference",

    "AttackVsDefenceHome",
    "AttackVsDefenceAway",
]


# --------------------------------------------------
# H3 additional features
# --------------------------------------------------

H3_FEATURES = [

    "HomeRecentShotsFor",
    "HomeRecentShotsAgainst",
    "HomeRecentSOTFor",
    "HomeRecentSOTAgainst",
    "HomeRecentSOTRate",

    "AwayRecentShotsFor",
    "AwayRecentShotsAgainst",
    "AwayRecentSOTFor",
    "AwayRecentSOTAgainst",
    "AwayRecentSOTRate",

    "RecentShotsDifference",
    "RecentSOTDifference",

    "HomeAttackSOTVsAwayDefence",
    "AwayAttackSOTVsHomeDefence",
]


MODEL_H3_FEATURES = (
    MODEL_2_FEATURES
    +
    H3_FEATURES
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

LABEL_MAP = {
    "H": 0,
    "D": 1,
    "A": 2,
}


def actual_result(
    home_goals,
    away_goals,
):

    if home_goals > away_goals:
        return "H"

    if home_goals == away_goals:
        return "D"

    return "A"


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


def calculate_team_stats(
    previous_matches,
    team,
):

    games = len(previous_matches)

    if games == 0:

        return {
            "games": 0,
            "goals_for_pg": 0.0,
            "goals_against_pg": 0.0,
            "goal_difference_pg": 0.0,
            "ppg": 0.0,
        }

    goals_for = 0.0
    goals_against = 0.0
    points = 0.0

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


def calculate_shot_stats(
    previous_matches,
    team,
):

    games = len(previous_matches)

    if games == 0:

        return {
            "shots_for_pg": 0.0,
            "shots_against_pg": 0.0,
            "sot_for_pg": 0.0,
            "sot_against_pg": 0.0,
            "sot_rate": 0.0,
        }

    shots_for = 0.0
    shots_against = 0.0

    sot_for = 0.0
    sot_against = 0.0

    for _, match in previous_matches.iterrows():

        if match["HomeTeam"] == team:

            shots_for += match["HS"]
            shots_against += match["AS"]

            sot_for += match["HST"]
            sot_against += match["AST"]

        else:

            shots_for += match["AS"]
            shots_against += match["HS"]

            sot_for += match["AST"]
            sot_against += match["HST"]

    sot_rate = (
        sot_for / shots_for
        if shots_for > 0
        else 0.0
    )

    return {
        "shots_for_pg":
            shots_for / games,

        "shots_against_pg":
            shots_against / games,

        "sot_for_pg":
            sot_for / games,

        "sot_against_pg":
            sot_against / games,

        "sot_rate":
            sot_rate,
    }


def match_probabilities(
    home_xg,
    away_xg,
    max_goals=8,
):

    home = 0.0
    draw = 0.0
    away = 0.0

    for home_goals in range(
        max_goals + 1
    ):

        for away_goals in range(
            max_goals + 1
        ):

            probability = (
                poisson.pmf(
                    home_goals,
                    home_xg,
                )
                *
                poisson.pmf(
                    away_goals,
                    away_xg,
                )
            )

            if home_goals > away_goals:
                home += probability

            elif home_goals == away_goals:
                draw += probability

            else:
                away += probability

    total = (
        home
        +
        draw
        +
        away
    )

    return (
        home / total,
        draw / total,
        away / total,
    )


def predict_result(
    home_probability,
    draw_probability,
    away_probability,
):

    probabilities = {
        "H": home_probability,
        "D": draw_probability,
        "A": away_probability,
    }

    return max(
        probabilities,
        key=probabilities.get,
    )


def multiclass_brier(
    actual,
    probabilities,
):

    actual_matrix = np.zeros(
        (
            len(actual),
            3,
        )
    )

    for i, result in enumerate(actual):

        actual_matrix[
            i,
            LABEL_MAP[result]
        ] = 1

    return np.mean(
        np.sum(
            (
                probabilities
                -
                actual_matrix
            )
            ** 2,
            axis=1,
        )
    )


def calculate_metrics(
    results,
    prefix,
):

    actual = results[
        "ActualResult"
    ]

    predicted = results[
        f"{prefix}PredictedResult"
    ]

    probability_columns = [
        f"{prefix}HomeProbability",
        f"{prefix}DrawProbability",
        f"{prefix}AwayProbability",
    ]

    probabilities = (
        results[
            probability_columns
        ]
        .to_numpy()
    )

    numeric_actual = actual.map(
        LABEL_MAP
    )

    return {
        "Accuracy":
            (
                actual
                ==
                predicted
            ).mean(),

        "LogLoss":
            log_loss(
                numeric_actual,
                probabilities,
                labels=[0, 1, 2],
            ),

        "Brier":
            multiclass_brier(
                actual,
                probabilities,
            ),
    }


# --------------------------------------------------
# Load source data
# --------------------------------------------------

print()
print("FOOTBALL COPILOT")
print("H3 UNDERLYING FORM DIAGNOSTIC")
print("=============================")

matches = pd.read_csv(
    INPUT_FILE
)

matches["Date"] = pd.to_datetime(
    matches["Date"],
    errors="coerce",
)

matches = matches.sort_values(
    ["Date", "HomeTeam", "AwayTeam"]
).reset_index(
    drop=True
)


# --------------------------------------------------
# Build leakage-safe features
# --------------------------------------------------

print()
print("Building leakage-safe features...")

feature_rows = []


for _, match in matches.iterrows():

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

    home_last_5 = get_previous_matches(
        matches,
        home_team,
        current_date,
        n=5,
    )

    away_last_5 = get_previous_matches(
        matches,
        away_team,
        current_date,
        n=5,
    )

    home_last_10 = get_previous_matches(
        matches,
        home_team,
        current_date,
        n=10,
    )

    away_last_10 = get_previous_matches(
        matches,
        away_team,
        current_date,
        n=10,
    )

    home_recent_home = get_previous_matches(
        matches,
        home_team,
        current_date,
        n=5,
        venue="Home",
    )

    away_recent_away = get_previous_matches(
        matches,
        away_team,
        current_date,
        n=5,
        venue="Away",
    )

    home_season_matches = get_previous_matches(
        matches,
        home_team,
        current_date,
        season=season,
    )

    away_season_matches = get_previous_matches(
        matches,
        away_team,
        current_date,
        season=season,
    )

    home_5 = calculate_team_stats(
        home_last_5,
        home_team,
    )

    away_5 = calculate_team_stats(
        away_last_5,
        away_team,
    )

    home_10 = calculate_team_stats(
        home_last_10,
        home_team,
    )

    away_10 = calculate_team_stats(
        away_last_10,
        away_team,
    )

    home_venue = calculate_team_stats(
        home_recent_home,
        home_team,
    )

    away_venue = calculate_team_stats(
        away_recent_away,
        away_team,
    )

    home_season = calculate_team_stats(
        home_season_matches,
        home_team,
    )

    away_season = calculate_team_stats(
        away_season_matches,
        away_team,
    )

    home_shots = calculate_shot_stats(
        home_last_5,
        home_team,
    )

    away_shots = calculate_shot_stats(
        away_last_5,
        away_team,
    )

    row = {

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

        # Model 2

        "HomeRecentGoalsFor":
            home_5["goals_for_pg"],

        "HomeRecentGoalsAgainst":
            home_5["goals_against_pg"],

        "HomeRecentPPG":
            home_5["ppg"],

        "AwayRecentGoalsFor":
            away_5["goals_for_pg"],

        "AwayRecentGoalsAgainst":
            away_5["goals_against_pg"],

        "AwayRecentPPG":
            away_5["ppg"],

        "Home10GoalsFor":
            home_10["goals_for_pg"],

        "Home10GoalsAgainst":
            home_10["goals_against_pg"],

        "Home10PPG":
            home_10["ppg"],

        "Away10GoalsFor":
            away_10["goals_for_pg"],

        "Away10GoalsAgainst":
            away_10["goals_against_pg"],

        "Away10PPG":
            away_10["ppg"],

        "HomeVenuePPG":
            home_venue["ppg"],

        "HomeVenueGoalsFor":
            home_venue["goals_for_pg"],

        "AwayVenuePPG":
            away_venue["ppg"],

        "AwayVenueGoalsFor":
            away_venue["goals_for_pg"],

        "HomeSeasonPPG":
            home_season["ppg"],

        "HomeSeasonGoalDifferencePG":
            home_season[
                "goal_difference_pg"
            ],

        "AwaySeasonPPG":
            away_season["ppg"],

        "AwaySeasonGoalDifferencePG":
            away_season[
                "goal_difference_pg"
            ],

        "RecentPPGDifference":
            (
                home_5["ppg"]
                -
                away_5["ppg"]
            ),

        "TenMatchPPGDifference":
            (
                home_10["ppg"]
                -
                away_10["ppg"]
            ),

        "SeasonPPGDifference":
            (
                home_season["ppg"]
                -
                away_season["ppg"]
            ),

        "AttackVsDefenceHome":
            (
                home_10["goals_for_pg"]
                -
                away_10[
                    "goals_against_pg"
                ]
            ),

        "AttackVsDefenceAway":
            (
                away_10["goals_for_pg"]
                -
                home_10[
                    "goals_against_pg"
                ]
            ),

        # H3 underlying performance

        "HomeRecentShotsFor":
            home_shots["shots_for_pg"],

        "HomeRecentShotsAgainst":
            home_shots[
                "shots_against_pg"
            ],

        "HomeRecentSOTFor":
            home_shots["sot_for_pg"],

        "HomeRecentSOTAgainst":
            home_shots["sot_against_pg"],

        "HomeRecentSOTRate":
            home_shots["sot_rate"],

        "AwayRecentShotsFor":
            away_shots["shots_for_pg"],

        "AwayRecentShotsAgainst":
            away_shots[
                "shots_against_pg"
            ],

        "AwayRecentSOTFor":
            away_shots["sot_for_pg"],

        "AwayRecentSOTAgainst":
            away_shots["sot_against_pg"],

        "AwayRecentSOTRate":
            away_shots["sot_rate"],

        "RecentShotsDifference":
            (
                home_shots["shots_for_pg"]
                -
                away_shots["shots_for_pg"]
            ),

        "RecentSOTDifference":
            (
                home_shots["sot_for_pg"]
                -
                away_shots["sot_for_pg"]
            ),

        "HomeAttackSOTVsAwayDefence":
            (
                home_shots["sot_for_pg"]
                -
                away_shots["sot_against_pg"]
            ),

        "AwayAttackSOTVsHomeDefence":
            (
                away_shots["sot_for_pg"]
                -
                home_shots["sot_against_pg"]
            ),

        "HomeHistoryGames10":
            home_10["games"],

        "AwayHistoryGames10":
            away_10["games"],
    }

    feature_rows.append(
        row
    )


df = pd.DataFrame(
    feature_rows
)


# --------------------------------------------------
# Same history requirement as Model 2
# --------------------------------------------------

df = df[
    (
        df["HomeHistoryGames10"]
        >= 10
    )
    &
    (
        df["AwayHistoryGames10"]
        >= 10
    )
].copy()

df = df.sort_values(
    "Date"
).reset_index(
    drop=True
)


# --------------------------------------------------
# Walk-forward
# --------------------------------------------------

SEASONS = sorted(
    df["Season"].unique()
)

print()
print(
    "Seasons:",
    SEASONS,
)

all_results = []


for test_index in range(
    2,
    len(SEASONS)
):

    test_season = (
        SEASONS[
            test_index
        ]
    )

    training_seasons = (
        SEASONS[
            :test_index
        ]
    )

    train = df[
        df["Season"].isin(
            training_seasons
        )
    ].copy()

    test = df[
        df["Season"]
        ==
        test_season
    ].copy()

    print()
    print(
        "Training on:",
        ", ".join(
            training_seasons
        ),
    )

    print(
        "Testing on:",
        test_season,
    )

    models = {}

    for model_name, features in {
        "M2": MODEL_2_FEATURES,
        "H3": MODEL_H3_FEATURES,
    }.items():

        home_model = PoissonRegressor(
            alpha=0.1,
            max_iter=1000,
        )

        away_model = PoissonRegressor(
            alpha=0.1,
            max_iter=1000,
        )

        home_model.fit(
            train[features],
            train["HomeGoals"],
        )

        away_model.fit(
            train[features],
            train["AwayGoals"],
        )

        models[
            model_name
        ] = {
            "home": home_model,
            "away": away_model,
            "features": features,
        }

    test = test.reset_index(
        drop=True
    )

    predictions = {}

    for model_name, model in models.items():

        features = model[
            "features"
        ]

        predictions[
            model_name
        ] = (
            model["home"].predict(
                test[features]
            ),
            model["away"].predict(
                test[features]
            ),
        )

    for i in range(
        len(test)
    ):

        actual = actual_result(
            test.loc[i, "HomeGoals"],
            test.loc[i, "AwayGoals"],
        )

        row_result = {
            "Season":
                test_season,

            "Date":
                test.loc[i, "Date"],

            "HomeTeam":
                test.loc[i, "HomeTeam"],

            "AwayTeam":
                test.loc[i, "AwayTeam"],

            "ActualResult":
                actual,
        }

        for model_name in [
            "M2",
            "H3",
        ]:

            home_xg = (
                predictions[
                    model_name
                ][0][i]
            )

            away_xg = (
                predictions[
                    model_name
                ][1][i]
            )

            (
                home_probability,
                draw_probability,
                away_probability,
            ) = match_probabilities(
                home_xg,
                away_xg,
            )

            row_result[
                f"{model_name}HomeXG"
            ] = home_xg

            row_result[
                f"{model_name}AwayXG"
            ] = away_xg

            row_result[
                f"{model_name}HomeProbability"
            ] = home_probability

            row_result[
                f"{model_name}DrawProbability"
            ] = draw_probability

            row_result[
                f"{model_name}AwayProbability"
            ] = away_probability

            row_result[
                f"{model_name}PredictedResult"
            ] = predict_result(
                home_probability,
                draw_probability,
                away_probability,
            )

        all_results.append(
            row_result
        )


results = pd.DataFrame(
    all_results
)


# --------------------------------------------------
# Overall comparison
# --------------------------------------------------

summary_rows = []


for name, prefix in [
    ("Model 2", "M2"),
    ("H3 Shots/SOT", "H3"),
]:

    metrics = calculate_metrics(
        results,
        prefix,
    )

    predicted_draws = (
        results[
            f"{prefix}PredictedResult"
        ]
        ==
        "D"
    ).sum()

    summary_rows.append(
        {
            "Model":
                name,

            "Matches":
                len(results),

            "AccuracyPct":
                metrics["Accuracy"]
                * 100,

            "LogLoss":
                metrics["LogLoss"],

            "Brier":
                metrics["Brier"],

            "PredictedDraws":
                predicted_draws,
        }
    )


summary = pd.DataFrame(
    summary_rows
)


print()
print("OVERALL H3 COMPARISON")
print("=====================")

print(
    summary.to_string(
        index=False,
        formatters={
            "AccuracyPct":
                "{:.4f}".format,
            "LogLoss":
                "{:.4f}".format,
            "Brier":
                "{:.4f}".format,
        },
    )
)


# --------------------------------------------------
# Seasonal comparison
# --------------------------------------------------

season_rows = []


for season in sorted(
    results["Season"].unique()
):

    season_data = results[
        results["Season"]
        ==
        season
    ].copy()

    for name, prefix in [
        ("Model 2", "M2"),
        ("H3 Shots/SOT", "H3"),
    ]:

        metrics = calculate_metrics(
            season_data,
            prefix,
        )

        season_rows.append(
            {
                "Season":
                    season,

                "Model":
                    name,

                "Matches":
                    len(season_data),

                "AccuracyPct":
                    metrics["Accuracy"]
                    * 100,

                "LogLoss":
                    metrics["LogLoss"],

                "Brier":
                    metrics["Brier"],
            }
        )


season_summary = pd.DataFrame(
    season_rows
)


print()
print("H3 COMPARISON BY SEASON")
print("=======================")

print(
    season_summary.to_string(
        index=False,
        formatters={
            "AccuracyPct":
                "{:.4f}".format,
            "LogLoss":
                "{:.4f}".format,
            "Brier":
                "{:.4f}".format,
        },
    )
)


# --------------------------------------------------
# Change vs Model 2
# --------------------------------------------------

m2 = summary[
    summary["Model"]
    ==
    "Model 2"
].iloc[0]

h3 = summary[
    summary["Model"]
    ==
    "H3 Shots/SOT"
].iloc[0]


print()
print("H3 CHANGE VS MODEL 2")
print("====================")

print(
    "Accuracy change: "
    f"{(
        h3['AccuracyPct']
        -
        m2['AccuracyPct']
    ):+.4f} percentage points"
)

print(
    "Log loss change: "
    f"{(
        h3['LogLoss']
        -
        m2['LogLoss']
    ):+.4f}"
)

print(
    "Brier change: "
    f"{(
        h3['Brier']
        -
        m2['Brier']
    ):+.4f}"
)

print(
    "Predicted draw change: "
    f"{int(
        h3['PredictedDraws']
        -
        m2['PredictedDraws']
    ):+d}"
)


# --------------------------------------------------
# Save
# --------------------------------------------------

Path(
    "reports/post_gw5"
).mkdir(
    parents=True,
    exist_ok=True,
)


results.to_csv(
    DETAIL_OUTPUT,
    index=False,
)

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)

season_summary.to_csv(
    SEASON_OUTPUT,
    index=False,
)


print()
print("FILES SAVED")
print("===========")

print(
    DETAIL_OUTPUT
)

print(
    SUMMARY_OUTPUT
)

print(
    SEASON_OUTPUT
)

print()
print("H3 UNDERLYING FORM DIAGNOSTIC COMPLETE")
print("======================================")