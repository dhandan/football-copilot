from pathlib import Path
from datetime import datetime
import argparse
import sys

import pandas as pd


# ==================================================
# PROJECT SETUP
# ==================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.append(
    str(PROJECT_ROOT)
)


from prediction.fixture_predictor import predict_fixture


# ==================================================
# PATHS
# ==================================================

FIXTURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "fixtures"
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "predictions"
)


# ==================================================
# COMMAND-LINE ARGUMENTS
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dry-run",
    action="store_true",
    help=(
        "Generate predictions without saving "
        "the official snapshot."
    ),
)

args = parser.parse_args()


# ==================================================
# HELPER
# ==================================================

def get_predicted_outcome(
    home_team,
    away_team,
    home_probability,
    draw_probability,
    away_probability,
):

    probabilities = {
        home_team:
            home_probability,

        "Draw":
            draw_probability,

        away_team:
            away_probability,
    }

    return max(
        probabilities,
        key=probabilities.get,
    )


# ==================================================
# FIND LATEST FIXTURE SNAPSHOT
# ==================================================

fixture_files = sorted(
    FIXTURE_DIR.glob(
        "fixtures_*.csv"
    ),
    key=lambda path:
        path.stat().st_mtime,
    reverse=True,
)


if not fixture_files:

    raise FileNotFoundError(
        "No live fixture snapshot found."
    )


fixture_file = fixture_files[0]


# ==================================================
# LOAD FIXTURES
# ==================================================

fixtures = pd.read_csv(
    fixture_file
)


if fixtures.empty:

    raise ValueError(
        "Fixture snapshot is empty."
    )


# ==================================================
# VALIDATE GAMEWEEK
# ==================================================

matchdays = (
    fixtures[
        "Matchday"
    ]
    .dropna()
    .unique()
)


if len(matchdays) != 1:

    raise ValueError(
        "Fixture snapshot contains more than one "
        f"matchday: {matchdays}"
    )


gameweek = int(
    matchdays[0]
)


season = str(
    fixtures.iloc[0][
        "Season"
    ]
)


season_slug = (
    season
    .replace("/", "_")
)


# ==================================================
# OUTPUT FILE
# ==================================================

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


output_file = (
    PREDICTION_DIR
    /
    (
        f"{season_slug}_gw"
        f"{gameweek:02d}_predictions.csv"
    )
)


# ==================================================
# PROTECT EXISTING OFFICIAL SNAPSHOT
# ==================================================

if (
    output_file.exists()
    and
    not args.dry_run
):

    print()
    print(
        "OFFICIAL PREDICTION SNAPSHOT "
        "ALREADY EXISTS"
    )

    print(
        "=" * 50
    )

    print()
    print(
        output_file
    )

    print()
    print(
        "The file has NOT been overwritten."
    )

    print(
        "This protects the integrity of the "
        "pre-match experiment."
    )

    raise SystemExit(1)


# ==================================================
# TIMESTAMP
# ==================================================

prediction_timestamp = (
    datetime.now()
    .astimezone()
    .isoformat(
        timespec="seconds"
    )
)


# ==================================================
# START
# ==================================================

print()
print("FOOTBALL COPILOT")


if args.dry_run:

    print(
        f"{season} GAMEWEEK {gameweek} "
        "DRY RUN"
    )

else:

    print(
        f"{season} GAMEWEEK {gameweek} "
        "OFFICIAL PREDICTIONS"
    )


print(
    "=" * 60
)

print()
print(
    f"Fixture snapshot: "
    f"{fixture_file.name}"
)

print(
    f"Prediction timestamp: "
    f"{prediction_timestamp}"
)

print(
    f"Mode: "
    f"{'DRY RUN' if args.dry_run else 'OFFICIAL'}"
)


# ==================================================
# GENERATE PREDICTIONS
# ==================================================

rows = []


for _, fixture in fixtures.iterrows():

    fixture_id = (
        fixture[
            "FixtureId"
        ]
    )

    home_team = (
        fixture[
            "HomeTeam"
        ]
    )

    away_team = (
        fixture[
            "AwayTeam"
        ]
    )


    print()
    print(
        f"{home_team} vs {away_team}"
    )


    try:

        result = predict_fixture(
            home_team,
            away_team,
        )


        metadata = result.get(
            "feature_metadata",
            {}
        )


        predicted_result = (
            get_predicted_outcome(
                home_team,
                away_team,
                result[
                    "home_win_probability"
                ],
                result[
                    "draw_probability"
                ],
                result[
                    "away_win_probability"
                ],
            )
        )


        most_likely_score = (
            result[
                "most_likely_scores"
            ][0][
                "score"
            ]
            if result.get(
                "most_likely_scores"
            )
            else None
        )


        row = {

            "PredictionTimestamp":
                prediction_timestamp,

            "Season":
                season,

            "Gameweek":
                gameweek,

            "FixtureId":
                fixture_id,

            "FixtureDate":
                fixture[
                    "FixtureDate"
                ],

            "FixtureTime":
                fixture[
                    "FixtureTime"
                ],

            "HomeTeam":
                home_team,

            "AwayTeam":
                away_team,

            "ModelVersion":
                "Model2_v1.0",

            "PredictionStatus":
                "Predicted",

            "PredictedResult":
                predicted_result,

            "MostLikelyScore":
                most_likely_score,

            "HomeWinProbability":
                result[
                    "home_win_probability"
                ],

            "DrawProbability":
                result[
                    "draw_probability"
                ],

            "AwayWinProbability":
                result[
                    "away_win_probability"
                ],

            "ExpectedHomeGoals":
                result[
                    "expected_home_goals"
                ],

            "ExpectedAwayGoals":
                result[
                    "expected_away_goals"
                ],

            "ColdStartUsed":
                metadata.get(
                    "cold_start_used",
                    False,
                ),

            "ColdStartTeams":
                "|".join(
                    metadata.get(
                        "cold_start_teams",
                        []
                    )
                ),

            "ColdStartMethod":
                metadata.get(
                    "cold_start_method"
                ),

            "FeatureSeason":
                result.get(
                    "feature_season"
                ),

            "ActualHomeGoals":
                None,

            "ActualAwayGoals":
                None,

            "ActualResult":
                None,

            "Evaluated":
                False,
        }


        print(
            f"Home "
            f"{row['HomeWinProbability']:.1f}%"
            f" | Draw "
            f"{row['DrawProbability']:.1f}%"
            f" | Away "
            f"{row['AwayWinProbability']:.1f}%"
        )

        print(
            f"xG "
            f"{row['ExpectedHomeGoals']:.2f}"
            f" - "
            f"{row['ExpectedAwayGoals']:.2f}"
        )

        print(
            f"Predicted outcome: "
            f"{row['PredictedResult']}"
        )

        print(
            f"Most likely scoreline: "
            f"{row['MostLikelyScore']}"
        )


        if row[
            "ColdStartUsed"
        ]:

            print(
                "Cold-start prior used for: "
                f"{row['ColdStartTeams']}"
            )


    except Exception as error:

        row = {

            "PredictionTimestamp":
                prediction_timestamp,

            "Season":
                season,

            "Gameweek":
                gameweek,

            "FixtureId":
                fixture_id,

            "FixtureDate":
                fixture[
                    "FixtureDate"
                ],

            "FixtureTime":
                fixture[
                    "FixtureTime"
                ],

            "HomeTeam":
                home_team,

            "AwayTeam":
                away_team,

            "ModelVersion":
                "Model2_v1.0",

            "PredictionStatus":
                "Unavailable",

            "PredictionError":
                str(error),

            "PredictedResult":
                None,

            "MostLikelyScore":
                None,

            "HomeWinProbability":
                None,

            "DrawProbability":
                None,

            "AwayWinProbability":
                None,

            "ExpectedHomeGoals":
                None,

            "ExpectedAwayGoals":
                None,

            "ColdStartUsed":
                False,

            "ColdStartTeams":
                "",

            "ColdStartMethod":
                None,

            "FeatureSeason":
                None,

            "ActualHomeGoals":
                None,

            "ActualAwayGoals":
                None,

            "ActualResult":
                None,

            "Evaluated":
                False,
        }


        print(
            f"Prediction unavailable: "
            f"{error}"
        )


    rows.append(
        row
    )


# ==================================================
# CREATE OUTPUT DATAFRAME
# ==================================================

output = pd.DataFrame(
    rows
)


# ==================================================
# GAMEWEEK PREDICTION SUMMARY
# ==================================================

print()
print(
    "=" * 70
)

print(
    "GAMEWEEK PREDICTION SUMMARY"
)

print(
    "=" * 70
)

print()


for _, prediction in output.iterrows():

    if (
        prediction[
            "PredictionStatus"
        ]
        !=
        "Predicted"
    ):

        continue


    print(
        f"{prediction['HomeTeam']} "
        f"vs "
        f"{prediction['AwayTeam']}"
    )

    print(
        f"  Predicted outcome:      "
        f"{prediction['PredictedResult']}"
    )

    print(
        f"  Most likely scoreline: "
        f"{prediction['MostLikelyScore']}"
    )

    print()


# ==================================================
# SUMMARY COUNTS
# ==================================================

predicted_count = (
    output[
        "PredictionStatus"
    ]
    ==
    "Predicted"
).sum()


failed_count = (
    output[
        "PredictionStatus"
    ]
    !=
    "Predicted"
).sum()


cold_start_count = (
    output[
        "ColdStartUsed"
    ]
    ==
    True
).sum()


print(
    "=" * 70
)

print(
    "RUN SUMMARY"
)

print(
    "=" * 70
)


print(
    f"Fixtures:            "
    f"{len(output)}"
)

print(
    f"Predicted:           "
    f"{predicted_count}"
)

print(
    f"Failed:              "
    f"{failed_count}"
)

print(
    f"Cold-start fixtures: "
    f"{cold_start_count}"
)


# ==================================================
# SAVE OR DRY RUN
# ==================================================

if args.dry_run:

    print()
    print(
        "DRY RUN COMPLETE."
    )

    print(
        "No official prediction file "
        "has been saved."
    )


else:

    output.to_csv(
        output_file,
        index=False,
    )


    print()
    print(
        "OFFICIAL PREDICTION SNAPSHOT SAVED"
    )

    print(
        f"Saved to: "
        f"{output_file}"
    )

    print()
    print(
        "IMPORTANT: Do not modify or "
        "overwrite this snapshot after publication."
    )