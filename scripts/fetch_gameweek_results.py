from pathlib import Path
from datetime import datetime
import argparse
import os

import pandas as pd
import requests
from dotenv import load_dotenv


# ==================================================
# PROJECT SETUP
# ==================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

load_dotenv(
    PROJECT_ROOT
    / ".env"
)

API_KEY = os.getenv(
    "FOOTBALL_DATA_API_KEY"
)

if not API_KEY:

    raise RuntimeError(
        "FOOTBALL_DATA_API_KEY is not set in .env"
    )


# ==================================================
# PATHS
# ==================================================

PREDICTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "predictions"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "results"
)


# ==================================================
# ARGUMENTS
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--gameweek",
    type=int,
    required=True,
    help="Premier League gameweek number.",
)

args = parser.parse_args()

gameweek = args.gameweek


# ==================================================
# LOCATE FROZEN PREDICTION SNAPSHOT
# ==================================================

prediction_file = (
    PREDICTION_DIR
    /
    f"2026_27_gw{gameweek:02d}_predictions.csv"
)


if not prediction_file.exists():

    raise FileNotFoundError(
        "Frozen prediction snapshot not found: "
        f"{prediction_file}"
    )


predictions = pd.read_csv(
    prediction_file
)


if predictions.empty:

    raise ValueError(
        "Prediction snapshot is empty."
    )


# ==================================================
# OUTPUT SETUP
# ==================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


output_file = (
    RESULTS_DIR
    /
    f"2026_27_gw{gameweek:02d}_results.csv"
)


if output_file.exists():

    print()
    print(
        "RESULTS SNAPSHOT ALREADY EXISTS"
    )

    print(
        "==============================="
    )

    print()
    print(
        output_file
    )

    print()
    print(
        "The file has NOT been overwritten."
    )

    raise SystemExit(1)


# ==================================================
# API CONFIGURATION
# ==================================================

HEADERS = {
    "X-Auth-Token":
        API_KEY
}


def fetch_match(
    fixture_id,
):

    url = (
        "https://api.football-data.org/"
        f"v4/matches/{fixture_id}"
    )


    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
    )


    response.raise_for_status()


    return response.json()


# ==================================================
# FETCH RESULTS
# ==================================================

print()
print("FOOTBALL COPILOT")
print(
    f"2026/27 GAMEWEEK {gameweek} "
    "RESULT INGESTION"
)
print("=" * 50)


rows = []


for _, prediction in (
    predictions.iterrows()
):

    fixture_id = int(
        prediction[
            "FixtureId"
        ]
    )

    expected_home_team = (
        prediction[
            "HomeTeam"
        ]
    )

    expected_away_team = (
        prediction[
            "AwayTeam"
        ]
    )


    print()
    print(
        f"Fetching FixtureId "
        f"{fixture_id}..."
    )


    match = fetch_match(
        fixture_id
    )


    status = match.get(
        "status"
    )


    api_home_team = (
        match.get(
            "homeTeam",
            {},
        )
        .get(
            "name"
        )
    )

    api_away_team = (
        match.get(
            "awayTeam",
            {},
        )
        .get(
            "name"
        )
    )


    score = match.get(
        "score",
        {},
    )


    full_time = score.get(
        "fullTime",
        {},
    )


    home_goals = (
        full_time.get(
            "home"
        )
    )

    away_goals = (
        full_time.get(
            "away"
        )
    )


    winner = score.get(
        "winner"
    )


    if status != "FINISHED":

        raise RuntimeError(
            f"Fixture {fixture_id} is not "
            f"finished. Status: {status}"
        )


    if (
        home_goals is None
        or
        away_goals is None
    ):

        raise RuntimeError(
            f"Fixture {fixture_id} has no "
            "full-time score."
        )


    if winner == "HOME_TEAM":

        actual_result = (
            expected_home_team
        )

        result_code = "H"


    elif winner == "AWAY_TEAM":

        actual_result = (
            expected_away_team
        )

        result_code = "A"


    elif winner == "DRAW":

        actual_result = "Draw"

        result_code = "D"


    else:

        raise RuntimeError(
            f"Unknown winner value for "
            f"{fixture_id}: {winner}"
        )


    actual_score = (
        f"{int(home_goals)}-"
        f"{int(away_goals)}"
    )


    print(
        f"{expected_home_team} "
        f"{int(home_goals)}-"
        f"{int(away_goals)} "
        f"{expected_away_team}"
    )


    rows.append(
        {

            "ResultTimestamp":
                datetime.now()
                .astimezone()
                .isoformat(
                    timespec="seconds"
                ),

            "Season":
                prediction[
                    "Season"
                ],

            "Gameweek":
                gameweek,

            "FixtureId":
                fixture_id,

            "FixtureDate":
                prediction[
                    "FixtureDate"
                ],

            "HomeTeam":
                expected_home_team,

            "AwayTeam":
                expected_away_team,

            "APIHomeTeam":
                api_home_team,

            "APIAwayTeam":
                api_away_team,

            "Status":
                status,

            "ActualHomeGoals":
                int(
                    home_goals
                ),

            "ActualAwayGoals":
                int(
                    away_goals
                ),

            "ActualScore":
                actual_score,

            "ActualResult":
                actual_result,

            "ActualResultCode":
                result_code,

            "APIWinner":
                winner,

            "LastUpdated":
                match.get(
                    "lastUpdated"
                ),
        }
    )


# ==================================================
# SAVE RESULTS SNAPSHOT
# ==================================================

results = pd.DataFrame(
    rows
)


if len(results) != len(predictions):

    raise RuntimeError(
        "Result count does not match "
        "prediction count."
    )


results.to_csv(
    output_file,
    index=False,
)


# ==================================================
# DISPLAY
# ==================================================

print()
print("=" * 50)
print("GW RESULT SNAPSHOT")
print("=" * 50)

print()

print(
    results[
        [
            "HomeTeam",
            "ActualHomeGoals",
            "ActualAwayGoals",
            "AwayTeam",
            "ActualResult",
        ]
    ]
    .to_string(
        index=False
    )
)


print()
print(
    f"Fixtures captured: "
    f"{len(results)}"
)

print()
print(
    f"Saved to: "
    f"{output_file}"
)