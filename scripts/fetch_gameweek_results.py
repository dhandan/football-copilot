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

# The football-data.org key is required only for legacy-source fixtures.


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
# PROVIDER-AWARE RESULT RETRIEVAL
# ==================================================

import json

sources = (
    predictions["FixtureIdSource"].fillna("football-data.org")
    if "FixtureIdSource" in predictions.columns
    else pd.Series("football-data.org", index=predictions.index)
)

if any(src != "FPL" for src in sources) and not API_KEY:
    raise RuntimeError("FOOTBALL_DATA_API_KEY is required for legacy fixtures.")

fpl_live = {}
fpl_original = {}
fpl_team_names = {}

if any(src == "FPL" for src in sources):
    snapshot_root = PROJECT_ROOT / "data/live/players/h07_snapshots"
    snapshots = sorted(
        (
            folder for folder in snapshot_root.iterdir()
            if folder.is_dir()
            and (folder / "fixtures.json").exists()
            and (folder / "bootstrap.json").exists()
        ),
        reverse=True,
    )
    if not snapshots:
        raise RuntimeError("Archived FPL fixture snapshot not found.")

    snapshot = snapshots[0]
    archived = json.loads(
        (snapshot / "fixtures.json").read_text(encoding="utf-8")
    )
    bootstrap = json.loads(
        (snapshot / "bootstrap.json").read_text(encoding="utf-8")
    )
    fpl_original = {
        int(f["id"]): f for f in archived if f.get("event") == gameweek
    }
    fpl_team_names = {
        int(t["id"]): t["name"] for t in bootstrap["teams"]
    }

    response = requests.get(
        "https://fantasy.premierleague.com/api/fixtures/",
        timeout=40,
        headers={"User-Agent": "FootballCopilot/1.0"},
    )
    response.raise_for_status()
    fpl_live = {
        int(f["id"]): f
        for f in response.json()
        if f.get("event") == gameweek
    }

    if len(fpl_original) != 10 or len(fpl_live) != 10:
        raise RuntimeError("Expected exactly ten archived and live FPL fixtures.")


def fetch_provider_match(fixture_id, source):
    if source != "FPL":
        if source not in ("football-data.org", "nan", ""):
            raise RuntimeError(f"Unsupported fixture ID source: {source}")
        return fetch_match(fixture_id)

    before = fpl_original.get(fixture_id)
    current = fpl_live.get(fixture_id)
    if before is None or current is None:
        raise RuntimeError(f"Missing FPL fixture {fixture_id}.")

    for key in ("id", "event", "team_h", "team_a", "kickoff_time"):
        if before.get(key) != current.get(key):
            raise RuntimeError(
                f"FPL fixture {fixture_id} identity mismatch: {key}."
            )

    if current.get("finished") is not True:
        raise RuntimeError(f"FPL fixture {fixture_id} is not finished.")

    home = current.get("team_h_score")
    away = current.get("team_a_score")
    if home is None or away is None:
        raise RuntimeError(f"FPL fixture {fixture_id} has no final score.")

    winner = (
        "HOME_TEAM" if home > away
        else "AWAY_TEAM" if away > home
        else "DRAW"
    )

    return {
        "status": "FINISHED",
        "homeTeam": {
            "name": fpl_team_names[int(current["team_h"])]
        },
        "awayTeam": {
            "name": fpl_team_names[int(current["team_a"])]
        },
        "score": {
            "fullTime": {"home": int(home), "away": int(away)},
            "winner": winner,
        },
        "lastUpdated": current.get("kickoff_time"),
    }


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


    source = str(prediction.get("FixtureIdSource", "football-data.org"))
    match = fetch_provider_match(fixture_id, source)


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