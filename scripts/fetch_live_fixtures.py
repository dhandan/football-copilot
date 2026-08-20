from pathlib import Path
from datetime import datetime
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
    PROJECT_ROOT / ".env"
)


API_KEY = os.getenv(
    "FOOTBALL_DATA_API_KEY"
)


if not API_KEY:

    raise RuntimeError(
        "FOOTBALL_DATA_API_KEY is not set in .env"
    )


# ==================================================
# CONFIG
# ==================================================

SEASON = "2026/27"

COMPETITION_CODE = "PL"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "fixtures"
)


# ==================================================
# TEAM NAME MAPPING
#
# API names → Football Copilot names
# ==================================================

TEAM_NAME_MAP = {

    "Manchester United FC":
        "Man United",

    "Manchester City FC":
        "Man City",

    "Nottingham Forest FC":
        "Nott'm Forest",

    "Tottenham Hotspur FC":
        "Tottenham",

    "Newcastle United FC":
        "Newcastle",

    "Brighton & Hove Albion FC":
        "Brighton",

    "Wolverhampton Wanderers FC":
        "Wolves",

    "West Ham United FC":
        "West Ham",

    "AFC Bournemouth":
        "Bournemouth",

    "Liverpool FC":
        "Liverpool",

    "Arsenal FC":
        "Arsenal",

    "Chelsea FC":
        "Chelsea",

    "Everton FC":
        "Everton",

    "Fulham FC":
        "Fulham",

    "Brentford FC":
        "Brentford",

    "Aston Villa FC":
        "Aston Villa",

    "Crystal Palace FC":
        "Crystal Palace",

    "Leeds United FC":
        "Leeds",

    "Sunderland AFC":
        "Sunderland",

    "Ipswich Town FC":
        "Ipswich",
}


def normalise_team_name(
    api_name
):

    return TEAM_NAME_MAP.get(
        api_name,
        api_name
        .replace(" FC", "")
        .replace(" AFC", "")
    )


# ==================================================
# FETCH FIXTURES
# ==================================================

def fetch_fixtures(
    date_from,
    date_to,
):

    url = (
        "https://api.football-data.org/"
        f"v4/competitions/{COMPETITION_CODE}/matches"
    )


    headers = {
        "X-Auth-Token":
            API_KEY
    }


    params = {
        "dateFrom":
            date_from,

        "dateTo":
            date_to,
    }


    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
    )


    response.raise_for_status()


    return response.json()


# ==================================================
# CONVERT TO DATAFRAME
# ==================================================

def convert_matches(
    response_data
):

    rows = []


    for match in response_data.get(
        "matches",
        []
    ):

        utc_date = pd.to_datetime(
            match[
                "utcDate"
            ],
            utc=True,
        )


        uk_date = utc_date.tz_convert(
            "Europe/London"
        )


        rows.append(
            {

                "Season":
                    SEASON,

                "FixtureId":
                    match[
                        "id"
                    ],

                "Matchday":
                    match.get(
                        "matchday"
                    ),

                "FixtureDate":
                    uk_date.strftime(
                        "%Y-%m-%d"
                    ),

                "FixtureTime":
                    uk_date.strftime(
                        "%H:%M"
                    ),

                "HomeTeam":
                    normalise_team_name(
                        match[
                            "homeTeam"
                        ][
                            "name"
                        ]
                    ),

                "AwayTeam":
                    normalise_team_name(
                        match[
                            "awayTeam"
                        ][
                            "name"
                        ]
                    ),

                "Status":
                    match.get(
                        "status"
                    ),

                "LastUpdated":
                    match.get(
                        "lastUpdated"
                    ),
            }
        )


    return pd.DataFrame(
        rows
    )


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    print()
    print("FOOTBALL COPILOT")
    print("LIVE FIXTURE INGESTION")
    print("======================")


    date_from = input(
        "Start date (YYYY-MM-DD): "
    ).strip()


    date_to = input(
        "End date (YYYY-MM-DD): "
    ).strip()


    data = fetch_fixtures(
        date_from,
        date_to,
    )


    fixtures = convert_matches(
        data
    )


    if fixtures.empty:

        print()
        print(
            "No Premier League fixtures found "
            "for that date range."
        )

        raise SystemExit


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    timestamp = (
        datetime.now()
        .astimezone()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )


    output_file = (
        OUTPUT_DIR
        /
        (
            f"fixtures_"
            f"{date_from}_"
            f"{date_to}_"
            f"{timestamp}.csv"
        )
    )


    fixtures.to_csv(
        output_file,
        index=False,
    )


    print()
    print(
        fixtures.to_string(
            index=False
        )
    )


    print()
    print(
        f"Saved to: "
        f"{output_file}"
    )