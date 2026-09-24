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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("ODDS_API_KEY")

if not API_KEY:
    raise RuntimeError("ODDS_API_KEY is not set in .env")


# ==================================================
# CONFIG
# ==================================================

SEASON = "2026/27"
SPORT = "soccer_epl"
REGION = "uk"
MARKET = "h2h"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "market"
)

TEAM_NAME_MAP = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham",
    "Newcastle United": "Newcastle",
    "Brighton and Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Leeds United": "Leeds",
    "Ipswich Town": "Ipswich",
}


def normalise_team_name(name):
    return TEAM_NAME_MAP.get(name, name)


# ==================================================
# FETCH ODDS
# ==================================================

def fetch_odds(date_from, date_to):

    url = (
        f"https://api.the-odds-api.com/v4/"
        f"sports/{SPORT}/odds/"
    )

    params = {
        "apiKey": API_KEY,
        "regions": REGION,
        "markets": MARKET,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "commenceTimeFrom": f"{date_from}T00:00:00Z",
        "commenceTimeTo": f"{date_to}T23:59:59Z",
    }

    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    print(
        "API requests remaining:",
        response.headers.get("x-requests-remaining"),
    )

    return response.json()


# ==================================================
# BUILD CONSENSUS
# ==================================================

def build_consensus(data, gameweek):

    rows = []

    snapshot_time = (
        datetime.now()
        .astimezone()
        .isoformat(timespec="seconds")
    )

    for event in data:

        api_home = event["home_team"]
        api_away = event["away_team"]

        home_team = normalise_team_name(api_home)
        away_team = normalise_team_name(api_away)

        home_odds = []
        draw_odds = []
        away_odds = []

        bookmakers_used = []

        for bookmaker in event.get("bookmakers", []):

            h2h_market = next(
                (
                    market
                    for market in bookmaker.get("markets", [])
                    if market.get("key") == "h2h"
                ),
                None,
            )

            if not h2h_market:
                continue

            prices = {
                outcome["name"]: outcome["price"]
                for outcome in h2h_market.get("outcomes", [])
            }

            if (
                api_home not in prices
                or api_away not in prices
                or "Draw" not in prices
            ):
                continue

            home_odds.append(float(prices[api_home]))
            draw_odds.append(float(prices["Draw"]))
            away_odds.append(float(prices[api_away]))

            bookmakers_used.append(bookmaker["title"])

        if not home_odds:
            raise RuntimeError(
                f"No complete H/D/A bookmaker odds for "
                f"{api_home} vs {api_away}"
            )

        avg_h = sum(home_odds) / len(home_odds)
        avg_d = sum(draw_odds) / len(draw_odds)
        avg_a = sum(away_odds) / len(away_odds)

        raw_h = 1.0 / avg_h
        raw_d = 1.0 / avg_d
        raw_a = 1.0 / avg_a

        overround = raw_h + raw_d + raw_a

        fair_h = raw_h / overround
        fair_d = raw_d / overround
        fair_a = raw_a / overround

        kickoff_utc = pd.to_datetime(
            event["commence_time"],
            utc=True,
        )

        kickoff_uk = kickoff_utc.tz_convert(
            "Europe/London"
        )

        rows.append(
            {
                "Season": SEASON,
                "Gameweek": gameweek,
                "OddsEventId": event["id"],
                "FixtureDate": kickoff_uk.strftime("%Y-%m-%d"),
                "FixtureTime": kickoff_uk.strftime("%H:%M"),
                "HomeTeam": home_team,
                "AwayTeam": away_team,
                "SnapshotTimestamp": snapshot_time,
                "MarketSource": "The Odds API",
                "Region": REGION,
                "Market": MARKET,
                "BookmakerCount": len(bookmakers_used),
                "Bookmakers": " | ".join(bookmakers_used),
                "AvgH": avg_h,
                "AvgD": avg_d,
                "AvgA": avg_a,
                "Overround": overround,
                "MarketHomeProbability": fair_h,
                "MarketDrawProbability": fair_d,
                "MarketAwayProbability": fair_a,
            }
        )

    return pd.DataFrame(rows)


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gameweek",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--date-from",
        required=True,
    )

    parser.add_argument(
        "--date-to",
        required=True,
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Create a test snapshot rather than an official snapshot.",
    )

    args = parser.parse_args()

    print()
    print("FOOTBALL COPILOT")
    print("OPENING MARKET SNAPSHOT")
    print("=======================")
    print()

    data = fetch_odds(
        args.date_from,
        args.date_to,
    )

    market = build_consensus(
        data,
        args.gameweek,
    )

    market = market.sort_values(
        ["FixtureDate", "FixtureTime"]
    ).reset_index(drop=True)

    if len(market) != 10:
        raise RuntimeError(
            f"Expected exactly 10 fixtures for GW{args.gameweek}, "
            f"but received {len(market)}. Snapshot NOT saved."
        )

    probability_sum = (
        market["MarketHomeProbability"]
        + market["MarketDrawProbability"]
        + market["MarketAwayProbability"]
    )

    if not probability_sum.between(
        0.999999,
        1.000001,
    ).all():
        raise RuntimeError(
            "Market probabilities do not sum to 1."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_type = (
        "test"
        if args.test
        else "official"
    )

    timestamp = (
        datetime.now()
        .astimezone()
        .strftime("%Y%m%d_%H%M%S")
    )

    output_file = (
        OUTPUT_DIR
        / (
            f"2026_27_gw{args.gameweek:02d}_"
            f"opening_market_{snapshot_type}_"
            f"{timestamp}.csv"
        )
    )

    market.to_csv(
        output_file,
        index=False,
    )

    display_columns = [
        "FixtureDate",
        "FixtureTime",
        "HomeTeam",
        "AwayTeam",
        "BookmakerCount",
        "AvgH",
        "AvgD",
        "AvgA",
        "MarketHomeProbability",
        "MarketDrawProbability",
        "MarketAwayProbability",
    ]

    print(
        market[display_columns]
        .to_string(index=False)
    )

    print()
    print(
        f"Fixtures: {len(market)}"
    )

    print(
        "Bookmaker coverage:",
        f"min={market['BookmakerCount'].min()},",
        f"max={market['BookmakerCount'].max()},",
        f"mean={market['BookmakerCount'].mean():.1f}",
    )

    print()
    print(
        f"Snapshot type: {snapshot_type.upper()}"
    )

    print(
        f"Saved to: {output_file}"
    )

    if args.test:
        print()
        print(
            "TEST ONLY - this is NOT the frozen "
            "prospective GW snapshot."
        )