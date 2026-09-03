import json
import re

import pandas as pd
import requests


SEASONS = [
    "2021",
    "2022",
    "2023",
    "2024",
    "2025",
]


def fetch_page(season):
    url = f"https://understat.com/league/EPL/{season}"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()

    return response.text


def extract_variable(html, variable_name):
    """
    Understat has used more than one page-data format over time.

    Try:
    1. var X = JSON.parse('...')
    2. var X = {...};
    3. var X = [...];
    """

    old_pattern = (
        rf"var\s+{re.escape(variable_name)}\s*=\s*"
        rf"JSON\.parse\('(.+?)'\)"
    )

    match = re.search(
        old_pattern,
        html,
        flags=re.DOTALL,
    )

    if match:
        raw = match.group(1)

        try:
            decoded = bytes(
                raw,
                "utf-8",
            ).decode("unicode_escape")

            return json.loads(decoded)

        except Exception:
            pass

    object_pattern = (
        rf"var\s+{re.escape(variable_name)}\s*=\s*"
        rf"(\{{.*?\}})\s*;"
    )

    match = re.search(
        object_pattern,
        html,
        flags=re.DOTALL,
    )

    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    array_pattern = (
        rf"var\s+{re.escape(variable_name)}\s*=\s*"
        rf"(\[.*?\])\s*;"
    )

    match = re.search(
        array_pattern,
        html,
        flags=re.DOTALL,
    )

    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    return None


def normalise_team_history(teams_data, season):
    rows = []

    if not isinstance(teams_data, dict):
        return pd.DataFrame()

    for team_id, team_data in teams_data.items():
        team_name = team_data.get("title")
        history = team_data.get("history", [])

        for match in history:
            rows.append(
                {
                    "season": season,
                    "team_id": team_id,
                    "team": team_name,
                    "date": match.get("date"),
                    "home_or_away": match.get("h_a"),
                    "result": match.get("result"),
                    "goals_for": match.get("scored"),
                    "goals_against": match.get("missed"),
                    "xg_for": match.get("xG"),
                    "xg_against": match.get("xGA"),
                    "ppda": match.get("ppda"),
                    "ppda_allowed": match.get("ppda_allowed"),
                    "deep": match.get("deep"),
                    "deep_allowed": match.get("deep_allowed"),
                }
            )

    return pd.DataFrame(rows)


def main():
    print("Testing current Understat Premier League extraction...")
    print()

    all_frames = []

    for season in SEASONS:
        print(f"Fetching EPL season starting {season}...")

        html = fetch_page(season)

        print(f"  HTML length: {len(html):,}")

        teams_data = extract_variable(
            html,
            "teamsData",
        )

        players_data = extract_variable(
            html,
            "playersData",
        )

        print(
            "  teamsData found:",
            teams_data is not None,
        )

        print(
            "  playersData found:",
            players_data is not None,
        )

        if teams_data is None:
            print("  Could not extract teamsData")
            print()

            # Useful diagnostic: show which Understat-style
            # variable names actually appear in the page.
            variable_names = sorted(
                set(
                    re.findall(
                        r"var\s+([A-Za-z0-9_]+)\s*=",
                        html,
                    )
                )
            )

            print(
                "  Variables found:",
                variable_names,
            )

            print()
            continue

        df = normalise_team_history(
            teams_data,
            season,
        )

        print(
            "  Team-match rows:",
            len(df),
        )

        if not df.empty:
            print(
                "  Teams:",
                df["team"].nunique(),
            )

            print(
                "  Missing xG:",
                df["xg_for"].isna().sum(),
            )

            print(
                "  Missing xGA:",
                df["xg_against"].isna().sum(),
            )

            print(
                "  Date range:",
                df["date"].min(),
                "to",
                df["date"].max(),
            )

            all_frames.append(df)

        print()

    if not all_frames:
        print("=" * 70)
        print("NO UNDERSTAT TEAM HISTORY EXTRACTED")
        print("=" * 70)
        return

    combined = pd.concat(
        all_frames,
        ignore_index=True,
    )

    print("=" * 70)
    print("COMBINED SUMMARY")
    print("=" * 70)

    print()
    print("Team-match rows by season:")

    print(
        combined
        .groupby("season")
        .size()
        .to_string()
    )

    print()
    print("Unique teams by season:")

    print(
        combined
        .groupby("season")["team"]
        .nunique()
        .to_string()
    )

    print()
    print("Missing xG values:")

    print(
        combined[
            ["xg_for", "xg_against"]
        ]
        .isna()
        .sum()
        .to_string()
    )

    print()
    print("Sample:")

    print(
        combined.head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()