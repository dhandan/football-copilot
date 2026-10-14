from pathlib import Path
import pandas as pd

ROOT = Path("experiments/h07_player_impact")
BASE_URL = "https://raw.githubusercontent.com/"
SOURCES = {
    "2021/22": BASE_URL + "vaastav/Fantasy-Premier-League/master/data/2021-22/gws/merged_gw.csv",
    **{
        f"{year}/{str(year+1)[-2:]}":
        BASE_URL +
        "imadeddine-belkat/Premier-League-Stats/main/fpl_scraper/fpl_stats/_merged/players/" +
        f"{year}-{str(year+1)[-2:]}_all_players_gw.csv"
        for year in (2022, 2023, 2024)
    }
}


SOURCES["2025/26"] = (
    BASE_URL +
    "vaastav/Fantasy-Premier-League/master/data/2025-26/gws/merged_gw.csv"
)

NAMES = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
    "Sheffield Utd": "Sheffield United",
}

METRICS = [
    "minutes", "starts", "goals_scored", "assists",
    "expected_goals", "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
]

canonical = pd.read_csv(
    "data/processed/premier_league_matches.csv",
    usecols=["Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
)
canonical["Date"] = pd.to_datetime(
    canonical["Date"], dayfirst=True
).dt.strftime("%Y-%m-%d")

outputs = []

for season, url in SOURCES.items():
    print(f"Processing {season}...", flush=True)
    d = pd.read_csv(url, low_memory=False)

    required = [
        "name", "team", "position", "element", "fixture",
        "kickoff_time", "was_home", "team_h_score",
        "team_a_score", "GW", "minutes",
        "goals_scored", "assists",
    ]
    missing = sorted(set(required) - set(d.columns))
    if missing:
        raise ValueError(f"{season} missing required fields: {missing}")

    # Exclude non-playing manager records and standardise goalkeepers.
    d["position"] = d["position"].astype(str).str.upper()

    managers = d["position"] == "AM"

    if managers.any():
        playing = pd.to_numeric(
            d.loc[managers, "minutes"], errors="coerce"
        ).fillna(0)

        if (playing != 0).any():
            raise ValueError(
                f"{season}: AM records include playing minutes"
            )

        print(f"  Excluded {int(managers.sum())} manager records")
        d = d.loc[~managers].copy()

    d["position"] = d["position"].replace({"GKP": "GK"})

    d["team"] = d["team"].replace(NAMES)
    d["Date"] = pd.to_datetime(
        d["kickoff_time"], utc=True
    ).dt.strftime("%Y-%m-%d")

    home = d[d["was_home"] == True].groupby("fixture")["team"].first()
    away = d[d["was_home"] == False].groupby("fixture")["team"].first()

    fixtures = d.groupby("fixture").agg(
        Date=("Date", "first"),
        FTHG=("team_h_score", "first"),
        FTAG=("team_a_score", "first"),
    )
    fixtures["HomeTeam"] = home
    fixtures["AwayTeam"] = away
    fixtures = fixtures.reset_index()

    expected = canonical[canonical["Season"] == season]
    joined = fixtures.merge(
        expected,
        on=["Date", "HomeTeam", "AwayTeam"],
        how="outer",
        suffixes=("_source", "_canonical"),
        indicator=True,
        validate="one_to_one",
    )

    matched = joined[joined["_merge"] == "both"]
    scores_ok = (
        (matched["FTHG_source"] == matched["FTHG_canonical"]) &
        (matched["FTAG_source"] == matched["FTAG_canonical"])
    )

    if (
        len(fixtures) != 380 or
        len(matched) != 380 or
        not scores_ok.all()
    ):
        raise ValueError(f"{season} fixture reconciliation failed")

    exact_duplicates = int(d.duplicated().sum())
    if exact_duplicates:
        print(f"  Removed {exact_duplicates} identical source duplicates")
        d = d.drop_duplicates().copy()

    if d.duplicated(["fixture", "element"]).any():
        raise ValueError(
            f"{season} conflicting duplicate player-fixture records"
        )

    for metric in METRICS:
        if metric not in d.columns:
            d[metric] = float("nan")

    d["KickoffUTC"] = pd.to_datetime(
        d["kickoff_time"], utc=True
    )

    result = d[
        ["Date", "KickoffUTC", "fixture", "GW", "element", "name",
         "team", "position", "was_home"] + METRICS
    ].copy()

    result.insert(0, "Season", season)
    result = result.rename(columns={
        "fixture": "SourceFixtureId",
        "GW": "SourceGameweek",
        "element": "SourcePlayerId",
        "name": "PlayerName",
        "team": "Team",
        "position": "Position",
        "was_home": "WasHome",
    })

    outputs.append(result)
    print(f"  PASS: {len(result):,} player records, 380 fixtures")

combined = pd.concat(outputs, ignore_index=True)
combined = combined.sort_values(
    ["Season", "Date", "SourceFixtureId", "SourcePlayerId"]
)

ROOT.mkdir(parents=True, exist_ok=True)
destination = ROOT / "player_performance_2021_2025.csv"
combined.to_csv(destination, index=False)

print("\nH07 INGESTION COMPLETE")
print("Total player records:", len(combined))
print("Output:", destination)
print("Missing starts:", combined["starts"].isna().sum())
print("Missing xG:", combined["expected_goals"].isna().sum())
