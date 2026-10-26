from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import prediction.fixture_predictor as fp

parser = argparse.ArgumentParser()
parser.add_argument("--gameweek", type=int, default=7)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

fixture_dir = ROOT / "data/live/fixtures"
history_path = ROOT / "data/live/model2/matches_asof_gw07.csv"
output_dir = ROOT / "data/live/model2/fixture_freezes"

snapshot_root = ROOT / "data/live/players/h07_snapshots"

snapshots = sorted(
    p for p in snapshot_root.iterdir()
    if p.is_dir()
    and (p / "fixtures.csv").exists()
    and (p / "teams.csv").exists()
)

if not snapshots:
    raise FileNotFoundError("No H07 FPL fixture snapshot found")

snapshot = snapshots[-1]
fpl_fixtures = pd.read_csv(snapshot / "fixtures.csv")
fpl_teams = pd.read_csv(snapshot / "teams.csv")

team_names = fpl_teams.set_index("id")["name"].to_dict()

fixtures = fpl_fixtures[
    fpl_fixtures["event"] == args.gameweek
].copy()

if len(fixtures) != 10:
    raise ValueError(
        f"Expected 10 GW{args.gameweek} fixtures, "
        f"found {len(fixtures)}"
    )

fixtures["FixtureId"] = fixtures["id"]
fixtures["HomeTeam"] = fixtures["team_h"].map(team_names)
fixtures["AwayTeam"] = fixtures["team_a"].map(team_names)

kickoff = pd.to_datetime(
    fixtures["kickoff_time"], utc=True
).dt.tz_convert("Europe/London")

fixtures["FixtureDate"] = kickoff.dt.strftime("%Y-%m-%d")
fixtures["FixtureTime"] = kickoff.dt.strftime("%H:%M")

# Translate official FPL club names to the names used
# in Model 2's historical dataset.
aliases = {
    "Manchester United": "Man United",
    "Man Utd": "Man United",
    "Manchester City": "Man City",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham",
    "Spurs": "Tottenham",
    "Newcastle United": "Newcastle",
    "Brighton and Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Leeds United": "Leeds",
    "Ipswich Town": "Ipswich",
    "Wolverhampton Wanderers": "Wolves",
}

for col in ["HomeTeam", "AwayTeam"]:
    fixtures[col] = fixtures[col].replace(aliases)

fixtures = fixtures.sort_values(
    ["FixtureDate", "FixtureTime"]
)

fixture_file = snapshot / "fixtures.csv"

if fixtures["FixtureId"].duplicated().any():
    raise ValueError("Duplicate fixture IDs")

history = pd.read_csv(history_path)
history["Date"] = pd.to_datetime(history["Date"], errors="raise")
history["Season"] = history["Season"].astype(str)

if history.duplicated(
    ["Season", "Date", "HomeTeam", "AwayTeam"]
).any():
    raise ValueError("Duplicate historical matches")

now = pd.Timestamp.now(tz="UTC")
model_path = ROOT / "models/production_model_v2.pkl"
model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

print("GW7 FIXTURE-LEVEL FREEZE RUNNER")
print("Mode:", "DRY RUN" if args.dry_run else "OFFICIAL")
print("Fixture source:", fixture_file.name)
print("Current UTC:", now.isoformat())
print("Frozen model SHA256:", model_hash[:16])

created = 0
pending = 0
missed = 0

for f in fixtures.itertuples(index=False):
    kickoff_text = f"{f.FixtureDate} {f.FixtureTime}"
    kickoff = pd.Timestamp(kickoff_text)

    if kickoff.tzinfo is None:
        kickoff = kickoff.tz_localize(
            "Europe/London", ambiguous="raise",
            nonexistent="raise"
        )

    kickoff = kickoff.tz_convert("UTC")
    cutoff = kickoff - pd.Timedelta(hours=24)

    fixture_id = int(f.FixtureId)
    path = output_dir / (
        f"gw{args.gameweek:02d}_fixture_{fixture_id}.json"
    )

    if path.exists():
        print(f"EXISTS  {f.HomeTeam} vs {f.AwayTeam}")
        continue

    if not args.dry_run and now < cutoff:
        pending += 1
        print(f"PENDING {f.HomeTeam} vs {f.AwayTeam}")
        continue

    if not args.dry_run and now > cutoff + pd.Timedelta(minutes=10):
        missed += 1
        print(f"MISSED  {f.HomeTeam} vs {f.AwayTeam}")
        continue

    # Date-only historical source: conservatively exclude all
    # matches on the UTC calendar date of the freeze.
    eligible = history[
        history["Date"].dt.date < cutoff.date()
    ].copy()

    current_count = int(
        (eligible["Season"] == "2026/27").sum()
    )

    original = fp.matches

    try:
        fp.matches = eligible.sort_values(
            "Date"
        ).reset_index(drop=True)

        prediction = fp.predict_fixture(
            f.HomeTeam, f.AwayTeam
        )
    finally:
        fp.matches = original

    probabilities = [
        float(prediction["home_win_probability"]),
        float(prediction["draw_probability"]),
        float(prediction["away_win_probability"])
    ]

    if abs(sum(probabilities) - 100) > 0.2:
        raise ValueError(
            f"Invalid probabilities for fixture {fixture_id}"
        )

    record = {
        "Season": "2026/27",
        "Gameweek": args.gameweek,
        "FixtureId": fixture_id,
        "HomeTeam": f.HomeTeam,
        "AwayTeam": f.AwayTeam,
        "KickoffUTC": kickoff.isoformat(),
        "FreezeUTC": cutoff.isoformat(),
        "PredictionUTC": now.isoformat(),
        "ModelVersion": "Model2_v1.0",
        "ModelSHA256": model_hash,
        "HistorySource": str(history_path),
        "HistoryMatchesUsed": len(eligible),
        "CurrentSeasonMatchesUsed": current_count,
        "FeatureSeason": prediction.get("feature_season"),
        "HomeWinProbability": probabilities[0],
        "DrawProbability": probabilities[1],
        "AwayWinProbability": probabilities[2],
        "ExpectedHomeGoals": float(
            prediction["expected_home_goals"]
        ),
        "ExpectedAwayGoals": float(
            prediction["expected_away_goals"]
        ),
        "Official": not args.dry_run
    }

    if args.dry_run:
        print(
            f"READY   {f.HomeTeam} vs {f.AwayTeam} | "
            f"Freeze {cutoff.isoformat()} | "
            f"2026/27 matches {current_count} | "
            f"H/D/A {probabilities}"
        )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Exclusive creation prevents accidental overwriting.
        with path.open("x", encoding="utf-8") as file:
            json.dump(record, file, indent=2)

        created += 1
        print("FROZEN ", f.HomeTeam, "vs", f.AwayTeam)

print()
print("Created:", created)
print("Pending:", pending)
print("Missed:", missed)
print("PASS: Freeze runner completed.")
