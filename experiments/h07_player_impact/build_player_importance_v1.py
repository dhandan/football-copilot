from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("experiments/h07_player_impact")
SOURCE = ROOT / "player_performance_2021_2025.csv"
OUTPUT = ROOT / "h07_player_importance_v1.csv"

d = pd.read_csv(SOURCE, low_memory=False)
d["KickoffUTC"] = pd.to_datetime(d["KickoffUTC"], utc=True)

metrics = [
    "minutes", "starts", "goals_scored", "assists",
    "expected_goals", "expected_assists"
]
for metric in metrics:
    d[metric] = pd.to_numeric(d[metric], errors="coerce")

d["Position"] = d["Position"].astype(str).str.upper()

results = []

for (season, team), club in d.groupby(["Season", "Team"], sort=False):
    club = club.sort_values(["KickoffUTC", "SourceFixtureId"])
    fixtures = list(club.groupby("SourceFixtureId", sort=False))

    for fixture_id, current in fixtures:
        kickoff = current["KickoffUTC"].iloc[0]
        cutoff = kickoff - pd.Timedelta(hours=24)

        history = club[
            (club["KickoffUTC"] + pd.Timedelta(hours=3) <= cutoff)
        ].copy()

        prior_fixtures = (
            history[["SourceFixtureId", "KickoffUTC"]]
            .drop_duplicates()
            .sort_values("KickoffUTC")
            .tail(5)["SourceFixtureId"]
        )

        history = history[
            history["SourceFixtureId"].isin(prior_fixtures)
        ]

        if history.empty:
            continue

        players = history.groupby("SourcePlayerId").agg(
            PlayerName=("PlayerName", "last"),
            Position=("Position", "last"),
            Minutes=("minutes", "sum"),
            Starts=("starts", lambda x: x.sum(min_count=1)),
            Goals=("goals_scored", "sum"),
            Assists=("assists", "sum"),
            XG=("expected_goals", lambda x: x.sum(min_count=1)),
            XA=("expected_assists", lambda x: x.sum(min_count=1)),
        ).reset_index()

        team_minutes = players["Minutes"].sum()

        if team_minutes <= 0:
            continue

        players["ImportanceMinutesShare"] = (
            players["Minutes"] / team_minutes
        )

        for numerator, feature in [
            ("Goals", "GoalsPer90"),
            ("Assists", "AssistsPer90"),
            ("XG", "XGPer90"),
            ("XA", "XAPer90")
        ]:
            players[feature] = np.where(
                players["Minutes"] > 0,
                90 * players[numerator] / players["Minutes"],
                np.nan
            )

        players["PositionDepthRank"] = (
            players.groupby("Position")["Minutes"]
            .rank(method="first", ascending=False)
        )

        players["Season"] = season
        players["Team"] = team
        players["SourceFixtureId"] = fixture_id
        players["KickoffUTC"] = kickoff
        players["PriorMatches"] = len(prior_fixtures)

        results.append(players)

out = pd.concat(results, ignore_index=True)

if out.duplicated([
    "Season", "Team", "SourceFixtureId", "SourcePlayerId"
]).any():
    raise ValueError("Duplicate player-feature records")

if (out["Minutes"] < 0).any():
    raise ValueError("Invalid negative playing minutes")

out.to_csv(OUTPUT, index=False)

print("H07 PLAYER IMPORTANCE V1")
print("Player-fixture feature records:", len(out))
print("Fixtures with player history:",
      out[["Season", "SourceFixtureId"]].drop_duplicates().shape[0])
print("Players represented:", out[
    ["Season", "SourcePlayerId"]
].drop_duplicates().shape[0])
print("Output:", OUTPUT)

print("\nSample top-importance players:")
print(
    out.sort_values("ImportanceMinutesShare", ascending=False)
    [["Season", "Team", "PlayerName", "Position",
      "Minutes", "ImportanceMinutesShare", "PositionDepthRank"]]
    .head(10).to_string(index=False)
)
