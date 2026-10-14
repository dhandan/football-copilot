from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("experiments/h07_player_impact")
source = ROOT / "player_performance_2021_2025.csv"
output = ROOT / "h07_player_features_v1.csv"

d = pd.read_csv(source, low_memory=False)
d["Date"] = pd.to_datetime(d["Date"])
d["KickoffUTC"] = pd.to_datetime(d["KickoffUTC"], utc=True)

metrics = [
    "minutes", "starts", "goals_scored", "assists",
    "expected_goals", "expected_assists"
]

for col in metrics:
    d[col] = pd.to_numeric(d[col], errors="coerce")

results = []

# Separate histories for each club and season.
for (season, team), club in d.groupby(["Season", "Team"], sort=False):
    history = deque(maxlen=5)
    pending = deque()

    # Process actual match dates, not gameweek numbers.
    for kickoff, day in club.groupby("KickoffUTC", sort=True):
        date = kickoff.date()
        fixtures = list(day.groupby("SourceFixtureId"))

        # Only release completed performances before the T-24h cutoff.
        cutoff = kickoff - pd.Timedelta(hours=24)
        while pending and pending[0][0] <= cutoff:
            _, completed_fixture = pending.popleft()
            history.append(completed_fixture)

        # Calculate features before incorporating today's performances.
        previous = pd.concat(list(history), ignore_index=True) if history else pd.DataFrame()

        features = {
            "Season": season,
            "Team": team,
            "Date": date,
            "KickoffUTC": kickoff,
            "PriorMatches": len(history),
            "MinutesTop3": np.nan,
            "MinutesNext3": np.nan,
            "SquadDepth90": np.nan,
            "RecentGoalsPer90": np.nan,
            "RecentAssistsPer90": np.nan,
            "RecentXGPer90": np.nan,
            "RecentXAPer90": np.nan,
            "RecentStarts": np.nan,
        }

        if not previous.empty:
            active = previous[previous["minutes"] > 0]

            players = active.groupby("SourcePlayerId").agg(
                Minutes=("minutes", "sum"),
                Starts=("starts", lambda x: x.sum(min_count=1))
            ).sort_values("Minutes", ascending=False)

            ranking = players["Minutes"].to_numpy()

            features["MinutesTop3"] = ranking[:3].sum()
            features["MinutesNext3"] = ranking[3:6].sum()
            features["SquadDepth90"] = int((players["Minutes"] >= 90).sum())

            total_minutes = active["minutes"].sum()

            if total_minutes > 0:
                for metric, feature in [
                    ("goals_scored", "RecentGoalsPer90"),
                    ("assists", "RecentAssistsPer90"),
                    ("expected_goals", "RecentXGPer90"),
                    ("expected_assists", "RecentXAPer90")
                ]:
                    if active[metric].notna().any():
                        features[feature] = (
                            active[metric].sum(min_count=1)
                            * 90 / total_minutes
                        )

            if players["Starts"].notna().any():
                features["RecentStarts"] = players["Starts"].sum()

        for fixture_id, fixture in fixtures:
            row = features.copy()
            row["SourceFixtureId"] = fixture_id
            row["WasHome"] = bool(fixture["WasHome"].iloc[0])
            results.append(row)

        # Only completed prior-date matches enter future histories.
        # Same-day matches are added after all features for the date.
        for _, fixture in fixtures:
            pending.append((
                kickoff + pd.Timedelta(hours=3),
                fixture.copy()
            ))

features = pd.DataFrame(results).sort_values(
    ["Season", "Date", "Team"]
)

if len(features) != 3800:
    raise ValueError(f"Expected 3800 team-fixture rows, got {len(features)}")

if features.duplicated(["Season", "Team", "SourceFixtureId"]).any():
    raise ValueError("Duplicate team-fixture feature rows")

features.to_csv(output, index=False)

print("H07 ROLLING FEATURES V1")
print("Team-fixture records:", len(features))
print("Fixtures represented:", len(features) // 2)
print("Feature columns:", len(features.columns))
print("Rows with prior history:", int((features["PriorMatches"] > 0).sum()))
print("Output:", output)
print("\nSample:")
print(features.tail(5).to_string(index=False))
