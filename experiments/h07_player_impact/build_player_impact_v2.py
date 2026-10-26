from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("experiments/h07_player_impact")
SOURCE = ROOT / "h07_player_importance_v1.csv"
OUTPUT = ROOT / "h07_player_impact_v2.csv"

d = pd.read_csv(SOURCE, low_memory=False)

KEYS = ["Season", "Team", "SourceFixtureId", "KickoffUTC"]
POSITIONS = ["GK", "DEF", "MID", "FWD"]

if d.duplicated(KEYS + ["SourcePlayerId"]).any():
    raise ValueError("Duplicate player observations")

if not set(d["Position"].dropna().unique()).issubset(POSITIONS):
    raise ValueError("Unexpected player position")

results = []

for key, group in d.groupby(KEYS, sort=False):
    active = group[group["Minutes"] > 0].copy()

    if active.empty:
        raise ValueError(f"No historical playing time for {key}")

    ranked = active.sort_values(
        ["Minutes", "SourcePlayerId"],
        ascending=[False, True]
    )

    total_minutes = ranked["Minutes"].sum()

    row = dict(zip(KEYS, key))
    row["PriorMatches"] = int(group["PriorMatches"].iloc[0])
    row["ActivePlayers"] = len(active)

    row["Top3MinutesShare"] = (
        ranked.head(3)["Minutes"].sum() / total_minutes
    )

    row["Top5MinutesShare"] = (
        ranked.head(5)["Minutes"].sum() / total_minutes
    )

    # Herfindahl concentration of historical playing time.
    shares = ranked["Minutes"] / total_minutes
    row["MinutesConcentration"] = float((shares ** 2).sum())

    # Attacking output is kept separate from playing-time dependence.
    for metric, output in [
        ("XG", "Top3XGShare"),
        ("XA", "Top3XAShare"),
        ("Goals", "Top3GoalsShare"),
        ("Assists", "Top3AssistsShare"),
    ]:
        valid = ranked[ranked[metric].notna()]
        total = valid[metric].sum()

        if total > 0:
            leaders = valid.nlargest(3, metric)
            row[output] = leaders[metric].sum() / total
        else:
            row[output] = np.nan

    # Position-specific historical involvement and depth.
    for position in POSITIONS:
        players = ranked[
            ranked["Position"] == position
        ].sort_values(
            ["Minutes", "SourcePlayerId"],
            ascending=[False, True]
        )

        minutes = players["Minutes"].to_numpy()

        row[f"{position}_ActivePlayers"] = len(players)

        row[f"{position}_TopMinutes"] = (
            float(minutes[0]) if len(minutes) else np.nan
        )

        row[f"{position}_SecondMinutes"] = (
            float(minutes[1]) if len(minutes) > 1 else np.nan
        )

        row[f"{position}_MinutesGap"] = (
            float(minutes[0] - minutes[1])
            if len(minutes) > 1 else np.nan
        )

        # Per-player xG+xA output is not comparable across positions
        # without modelling, so retain position-specific aggregates.
        known = players[
            players["XG"].notna() & players["XA"].notna()
        ]

        row[f"{position}_XGXA"] = (
            float((known["XG"] + known["XA"]).sum())
            if len(known) else np.nan
        )

    results.append(row)

out = pd.DataFrame(results)

if len(out) != 3700:
    raise ValueError(f"Expected 3700 rows, found {len(out)}")

if out.duplicated(KEYS).any():
    raise ValueError("Duplicate team-fixture output")

out.to_csv(OUTPUT, index=False)

print("H07 PLAYER IMPACT V2")
print("Team-fixtures:", len(out))
print("Feature columns:", len(out.columns))
print("Seasons:", out["Season"].nunique())
print("Missing Top3XGShare:",
      int(out["Top3XGShare"].isna().sum()))
print("Output:", OUTPUT)

print("\nExample:")
print(
    out[[
        "Season", "Team", "Top3MinutesShare",
        "Top3XGShare", "MinutesConcentration",
        "FWD_ActivePlayers", "FWD_MinutesGap"
    ]].tail(5).to_string(index=False)
)
