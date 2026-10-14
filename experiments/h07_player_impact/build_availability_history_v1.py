
import re
from pathlib import Path

import pandas as pd
import requests

ROOT = Path("experiments/h07_player_impact")
BASE = "https://raw.githubusercontent.com/withqwerty/availability-data/main/"
OUTPUT = ROOT / "h07_availability_history_v1.csv"

crosswalk = pd.read_csv(
    ROOT / "h07_identity_audit_all_seasons_v1.csv"
)

crosswalk = crosswalk[
    crosswalk["SourcePlayerId"].notna()
].copy()

crosswalk["SourcePlayerId"] = (
    crosswalk["SourcePlayerId"].astype(int)
)

keys = ["Season", "Team", "TransfermarktID"]

if crosswalk.duplicated(keys).any():
    raise ValueError("Duplicate availability identity mapping")

# Resolve club names from previously validated identity audit.
clubs = crosswalk[
    ["Season", "Team"]
].drop_duplicates()

session = requests.Session()

response = session.get(
    "https://api.github.com/repos/withqwerty/availability-data/git/trees/main?recursive=1",
    timeout=60
)
response.raise_for_status()

tree = response.json()
if tree.get("truncated"):
    raise RuntimeError("Incomplete repository inventory")

paths = sorted(
    item["path"] for item in tree["tree"]
    if re.fullmatch(
        r"raw/GB1/202[1-5]/[^/]+\.json",
        item["path"]
    )
)

assert len(paths) == 100

records = []

for index, path in enumerate(paths, 1):
    r = session.get(BASE + path, timeout=60)
    r.raise_for_status()
    data = r.json()

    year = int(path.split("/")[2])
    season = f"{year}/{str(year+1)[-2:]}"

    club_name = data.get("club")

    # Reuse only clubs represented in our validated identity audit.
    candidates = crosswalk[
        (crosswalk["Season"] == season)
    ]

    # Identify the club through known Transfermarkt player identities.
    tm_ids = {
        int(p["tmId"])
        for c in data.get("competitions", [])
        if c.get("code") == "GB1"
        for p in c.get("players", [])
        if p.get("tmId") is not None
    }

    scores = (
        candidates[candidates["TransfermarktID"].isin(tm_ids)]
        .groupby("Team")["TransfermarktID"]
        .nunique()
        .sort_values(ascending=False)
    )

    if len(scores) == 0 or (
        len(scores) > 1 and scores.iloc[0] == scores.iloc[1]
    ):
        raise ValueError(f"Ambiguous club identity: {season} {club_name}")

    team = scores.index[0]

    for competition in data.get("competitions", []):
        if competition.get("code") != "GB1":
            continue

        for player in competition.get("players", []):
            tm_id = player.get("tmId")
            if tm_id is None:
                continue

            for match in player.get("matches", []):
                status = str(match.get("status") or "").lower()
                detail = str(match.get("detail") or "")
                detail_lower = detail.lower()

                if status == "injured":
                    reason = "injury"
                elif status == "absent" and "suspension" in detail_lower:
                    reason = "suspension"
                elif status == "absent" and "national team" in detail_lower:
                    reason = "international_duty"
                elif status == "absent":
                    reason = "other_absence"
                else:
                    reason = status

                records.append({
                    "Season": season,
                    "Team": team,
                    "TransfermarktID": int(tm_id),
                    "AvailabilityName": player.get("name"),
                    "Round": str(match.get("round")),
                    "Status": status,
                    "Reason": reason,
                    "Detail": detail,
                    "ScrapedAt": data.get("scrapedAt"),
                })

    if index % 20 == 0:
        print(f"Processed {index}/100 club files", flush=True)

raw = pd.DataFrame(records)

out = raw.merge(
    crosswalk[
        keys + ["SourcePlayerId", "MatchMethod"]
    ],
    on=keys,
    how="left",
    validate="many_to_one"
)

if out.duplicated(
    ["Season", "Team", "TransfermarktID", "Round"]
).any():
    raise ValueError("Duplicate player-round availability records")

out.to_csv(OUTPUT, index=False)

print("\nH07 HISTORICAL AVAILABILITY DATASET")
print("Player-round records:", len(out))
print("Matched player-round records:",
      int(out["SourcePlayerId"].notna().sum()))
print("Injury records:",
      int((out["Reason"] == "injury").sum()))
print("Suspension records:",
      int((out["Reason"] == "suspension").sum()))
print("Output:", OUTPUT)
