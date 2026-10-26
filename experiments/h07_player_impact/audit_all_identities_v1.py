
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

ROOT = Path("experiments/h07_player_impact")
BASE = "https://raw.githubusercontent.com/withqwerty/availability-data/main/"

def normalise(value):
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(
        c for c in value if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z0-9]", "", value.lower())

response = requests.get(
    "https://api.github.com/repos/withqwerty/availability-data/git/trees/main?recursive=1",
    timeout=60
)
response.raise_for_status()

tree = response.json()
if tree.get("truncated"):
    raise RuntimeError("Incomplete repository inventory")

paths = sorted(
    item["path"] for item in tree["tree"]
    if re.fullmatch(r"raw/GB1/202[1-5]/[^/]+\.json", item["path"])
)

assert len(paths) == 100, f"Expected 100 files, found {len(paths)}"

fpl = pd.read_csv(
    ROOT / "player_performance_2021_2025.csv",
    low_memory=False
)

fpl = fpl[
    ["Season", "Team", "SourcePlayerId", "PlayerName"]
].drop_duplicates()

fpl["Key"] = fpl["PlayerName"].map(normalise)

reviewed = pd.read_csv(
    ROOT / "verified_identity_crosswalk_v1.csv",
    encoding="utf-8-sig"
)

additional = pd.read_csv(
    ROOT / "additional_identity_crosswalk_v1.csv",
    encoding="utf-8-sig"
)

reviewed = pd.concat([reviewed, additional], ignore_index=True)

if reviewed.duplicated(
    ["Season", "Team", "TransfermarktID"]
).any():
    raise ValueError("Conflicting reviewed Transfermarkt identities")

if reviewed.duplicated(
    ["Season", "Team", "FPLPlayerID"]
).any():
    raise ValueError("Conflicting reviewed FPL identities")

manual = {
    (str(r.Season), str(r.Team), int(r.TransfermarktID)):
    int(r.FPLPlayerID)
    for r in reviewed.itertuples()
}

records = []
session = requests.Session()

for index, path in enumerate(paths, 1):
    r = session.get(BASE + path, timeout=60)
    r.raise_for_status()
    data = r.json()

    year = int(path.split("/")[2])
    season = f"{year}/{str(year + 1)[-2:]}"

    for competition in data.get("competitions", []):
        if competition.get("code") != "GB1":
            continue

        for player in competition.get("players", []):
            matches = player.get("matches", [])
            detail = lambda m: str(m.get("detail") or "").lower()

            records.append({
                "Season": season,
                "ClubName": data.get("club"),
                "AvailabilityName": player.get("name"),
                "TransfermarktID": player.get("tmId"),
                "Key": normalise(player.get("name")),
                "InjuredRounds": sum(
                    m.get("status") == "injured" for m in matches
                ),
                "SuspendedRounds": sum(
                    m.get("status") == "absent"
                    and "suspension" in detail(m)
                    for m in matches
                ),
            })

    if index % 20 == 0:
        print(f"Downloaded {index}/100 club files", flush=True)

availability = pd.DataFrame(records)

# Match clubs by their player-name overlap, without guessing aliases.
club_links = []
for (season, club), group in availability.groupby(
    ["Season", "ClubName"]
):
    candidate = fpl[fpl["Season"] == season]

    overlaps = (
        candidate[candidate["Key"].isin(group["Key"])]
        .groupby("Team")["Key"].nunique()
        .sort_values(ascending=False)
    )

    if len(overlaps) == 0 or (
        len(overlaps) > 1 and overlaps.iloc[0] == overlaps.iloc[1]
    ):
        raise ValueError(f"Ambiguous club mapping: {season} {club}")

    club_links.append({
        "Season": season,
        "ClubName": club,
        "Team": overlaps.index[0],
        "ClubMatchCount": int(overlaps.iloc[0])
    })

clubs = pd.DataFrame(club_links)

if clubs.duplicated(["Season", "Team"]).any():
    raise ValueError("Multiple availability clubs assigned to same FPL club")

availability = availability.merge(
    clubs, on=["Season", "ClubName"],
    validate="many_to_one"
)

fpl_groups = defaultdict(list)
for row in fpl.itertuples():
    fpl_groups[(row.Season, row.Team, row.Key)].append(
        (int(row.SourcePlayerId), row.PlayerName)
    )

rows = []

for row in availability.itertuples():
    tm_id = int(row.TransfermarktID)
    manual_key = (row.Season, row.Team, tm_id)
    candidates = fpl_groups[(row.Season, row.Team, row.Key)]

    if manual_key in manual:
        player_id = manual[manual_key]
        method = "reviewed"
    elif len(candidates) == 1:
        player_id = candidates[0][0]
        method = "exact_unique"
    elif len(candidates) > 1:
        player_id = None
        method = "ambiguous"
    else:
        player_id = None
        method = "unmatched"

    rows.append({
        "Season": row.Season,
        "Team": row.Team,
        "AvailabilityName": row.AvailabilityName,
        "TransfermarktID": tm_id,
        "SourcePlayerId": player_id,
        "MatchMethod": method,
        "InjuredRounds": row.InjuredRounds,
        "SuspendedRounds": row.SuspendedRounds
    })

out = pd.DataFrame(rows)

matched = out["SourcePlayerId"].notna()

print("\nH07 FIVE-SEASON IDENTITY AUDIT")
print("Availability identities:", len(out))
print("Matched identities:", int(matched.sum()))
print("Unmatched identities:", int((~matched).sum()))
print("Ambiguous identities:", int((out["MatchMethod"] == "ambiguous").sum()))

for metric in ["InjuredRounds", "SuspendedRounds"]:
    print(
        f"{metric}:",
        int(out.loc[matched, metric].sum()),
        "/",
        int(out[metric].sum()),
        "covered"
    )

print("\nCoverage by season:")
for season, group in out.groupby("Season"):
    found = group["SourcePlayerId"].notna()
    print(
        season,
        f"identities {found.sum()}/{len(group)}",
        f"injuries {group.loc[found, 'InjuredRounds'].sum()}"
        f"/{group['InjuredRounds'].sum()}"
    )

output = ROOT / "h07_identity_audit_all_seasons_v1.csv"
out.to_csv(output, index=False)
print("\nOutput:", output)
