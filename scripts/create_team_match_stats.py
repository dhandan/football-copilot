import pandas as pd
from pathlib import Path


INPUT_FILE = "data/processed/matches_clean.csv"
OUTPUT_FILE = "data/processed/team_match_stats.csv"


print("\nCreating team-match analytical dataset...")
print("========================================")


# --------------------------------------------------
# 1. Load clean match data
# --------------------------------------------------

matches = pd.read_csv(INPUT_FILE)

matches["Date"] = pd.to_datetime(
    matches["Date"],
    errors="coerce"
)

print(f"\nMatches loaded: {len(matches)}")


# --------------------------------------------------
# 2. Create home-team records
# --------------------------------------------------

home = pd.DataFrame()

home["Season"] = matches["Season"]
home["Date"] = matches["Date"]

home["Team"] = matches["HomeTeam"]
home["Opponent"] = matches["AwayTeam"]

home["Venue"] = "Home"

home["GoalsFor"] = matches["FTHG"]
home["GoalsAgainst"] = matches["FTAG"]

home["Result"] = matches["FTR"].map({
    "H": "W",
    "D": "D",
    "A": "L"
})


# --------------------------------------------------
# 3. Create away-team records
# --------------------------------------------------

away = pd.DataFrame()

away["Season"] = matches["Season"]
away["Date"] = matches["Date"]

away["Team"] = matches["AwayTeam"]
away["Opponent"] = matches["HomeTeam"]

away["Venue"] = "Away"

away["GoalsFor"] = matches["FTAG"]
away["GoalsAgainst"] = matches["FTHG"]

away["Result"] = matches["FTR"].map({
    "H": "L",
    "D": "D",
    "A": "W"
})


# --------------------------------------------------
# 4. Combine home and away records
# --------------------------------------------------

team_matches = pd.concat(
    [home, away],
    ignore_index=True
)


# --------------------------------------------------
# 5. Add analytical fields
# --------------------------------------------------

team_matches["Points"] = team_matches["Result"].map({
    "W": 3,
    "D": 1,
    "L": 0
})

team_matches["GoalDifference"] = (
    team_matches["GoalsFor"]
    - team_matches["GoalsAgainst"]
)

team_matches["Win"] = (
    team_matches["Result"] == "W"
).astype(int)

team_matches["Draw"] = (
    team_matches["Result"] == "D"
).astype(int)

team_matches["Loss"] = (
    team_matches["Result"] == "L"
).astype(int)

team_matches["CleanSheet"] = (
    team_matches["GoalsAgainst"] == 0
).astype(int)

team_matches["FailedToScore"] = (
    team_matches["GoalsFor"] == 0
).astype(int)


# --------------------------------------------------
# 6. Sort the dataset
# --------------------------------------------------

team_matches = team_matches.sort_values(
    ["Date", "Team"]
).reset_index(drop=True)


# --------------------------------------------------
# 7. Quality checks
# --------------------------------------------------

print("\nQUALITY CHECKS")
print("==============")

expected_rows = len(matches) * 2
actual_rows = len(team_matches)

print(f"\nExpected rows: {expected_rows}")
print(f"Actual rows:   {actual_rows}")

print(
    "Row count correct:",
    expected_rows == actual_rows
)

print(
    "Missing teams:",
    team_matches["Team"].isna().sum()
)

print(
    "Missing opponents:",
    team_matches["Opponent"].isna().sum()
)

print(
    "Goals balance:",
    team_matches["GoalsFor"].sum()
    ==
    team_matches["GoalsAgainst"].sum()
)

print("\nVenue counts:")
print(
    team_matches["Venue"].value_counts()
)

print("\nResult counts:")
print(
    team_matches["Result"].value_counts()
)

print("\nPoints values:")
print(
    sorted(
        team_matches["Points"]
        .dropna()
        .unique()
    )
)


# --------------------------------------------------
# 8. Save the final dataset
# --------------------------------------------------

Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)

team_matches.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDataset saved.")
print(f"Saved to: {OUTPUT_FILE}")


# --------------------------------------------------
# 9. Show first 10 records
# --------------------------------------------------

print("\nFIRST 10 TEAM-MATCH RECORDS")
print("===========================")

print(
    team_matches[
        [
            "Season",
            "Date",
            "Team",
            "Opponent",
            "Venue",
            "GoalsFor",
            "GoalsAgainst",
            "Result",
            "Points"
        ]
    ].head(10)
)