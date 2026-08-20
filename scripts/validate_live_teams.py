from pathlib import Path

import pandas as pd


# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


MATCH_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "matches_clean.csv"
)


FIXTURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "fixtures"
)


# ==================================================
# FIND LATEST FIXTURE SNAPSHOT
# ==================================================

fixture_files = sorted(
    FIXTURE_DIR.glob(
        "fixtures_*.csv"
    ),
    key=lambda path:
        path.stat().st_mtime,
    reverse=True,
)


if not fixture_files:

    raise FileNotFoundError(
        "No live fixture files found."
    )


fixture_file = fixture_files[0]


print()
print("FOOTBALL COPILOT")
print("LIVE TEAM VALIDATION")
print("====================")

print()
print(
    f"Fixture snapshot: "
    f"{fixture_file.name}"
)


# ==================================================
# LOAD DATA
# ==================================================

fixtures = pd.read_csv(
    fixture_file
)


matches = pd.read_csv(
    MATCH_FILE
)


# ==================================================
# LOCAL HISTORICAL TEAMS
# ==================================================

historical_teams = sorted(
    set(
        matches[
            "HomeTeam"
        ].dropna()
    )
    |
    set(
        matches[
            "AwayTeam"
        ].dropna()
    )
)


# ==================================================
# LIVE TEAMS
# ==================================================

live_teams = sorted(
    set(
        fixtures[
            "HomeTeam"
        ].dropna()
    )
    |
    set(
        fixtures[
            "AwayTeam"
        ].dropna()
    )
)


# ==================================================
# COMPARE
# ==================================================

matched = []

unmatched = []


for team in live_teams:

    if team in historical_teams:

        matched.append(
            team
        )

    else:

        unmatched.append(
            team
        )


# ==================================================
# OUTPUT
# ==================================================

print()
print("MATCHED TEAMS")
print("=============")


for team in matched:

    print(
        f"✓ {team}"
    )


print()
print("UNMATCHED TEAMS")
print("===============")


if unmatched:

    for team in unmatched:

        print(
            f"✗ {team}"
        )

else:

    print(
        "None"
    )


print()
print("SUMMARY")
print("=======")

print(
    f"Live teams: "
    f"{len(live_teams)}"
)

print(
    f"Matched: "
    f"{len(matched)}"
)

print(
    f"Unmatched: "
    f"{len(unmatched)}"
)


# ==================================================
# POSSIBLE NAME MATCHES
# ==================================================

if unmatched:

    print()
    print("POSSIBLE HISTORICAL NAME MATCHES")
    print("================================")


    for live_team in unmatched:

        print()
        print(
            f"{live_team}:"
        )


        live_words = set(
            live_team
            .lower()
            .split()
        )


        suggestions = []


        for historical_team in historical_teams:

            historical_words = set(
                historical_team
                .lower()
                .split()
            )


            if (
                live_words
                &
                historical_words
            ):

                suggestions.append(
                    historical_team
                )


        if suggestions:

            for suggestion in suggestions:

                print(
                    f"  -> {suggestion}"
                )

        else:

            print(
                "  -> No obvious historical match"
            )