from pathlib import Path
import sys


# Allow Python to import our analytics folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT)
)


from analytics.football_analytics import (
    get_league_table,
    get_team_record,
    get_team_form,
    get_form_summary,
    get_home_away_record,
    get_head_to_head,
    compare_teams
)


print("\nFOOTBALL COPILOT ANALYTICS TEST")
print("===============================")


# --------------------------------------------------
# Test 1: League table
# --------------------------------------------------

print("\n1. LEAGUE TABLE")
print("================")

league_table = get_league_table(
    "2025/26"
)

print(league_table)


# --------------------------------------------------
# Test 2: Liverpool season record
# --------------------------------------------------

print("\n2. LIVERPOOL RECORD")
print("===================")

record = get_team_record(
    "Liverpool",
    "2025/26"
)

print(record)


# --------------------------------------------------
# Test 3: Liverpool last 10 matches
# --------------------------------------------------

print("\n3. LIVERPOOL LAST 10")
print("====================")

form = get_team_form(
    "Liverpool",
    10
)

print(form)


# --------------------------------------------------
# Test 4: Liverpool home vs away
# --------------------------------------------------

print("\n4. LIVERPOOL HOME VS AWAY")
print("=========================")

home_away = get_home_away_record(
    "Liverpool",
    "2025/26"
)

print(home_away)


# --------------------------------------------------
# Test 5: Liverpool vs Arsenal
# --------------------------------------------------

print("\n5. LIVERPOOL VS ARSENAL")
print("=======================")

head_to_head = get_head_to_head(
    "Liverpool",
    "Arsenal"
)

print(head_to_head)


# --------------------------------------------------
# Test 6: Liverpool vs Arsenal season comparison
# --------------------------------------------------

print("\n6. TEAM COMPARISON")
print("==================")

comparison = compare_teams(
    "Liverpool",
    "Arsenal",
    "2025/26"
)

print(comparison)

print("\n7. FORM SUMMARY")
print("===============")

form_summary = get_form_summary(
    "Liverpool",
    10
)

print(form_summary)


print("\nAll analytics tests complete.")