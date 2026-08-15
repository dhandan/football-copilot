from pathlib import Path
import sys
import duckdb


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT)
)


from analytics.football_analytics import (
    get_teams,
    get_seasons,
    get_league_table,
    get_team_record,
    get_team_form,
    get_form_summary,
    get_home_away_record,
    get_head_to_head,
    compare_teams
)


DATABASE_FILE = "database/football.duckdb"


print("\nFOOTBALL COPILOT VALIDATION")
print("===========================")


# --------------------------------------------------
# 1. Database exists
# --------------------------------------------------

print("\n1. DATABASE FILE")

database_exists = Path(
    DATABASE_FILE
).exists()

print(
    "Database exists:",
    database_exists
)


# --------------------------------------------------
# 2. Database tables
# --------------------------------------------------

print("\n2. DATABASE TABLES")

connection = duckdb.connect(
    DATABASE_FILE,
    read_only=True
)

tables = connection.execute("""
    SHOW TABLES
""").fetchdf()

print(tables)


# --------------------------------------------------
# 3. Row counts
# --------------------------------------------------

print("\n3. ROW COUNTS")

matches_count = connection.execute("""
    SELECT COUNT(*)
    FROM matches
""").fetchone()[0]

team_count = connection.execute("""
    SELECT COUNT(*)
    FROM team_match_stats
""").fetchone()[0]

print(
    "Matches:",
    matches_count
)

print(
    "Team-match rows:",
    team_count
)

print(
    "Correct 2-to-1 relationship:",
    team_count == matches_count * 2
)


# --------------------------------------------------
# 4. Missing critical data
# --------------------------------------------------

print("\n4. MISSING DATA")

missing = connection.execute("""
    SELECT
        SUM(CASE WHEN Team IS NULL THEN 1 ELSE 0 END)
            AS MissingTeams,

        SUM(CASE WHEN Opponent IS NULL THEN 1 ELSE 0 END)
            AS MissingOpponents,

        SUM(CASE WHEN Result IS NULL THEN 1 ELSE 0 END)
            AS MissingResults

    FROM team_match_stats
""").fetchdf()

print(missing)


# --------------------------------------------------
# 5. Goals reconcile
# --------------------------------------------------

print("\n5. GOALS RECONCILIATION")

goal_check = connection.execute("""
    SELECT
        SUM(GoalsFor) AS GoalsFor,
        SUM(GoalsAgainst) AS GoalsAgainst
    FROM team_match_stats
""").fetchdf()

print(goal_check)

goals_match = (
    goal_check.loc[0, "GoalsFor"]
    ==
    goal_check.loc[0, "GoalsAgainst"]
)

print(
    "Goals reconcile:",
    goals_match
)


# --------------------------------------------------
# 6. Valid results
# --------------------------------------------------

print("\n6. RESULT VALUES")

results = connection.execute("""
    SELECT DISTINCT Result
    FROM team_match_stats
    ORDER BY Result
""").fetchdf()

print(results)


# --------------------------------------------------
# 7. Valid points
# --------------------------------------------------

print("\n7. POINT VALUES")

points = connection.execute("""
    SELECT DISTINCT Points
    FROM team_match_stats
    ORDER BY Points
""").fetchdf()

print(points)

print("\nDUPLICATE CHECK")

duplicates = connection.execute("""
    SELECT
        Season,
        Date,
        Team,
        Opponent,
        COUNT(*) AS Records
    FROM team_match_stats
    GROUP BY
        Season,
        Date,
        Team,
        Opponent
    HAVING COUNT(*) > 1
""").fetchdf()

print(
    "Duplicate records:",
    len(duplicates)
)

print("\nMATCHES BY SEASON")

matches_by_season = connection.execute("""
    SELECT
        Season,
        COUNT(*) / 2 AS Matches
    FROM team_match_stats
    GROUP BY Season
    ORDER BY Season
""").fetchdf()

print(matches_by_season)

connection.close()


# --------------------------------------------------
# 8. Seasons
# --------------------------------------------------

print("\n8. AVAILABLE SEASONS")

seasons = get_seasons()

print(seasons)


# --------------------------------------------------
# 9. Teams
# --------------------------------------------------

print("\n9. AVAILABLE TEAMS")

teams = get_teams()

print(
    f"Unique teams: {len(teams)}"
)

print(teams)


# --------------------------------------------------
# 10. League table test
# --------------------------------------------------

print("\n10. LEAGUE TABLE TEST")

league_table = get_league_table(
    "2025/26"
)

print(
    league_table.head()
)


# --------------------------------------------------
# 11. Team record test
# --------------------------------------------------

print("\n11. TEAM RECORD TEST")

record = get_team_record(
    "Liverpool",
    "2025/26"
)

print(record)


# --------------------------------------------------
# 12. Form test
# --------------------------------------------------

print("\n12. FORM TEST")

form = get_team_form(
    "Liverpool",
    5
)

print(form)


# --------------------------------------------------
# 13. Form summary
# --------------------------------------------------

print("\n13. FORM SUMMARY TEST")

form_summary = get_form_summary(
    "Liverpool",
    5
)

print(form_summary)


# --------------------------------------------------
# 14. Home/away test
# --------------------------------------------------

print("\n14. HOME/AWAY TEST")

home_away = get_home_away_record(
    "Liverpool",
    "2025/26"
)

print(home_away)


# --------------------------------------------------
# 15. Head-to-head
# --------------------------------------------------

print("\n15. HEAD-TO-HEAD TEST")

h2h = get_head_to_head(
    "Liverpool",
    "Arsenal"
)

print(h2h.head())


# --------------------------------------------------
# 16. Team comparison
# --------------------------------------------------

print("\n16. COMPARISON TEST")

comparison = compare_teams(
    "Liverpool",
    "Arsenal",
    "2025/26"
)

print(comparison)


print("\nVALIDATION COMPLETE")
print("===================")