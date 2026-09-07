from pathlib import Path
import sys

import duckdb


# ==================================================
# PROJECT ROOT
# ==================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.append(
    str(PROJECT_ROOT)
)


# ==================================================
# IMPORT ANALYTICS FUNCTIONS
# ==================================================

from analytics.football_analytics import (
    get_teams,
    get_seasons,
    get_league_table,
    get_team_record,
    get_team_form,
    get_form_summary,
    get_home_away_record,
    get_head_to_head,
    compare_teams,
)


# ==================================================
# REQUIRED RUNTIME FILES
# ==================================================

DATABASE_FILE = (
    PROJECT_ROOT
    / "database"
    / "football.duckdb"
)

MATCH_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "matches_clean.csv"
)

PROMOTED_PRIORS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "promoted_team_priors_2026_27.csv"
)

PRODUCTION_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "production_model_v2.pkl"
)


# ==================================================
# VALIDATION STATE
# ==================================================

validation_failures = []


def check(
    name,
    condition,
):
    """
    Record and display a validation result.
    """

    passed = bool(condition)

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    print(
        f"{status}: {name}"
    )

    if not passed:
        validation_failures.append(
            name
        )

    return passed


# ==================================================
# START
# ==================================================

print()
print("=" * 70)
print("FOOTBALL COPILOT")
print("PROJECT VALIDATION")
print("=" * 70)

print()
print(
    f"Project root: "
    f"{PROJECT_ROOT}"
)

print(
    f"Python: "
    f"{sys.executable}"
)


# ==================================================
# 1. RUNTIME FILES
# ==================================================

print()
print("=" * 70)
print("1. REQUIRED RUNTIME FILES")
print("=" * 70)

runtime_files = {
    "DuckDB analytics database":
        DATABASE_FILE,

    "Clean historical match data":
        MATCH_FILE,

    "2026/27 promoted-team priors":
        PROMOTED_PRIORS_FILE,

    "Frozen production Model 2":
        PRODUCTION_MODEL_FILE,
}

for name, file_path in runtime_files.items():

    exists = file_path.exists()

    check(
        name,
        exists,
    )

    print(
        f"    {file_path}"
    )


# ==================================================
# STOP IF CORE FILES ARE MISSING
# ==================================================

if validation_failures:

    print()
    print("=" * 70)
    print("VALIDATION FAILED")
    print("=" * 70)

    print()
    print(
        "Football Copilot is missing one or more "
        "required runtime files."
    )

    print()
    print(
        "Missing or invalid checks:"
    )

    for failure in validation_failures:
        print(
            f"- {failure}"
        )

    print()
    print(
        "If this is a development rebuild, run:"
    )

    print()
    print(
        "python scripts/build_project.py"
    )

    print()

    sys.exit(1)


# ==================================================
# 2. DATABASE TABLES
# ==================================================

print()
print("=" * 70)
print("2. DATABASE STRUCTURE")
print("=" * 70)

connection = duckdb.connect(
    str(DATABASE_FILE),
    read_only=True,
)

tables = connection.execute(
    """
    SHOW TABLES
    """
).fetchdf()

print()
print(tables)

table_names = set(
    tables.iloc[:, 0].tolist()
)

check(
    "matches table exists",
    "matches" in table_names,
)

check(
    "team_match_stats table exists",
    "team_match_stats" in table_names,
)


# ==================================================
# 3. ROW COUNTS
# ==================================================

print()
print("=" * 70)
print("3. ROW COUNTS")
print("=" * 70)

matches_count = connection.execute(
    """
    SELECT COUNT(*)
    FROM matches
    """
).fetchone()[0]

team_count = connection.execute(
    """
    SELECT COUNT(*)
    FROM team_match_stats
    """
).fetchone()[0]

print()
print(
    f"Matches: "
    f"{matches_count}"
)

print(
    f"Team-match rows: "
    f"{team_count}"
)

check(
    "team_match_stats has two rows per match",
    team_count
    ==
    matches_count * 2,
)


# ==================================================
# 4. MISSING CRITICAL DATA
# ==================================================

print()
print("=" * 70)
print("4. MISSING CRITICAL DATA")
print("=" * 70)

missing = connection.execute(
    """
    SELECT
        SUM(
            CASE
                WHEN Team IS NULL
                THEN 1
                ELSE 0
            END
        ) AS MissingTeams,

        SUM(
            CASE
                WHEN Opponent IS NULL
                THEN 1
                ELSE 0
            END
        ) AS MissingOpponents,

        SUM(
            CASE
                WHEN Result IS NULL
                THEN 1
                ELSE 0
            END
        ) AS MissingResults

    FROM team_match_stats
    """
).fetchdf()

print()
print(missing)

check(
    "no missing teams",
    missing.loc[
        0,
        "MissingTeams",
    ]
    == 0,
)

check(
    "no missing opponents",
    missing.loc[
        0,
        "MissingOpponents",
    ]
    == 0,
)

check(
    "no missing results",
    missing.loc[
        0,
        "MissingResults",
    ]
    == 0,
)


# ==================================================
# 5. GOAL RECONCILIATION
# ==================================================

print()
print("=" * 70)
print("5. GOAL RECONCILIATION")
print("=" * 70)

goal_check = connection.execute(
    """
    SELECT
        SUM(GoalsFor)
            AS GoalsFor,

        SUM(GoalsAgainst)
            AS GoalsAgainst

    FROM team_match_stats
    """
).fetchdf()

print()
print(goal_check)

check(
    "goals for reconcile with goals against",
    goal_check.loc[
        0,
        "GoalsFor",
    ]
    ==
    goal_check.loc[
        0,
        "GoalsAgainst",
    ],
)


# ==================================================
# 6. VALID RESULT VALUES
# ==================================================

print()
print("=" * 70)
print("6. RESULT VALUES")
print("=" * 70)

results = connection.execute(
    """
    SELECT DISTINCT Result
    FROM team_match_stats
    ORDER BY Result
    """
).fetchdf()

print()
print(results)

result_values = set(
    results[
        "Result"
    ].tolist()
)

check(
    "result values are valid",
    result_values
    ==
    {
        "D",
        "L",
        "W",
    },
)


# ==================================================
# 7. VALID POINT VALUES
# ==================================================

print()
print("=" * 70)
print("7. POINT VALUES")
print("=" * 70)

points = connection.execute(
    """
    SELECT DISTINCT Points
    FROM team_match_stats
    ORDER BY Points
    """
).fetchdf()

print()
print(points)

point_values = set(
    points[
        "Points"
    ].tolist()
)

check(
    "point values are valid",
    point_values
    ==
    {
        0,
        1,
        3,
    },
)


# ==================================================
# 8. DUPLICATES
# ==================================================

print()
print("=" * 70)
print("8. DUPLICATE CHECK")
print("=" * 70)

duplicates = connection.execute(
    """
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
    """
).fetchdf()

print()
print(
    f"Duplicate records: "
    f"{len(duplicates)}"
)

check(
    "no duplicate team-match rows",
    len(duplicates)
    == 0,
)


# ==================================================
# 9. MATCHES BY SEASON
# ==================================================

print()
print("=" * 70)
print("9. MATCHES BY SEASON")
print("=" * 70)

matches_by_season = connection.execute(
    """
    SELECT
        Season,
        COUNT(*) / 2 AS Matches

    FROM team_match_stats

    GROUP BY Season

    ORDER BY Season
    """
).fetchdf()

print()
print(matches_by_season)

connection.close()


# ==================================================
# 10. ANALYTICS FUNCTIONS
# ==================================================

print()
print("=" * 70)
print("10. ANALYTICS FUNCTION TESTS")
print("=" * 70)

try:

    seasons = get_seasons()

    check(
        "get_seasons",
        len(seasons) > 0,
    )

except Exception as exc:

    print(
        f"get_seasons error: "
        f"{exc}"
    )

    validation_failures.append(
        "get_seasons"
    )


try:

    teams = get_teams()

    check(
        "get_teams",
        len(teams) > 0,
    )

except Exception as exc:

    print(
        f"get_teams error: "
        f"{exc}"
    )

    validation_failures.append(
        "get_teams"
    )


try:

    league_table = get_league_table(
        "2025/26"
    )

    check(
        "get_league_table",
        len(league_table) > 0,
    )

except Exception as exc:

    print(
        f"get_league_table error: "
        f"{exc}"
    )

    validation_failures.append(
        "get_league_table"
    )


try:

    record = get_team_record(
        "Liverpool",
        "2025/26",
    )

    check(
        "get_team_record",
        record is not None,
    )

except Exception as exc:

    print(
        f"get_team_record error: "
        f"{exc}"
    )

    validation_failures.append(
        "get_team_record"
    )


try:

    form = get_team_form(
        "Liverpool",
        5,
    )

    check(
        "get_team_form",
        form is not None,
    )

except Exception as exc:

    print(
        f"get_team_form error: "
        f"{exc}"
    )

    validation_failures.append(
        "get_team_form"
    )


try:

    form_summary = get_form_summary(
        "Liverpool",
        5,
    )

    check(
        "get_form_summary",
        form_summary is not None,
    )

except Exception as exc:

    print(
        f"get_form_summary error: "
        f"{exc}"
    )

    validation_failures.append(
        "get_form_summary"
    )


try:

    home_away = get_home_away_record(
        "Liverpool",
        "2025/26",
    )

    check(
        "get_home_away_record",
        home_away is not None,
    )

except Exception as exc:

    print(
        f"get_home_away_record error: "
        f"{exc}"
    )

    validation_failures.append(
        "get_home_away_record"
    )


try:

    h2h = get_head_to_head(
        "Liverpool",
        "Arsenal",
    )

    check(
        "get_head_to_head",
        h2h is not None,
    )

except Exception as exc:

    print(
        f"get_head_to_head error: "
        f"{exc}"
    )

    validation_failures.append(
        "get_head_to_head"
    )


try:

    comparison = compare_teams(
        "Liverpool",
        "Arsenal",
        "2025/26",
    )

    check(
        "compare_teams",
        comparison is not None,
    )

except Exception as exc:

    print(
        f"compare_teams error: "
        f"{exc}"
    )

    validation_failures.append(
        "compare_teams"
    )


# ==================================================
# FINAL RESULT
# ==================================================

print()
print("=" * 70)

if validation_failures:

    print("FOOTBALL COPILOT VALIDATION FAILED")
    print("=" * 70)

    print()
    print(
        "Failed checks:"
    )

    for failure in validation_failures:
        print(
            f"- {failure}"
        )

    print()

    sys.exit(1)


print("FOOTBALL COPILOT VALIDATION PASSED")
print("=" * 70)

print()
print(
    "The project has the required runtime files "
    "and the core analytics checks passed."
)

print()
print(
    "Start Football Copilot with:"
)

print()
print(
    "streamlit run app/streamlit_app.py"
)

print()

sys.exit(0)
