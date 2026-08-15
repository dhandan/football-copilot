import duckdb
from pathlib import Path


DATABASE_FILE = "database/football.duckdb"

MATCHES_FILE = "data/processed/matches_clean.csv"
TEAM_STATS_FILE = "data/processed/team_match_stats.csv"


print("\nBuilding Football Copilot database...")
print("====================================")


# --------------------------------------------------
# 1. Make sure database folder exists
# --------------------------------------------------

Path("database").mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 2. Connect to DuckDB
# --------------------------------------------------

connection = duckdb.connect(DATABASE_FILE)

print(f"\nConnected to: {DATABASE_FILE}")


# --------------------------------------------------
# 3. Rebuild matches table
# --------------------------------------------------

connection.execute("""
    DROP TABLE IF EXISTS matches
""")


connection.execute(f"""
    CREATE TABLE matches AS
    SELECT *
    FROM read_csv_auto(
        '{MATCHES_FILE}',
        HEADER = TRUE
    )
""")


print("Created table: matches")


# --------------------------------------------------
# 4. Rebuild team_match_stats table
# --------------------------------------------------

connection.execute("""
    DROP TABLE IF EXISTS team_match_stats
""")


connection.execute(f"""
    CREATE TABLE team_match_stats AS
    SELECT *
    FROM read_csv_auto(
        '{TEAM_STATS_FILE}',
        HEADER = TRUE
    )
""")


print("Created table: team_match_stats")


# --------------------------------------------------
# 5. Check row counts
# --------------------------------------------------

matches_count = connection.execute("""
    SELECT COUNT(*)
    FROM matches
""").fetchone()[0]


team_stats_count = connection.execute("""
    SELECT COUNT(*)
    FROM team_match_stats
""").fetchone()[0]


print("\nROW COUNTS")
print("==========")

print(f"matches:          {matches_count}")
print(f"team_match_stats: {team_stats_count}")


# --------------------------------------------------
# 6. Check the expected relationship
# --------------------------------------------------

print(
    "\nTeam rows = matches x 2:",
    team_stats_count == matches_count * 2
)


# --------------------------------------------------
# 7. Show available tables
# --------------------------------------------------

tables = connection.execute("""
    SHOW TABLES
""").fetchdf()


print("\nDATABASE TABLES")
print("===============")

print(tables)


# --------------------------------------------------
# 8. Close database
# --------------------------------------------------

connection.close()

print("\nDatabase build complete.")