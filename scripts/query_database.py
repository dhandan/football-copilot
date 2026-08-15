import duckdb


connection = duckdb.connect(
    "database/football.duckdb"
)


result = connection.execute("""
    SELECT
        Team,
        COUNT(*) AS Played,
        SUM(Win) AS Won,
        SUM(Draw) AS Drawn,
        SUM(Loss) AS Lost,
        SUM(GoalsFor) AS GoalsFor,
        SUM(GoalsAgainst) AS GoalsAgainst,
        SUM(GoalDifference) AS GoalDifference,
        SUM(Points) AS Points
    FROM team_match_stats
    WHERE Season = '2025/26'
    GROUP BY Team
    ORDER BY
        Points DESC,
        GoalDifference DESC,
        GoalsFor DESC
""").fetchdf()

print(result)


connection.close()