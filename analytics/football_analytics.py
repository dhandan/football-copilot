import duckdb
from difflib import get_close_matches


# --------------------------------------------------
# Database configuration
# --------------------------------------------------

DATABASE_FILE = "database/football.duckdb"


# --------------------------------------------------
# Database connection
# --------------------------------------------------

def get_connection():
    """
    Open a read-only connection to the
    Football Copilot DuckDB database.
    """

    return duckdb.connect(
        DATABASE_FILE,
        read_only=True
    )


# --------------------------------------------------
# Reference data
# --------------------------------------------------

def get_teams():
    """
    Return all unique team names available
    in the database.
    """

    connection = get_connection()

    result = connection.execute("""
        SELECT DISTINCT Team
        FROM team_match_stats
        ORDER BY Team
    """).fetchdf()

    connection.close()

    return result["Team"].tolist()


def get_seasons():
    """
    Return all seasons available in the database.
    """

    connection = get_connection()

    result = connection.execute("""
        SELECT DISTINCT Season
        FROM team_match_stats
        ORDER BY Season
    """).fetchdf()

    connection.close()

    return result["Season"].tolist()


# --------------------------------------------------
# Validation functions
# --------------------------------------------------

def validate_team(team):
    """
    Check whether a team exists in the database.

    If an exact match is not found, suggest
    similar team names where possible.
    """

    teams = get_teams()

    if team in teams:
        return True

    suggestions = get_close_matches(
        team,
        teams,
        n=3,
        cutoff=0.6
    )

    if suggestions:
        raise ValueError(
            f"Team '{team}' was not found. "
            f"Did you mean: {', '.join(suggestions)}?"
        )

    raise ValueError(
        f"Team '{team}' was not found in the database."
    )


def validate_season(season):
    """
    Check whether a season exists in the database.
    """

    seasons = get_seasons()

    if season not in seasons:
        raise ValueError(
            f"Season '{season}' was not found in the database. "
            f"Available seasons: {', '.join(seasons)}"
        )

    return True


def validate_games(games):
    """
    Validate the requested number of recent matches.
    """

    if not isinstance(games, int):
        raise ValueError(
            "Games must be an integer."
        )

    if games <= 0:
        raise ValueError(
            "Games must be greater than zero."
        )

    return True


# --------------------------------------------------
# League table
# --------------------------------------------------

def get_league_table(season):
    """
    Return the league table for a given season.
    """

    validate_season(season)

    connection = get_connection()

    query = """
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
        WHERE Season = ?
        GROUP BY Team
        ORDER BY
            Points DESC,
            GoalDifference DESC,
            GoalsFor DESC
    """

    result = connection.execute(
        query,
        [season]
    ).fetchdf()

    connection.close()

    result.index = result.index + 1
    result.index.name = "Position"

    return result


# --------------------------------------------------
# Team record
# --------------------------------------------------

def get_team_record(team, season=None):
    """
    Return an overall record for a team.

    If a season is supplied, only that season
    is included.

    If no season is supplied, all available
    seasons are included.
    """

    validate_team(team)

    if season:
        validate_season(season)

    connection = get_connection()

    if season:

        query = """
            SELECT
                Team,
                COUNT(*) AS Played,
                SUM(Win) AS Won,
                SUM(Draw) AS Drawn,
                SUM(Loss) AS Lost,
                SUM(GoalsFor) AS GoalsFor,
                SUM(GoalsAgainst) AS GoalsAgainst,
                SUM(GoalDifference) AS GoalDifference,
                SUM(Points) AS Points,

                ROUND(
                    SUM(Points) * 1.0 / COUNT(*),
                    2
                ) AS PointsPerGame,

                ROUND(
                    SUM(GoalsFor) * 1.0 / COUNT(*),
                    2
                ) AS GoalsPerGame,

                ROUND(
                    SUM(GoalsAgainst) * 1.0 / COUNT(*),
                    2
                ) AS GoalsAgainstPerGame,

                ROUND(
                    SUM(Win) * 100.0 / COUNT(*),
                    1
                ) AS WinPercentage

            FROM team_match_stats
            WHERE Team = ?
              AND Season = ?
            GROUP BY Team
        """

        parameters = [
            team,
            season
        ]

    else:

        query = """
            SELECT
                Team,
                COUNT(*) AS Played,
                SUM(Win) AS Won,
                SUM(Draw) AS Drawn,
                SUM(Loss) AS Lost,
                SUM(GoalsFor) AS GoalsFor,
                SUM(GoalsAgainst) AS GoalsAgainst,
                SUM(GoalDifference) AS GoalDifference,
                SUM(Points) AS Points,

                ROUND(
                    SUM(Points) * 1.0 / COUNT(*),
                    2
                ) AS PointsPerGame,

                ROUND(
                    SUM(GoalsFor) * 1.0 / COUNT(*),
                    2
                ) AS GoalsPerGame,

                ROUND(
                    SUM(GoalsAgainst) * 1.0 / COUNT(*),
                    2
                ) AS GoalsAgainstPerGame,

                ROUND(
                    SUM(Win) * 100.0 / COUNT(*),
                    1
                ) AS WinPercentage

            FROM team_match_stats
            WHERE Team = ?
            GROUP BY Team
        """

        parameters = [
            team
        ]

    result = connection.execute(
        query,
        parameters
    ).fetchdf()

    connection.close()

    return result


# --------------------------------------------------
# Recent matches
# --------------------------------------------------

def get_team_form(team, games=10):
    """
    Return a team's most recent matches.

    Default:
        10 matches
    """

    validate_team(team)
    validate_games(games)

    connection = get_connection()

    query = """
        SELECT
            Date,
            Season,
            Opponent,
            Venue,
            GoalsFor,
            GoalsAgainst,
            Result,
            Points
        FROM team_match_stats
        WHERE Team = ?
        ORDER BY Date DESC
        LIMIT ?
    """

    result = connection.execute(
        query,
        [
            team,
            games
        ]
    ).fetchdf()

    connection.close()

    return result


# --------------------------------------------------
# Recent form summary
# --------------------------------------------------

def get_form_summary(team, games=10):
    """
    Summarise a team's performance over its
    most recent number of matches.
    """

    validate_team(team)
    validate_games(games)

    connection = get_connection()

    query = """
        SELECT
            COUNT(*) AS Played,
            SUM(Win) AS Won,
            SUM(Draw) AS Drawn,
            SUM(Loss) AS Lost,
            SUM(GoalsFor) AS GoalsFor,
            SUM(GoalsAgainst) AS GoalsAgainst,
            SUM(GoalDifference) AS GoalDifference,
            SUM(Points) AS Points,

            ROUND(
                SUM(Points) * 1.0 / COUNT(*),
                2
            ) AS PointsPerGame,

            ROUND(
                SUM(GoalsFor) * 1.0 / COUNT(*),
                2
            ) AS GoalsPerGame,

            ROUND(
                SUM(GoalsAgainst) * 1.0 / COUNT(*),
                2
            ) AS GoalsAgainstPerGame,

            ROUND(
                SUM(Win) * 100.0 / COUNT(*),
                1
            ) AS WinPercentage

        FROM (
            SELECT *
            FROM team_match_stats
            WHERE Team = ?
            ORDER BY Date DESC
            LIMIT ?
        )
    """

    result = connection.execute(
        query,
        [
            team,
            games
        ]
    ).fetchdf()

    connection.close()

    return result


# --------------------------------------------------
# Home vs away record
# --------------------------------------------------

def get_home_away_record(team, season):
    """
    Compare a team's home and away performance
    during a given season.
    """

    validate_team(team)
    validate_season(season)

    connection = get_connection()

    query = """
        SELECT
            Venue,
            COUNT(*) AS Played,
            SUM(Win) AS Won,
            SUM(Draw) AS Drawn,
            SUM(Loss) AS Lost,
            SUM(GoalsFor) AS GoalsFor,
            SUM(GoalsAgainst) AS GoalsAgainst,
            SUM(GoalDifference) AS GoalDifference,
            SUM(Points) AS Points,

            ROUND(
                SUM(Points) * 1.0 / COUNT(*),
                2
            ) AS PointsPerGame,

            ROUND(
                SUM(GoalsFor) * 1.0 / COUNT(*),
                2
            ) AS GoalsPerGame,

            ROUND(
                SUM(GoalsAgainst) * 1.0 / COUNT(*),
                2
            ) AS GoalsAgainstPerGame,

            ROUND(
                SUM(Win) * 100.0 / COUNT(*),
                1
            ) AS WinPercentage

        FROM team_match_stats
        WHERE Team = ?
          AND Season = ?
        GROUP BY Venue
        ORDER BY Venue
    """

    result = connection.execute(
        query,
        [
            team,
            season
        ]
    ).fetchdf()

    connection.close()

    return result


# --------------------------------------------------
# Head-to-head
# --------------------------------------------------

def get_head_to_head(team1, team2):
    """
    Return all available matches between two teams
    from team1's perspective.
    """

    validate_team(team1)
    validate_team(team2)

    connection = get_connection()

    query = """
        SELECT
            Season,
            Date,
            Venue,
            Opponent,
            GoalsFor,
            GoalsAgainst,
            Result,
            Points
        FROM team_match_stats
        WHERE Team = ?
          AND Opponent = ?
        ORDER BY Date DESC
    """

    result = connection.execute(
        query,
        [
            team1,
            team2
        ]
    ).fetchdf()

    connection.close()

    return result


# --------------------------------------------------
# Team comparison
# --------------------------------------------------

def compare_teams(team1, team2, season):
    """
    Compare two teams during a given season.
    """

    validate_team(team1)
    validate_team(team2)
    validate_season(season)

    connection = get_connection()

    query = """
        SELECT
            Team,
            COUNT(*) AS Played,
            SUM(Win) AS Won,
            SUM(Draw) AS Drawn,
            SUM(Loss) AS Lost,
            SUM(GoalsFor) AS GoalsFor,
            SUM(GoalsAgainst) AS GoalsAgainst,
            SUM(GoalDifference) AS GoalDifference,
            SUM(Points) AS Points,

            ROUND(
                SUM(Points) * 1.0 / COUNT(*),
                2
            ) AS PointsPerGame,

            ROUND(
                SUM(GoalsFor) * 1.0 / COUNT(*),
                2
            ) AS GoalsPerGame,

            ROUND(
                SUM(GoalsAgainst) * 1.0 / COUNT(*),
                2
            ) AS GoalsAgainstPerGame,

            ROUND(
                SUM(Win) * 100.0 / COUNT(*),
                1
            ) AS WinPercentage

        FROM team_match_stats
        WHERE Team IN (?, ?)
          AND Season = ?
        GROUP BY Team
        ORDER BY
            Points DESC,
            GoalDifference DESC,
            GoalsFor DESC
    """

    result = connection.execute(
        query,
        [
            team1,
            team2,
            season
        ]
    ).fetchdf()

    connection.close()

    return result