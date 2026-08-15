from analytics.football_analytics import (
    get_team_record,
    get_team_form,
    get_form_summary,
    get_league_table,
    get_home_away_record,
    get_head_to_head,
    compare_teams,
)

from prediction.fixture_predictor import (
    predict_fixture as run_fixture_prediction
)


def team_record(
    team: str,
    season: str,
):
    """
    Get a team's overall Premier League record
    for a specific season.
    """

    result = get_team_record(
        team,
        season
    )

    return {
        "type":
            "team_record",

        "data":
            result.to_dict(
                orient="records"
            ),
    }


def recent_form(
    team: str,
    games: int = 10,
):
    """
    Get a team's most recent Premier League matches.
    """

    result = get_team_form(
        team,
        games
    )

    return {
        "type":
            "recent_form",

        "data":
            result.to_dict(
                orient="records"
            ),
    }


def form_summary(
    team: str,
    games: int = 10,
):
    """
    Summarise a team's recent performance.
    """

    result = get_form_summary(
        team,
        games
    )

    return {
        "type":
            "form_summary",

        "data":
            result.to_dict(
                orient="records"
            ),
    }


def league_table(
    season: str,
):
    """
    Get the Premier League table for a season.
    """

    result = get_league_table(
        season
    )

    result = result.reset_index()

    return {
        "type":
            "league_table",

        "data":
            result.to_dict(
                orient="records"
            ),
    }


def home_away_record(
    team: str,
    season: str,
):
    """
    Compare a team's home and away performance.
    """

    result = get_home_away_record(
        team,
        season
    )

    return {
        "type":
            "home_away_record",

        "data":
            result.to_dict(
                orient="records"
            ),
    }


def head_to_head(
    team1: str,
    team2: str,
):
    """
    Get historical Premier League meetings
    between two teams.
    """

    result = get_head_to_head(
        team1,
        team2
    )

    return {
        "type":
            "head_to_head",

        "data":
            result.to_dict(
                orient="records"
            ),
    }


def team_comparison(
    team1: str,
    team2: str,
    season: str,
):
    """
    Compare two Premier League teams.
    """

    result = compare_teams(
        team1,
        team2,
        season
    )

    return {
        "type":
            "team_comparison",

        "data":
            result.to_dict(
                orient="records"
            ),
    }


def fixture_prediction(
    home_team: str,
    away_team: str,
):
    """
    Predict the probabilities of a future Premier League
    fixture using the Model 2 Poisson prediction engine.

    Args:
        home_team: The home team.
        away_team: The away team.

    Returns:
        Home win, draw and away win probabilities,
        expected goals and likely scorelines.
    """

    return run_fixture_prediction(
        home_team,
        away_team,
    )