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

def frozen_gameweek_predictions(gameweek: int = 8):
    """Read previously frozen, immutable match forecasts. No live recomputation."""
    from pathlib import Path
    import pandas as pd

    gw = int(gameweek)
    if not 1 <= gw <= 38:
        return {"type": "frozen_gameweek_predictions", "error": "Gameweek must be 1 to 38."}
    root = Path(__file__).resolve().parent.parent / "data" / "live"
    prefix = f"2026_27_gw{gw:02d}"
    base_path = root / "predictions" / f"{prefix}_predictions.csv"
    if not base_path.exists():
        return {"type": "frozen_gameweek_predictions", "gameweek": gw, "error": "No frozen forecasts found for this gameweek."}
    model = pd.read_csv(base_path)
    if len(model) != 10:
        return {"type": "frozen_gameweek_predictions", "gameweek": gw, "error": "Incomplete frozen benchmark; refusing to present."}
    keys = ["FixtureDate", "HomeTeam", "AwayTeam"]
    if model[keys].duplicated().any():
        return {"type": "frozen_gameweek_predictions", "gameweek": gw, "error": "Duplicate benchmark fixtures."}
    blend_path = root / "challenger_predictions" / f"{prefix}_market_shadow_predictions.csv"
    adaptive_path = root / "challenger_predictions" / f"{prefix}_adaptive_confidence_v1.csv"
    combined = model.copy()
    variants = []
    if blend_path.exists():
        blend = pd.read_csv(blend_path)
        if len(blend) != 10 or blend[keys].duplicated().any():
            return {"type": "frozen_gameweek_predictions", "gameweek": gw, "error": "Incomplete or duplicate 40/60 forecast."}
        combined = combined.merge(
            blend[keys + ["ShadowHomeProbability", "ShadowDrawProbability", "ShadowAwayProbability", "ShadowPredictedResult"]],
            on=keys, validate="one_to_one", how="left"
        )
        if combined["ShadowHomeProbability"].isna().any():
            return {"type": "frozen_gameweek_predictions", "gameweek": gw, "error": "Champion fixtures do not match benchmark."}
        variants.append("Model2_OpeningMarket_40_60_v1.0")
    if adaptive_path.exists():
        adaptive = pd.read_csv(adaptive_path)
        if len(adaptive) != 10 or adaptive[keys].duplicated().any():
            return {"type": "frozen_gameweek_predictions", "gameweek": gw, "error": "Incomplete adaptive forecast."}
        combined = combined.merge(
            adaptive[keys + ["HomeProbability", "DrawProbability", "AwayProbability", "PredictedResult"]].rename(columns={
                "HomeProbability": "AdaptiveHomeProbability", "DrawProbability": "AdaptiveDrawProbability",
                "AwayProbability": "AdaptiveAwayProbability", "PredictedResult": "AdaptivePredictedResult"
            }),
            on=keys, validate="one_to_one", how="left"
        )
        if combined["AdaptiveHomeProbability"].isna().any():
            return {"type": "frozen_gameweek_predictions", "gameweek": gw, "error": "Adaptive fixtures do not match benchmark."}
        variants.append("AdaptiveMarketConfidence_v1.0")
    columns = keys + ["ModelVersion", "PredictedResult", "MostLikelyScore",
                      "HomeWinProbability", "DrawProbability", "AwayWinProbability"]
    columns += [c for c in ["ShadowHomeProbability", "ShadowDrawProbability", "ShadowAwayProbability",
                             "ShadowPredictedResult", "AdaptiveHomeProbability", "AdaptiveDrawProbability",
                             "AdaptiveAwayProbability", "AdaptivePredictedResult"] if c in combined]
    output = combined[columns].where(pd.notnull(combined[columns]), None)
    return {
        "type": "frozen_gameweek_predictions",
        "gameweek": gw,
        "source": str(base_path.relative_to(root.parent.parent)),
        "models": ["Model2_v1.0"] + variants,
        "note": "Model 2 supplies historical most-likely scorelines; blended forecasts do not generate their own exact scores. All probabilities come from frozen files.",
        "data": output.to_dict(orient="records")
    }
