"""Read-only, pre-match opponent-strength context for Premier League V2.

This is an *observational proxy*, not a trained or validated fixture difficulty
model. All rating inputs are completed matches strictly BEFORE the fixture.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from analytics.football_analytics import get_connection


def _history():
    """Load the committed historical team-match table and validated live snapshots."""
    import pandas as pd

    connection = get_connection()
    try:
        history = connection.execute(
            "SELECT Date, Season, Team, Opponent, Venue, GoalsFor, "
            "GoalsAgainst, Points FROM team_match_stats "
            "WHERE Season <> '2026/27'"
        ).fetchdf()
    finally:
        connection.close()

    rows = []
    for row in history.itertuples(index=False):
        venue = {"h": "Home", "home": "Home",
                 "a": "Away", "away": "Away"}.get(str(row.Venue).strip().lower())
        rows.append({
            "date": str(row.Date)[:10], "team": str(row.Team),
            "opponent": str(row.Opponent), "venue": venue,
            "gf": int(row.GoalsFor), "ga": int(row.GoalsAgainst),
            "points": int(row.Points), "season": str(row.Season),
        })

    folder = Path(__file__).resolve().parent.parent / "data/live/results"
    seen = set()
    for file in sorted(folder.glob("2026_27_gw??_results.csv")):
        frame = pd.read_csv(file)
        required = {"Season", "FixtureId", "FixtureDate", "HomeTeam",
                    "AwayTeam", "Status", "ActualHomeGoals", "ActualAwayGoals"}
        if not required.issubset(frame.columns):
            continue
        if (len(frame) != 10 or (frame["Status"] != "FINISHED").any()
                or frame[["FixtureId", "FixtureDate"]].duplicated().any()
                or not (frame["Season"] == "2026/27").all()):
            continue
        for item in frame.itertuples(index=False):
            date = str(item.FixtureDate)[:10]
            home, away = str(item.HomeTeam), str(item.AwayTeam)
            key = (date, home, away)
            if key in seen:
                continue
            seen.add(key)
            hg, ag = item.ActualHomeGoals, item.ActualAwayGoals
            if pd.isna(hg) or pd.isna(ag) or int(hg) < 0 or int(ag) < 0:
                continue
            for team, opponent, venue, gf, ga in (
                (home, away, "Home", int(hg), int(ag)),
                (away, home, "Away", int(ag), int(hg)),
            ):
                rows.append({
                    "date": date, "team": team, "opponent": opponent,
                    "venue": venue, "gf": gf, "ga": ga,
                    "points": 3 if gf > ga else 1 if gf == ga else 0,
                    "season": "2026/27",
                })
    return rows


def rate_opponents(rows: list[dict], teams: tuple[str, str], games: int,
                   previous_games: int = 5, minimum_history: int = 3):
    """Return source-only opponent strength for each side's last N fixtures.

    An opponent's PPG and goals scored/match are calculated from at most five
    COMPLETED league matches strictly earlier than that fixture's date.
    Carrying prior-season matches is allowed, but no same-day or later match
    contributes. Missing history produces missing values, never a zero rating.
    """
    if not isinstance(games, int) or not 1 <= games <= 30:
        raise ValueError("games must be between 1 and 30")
    by_team = defaultdict(list)
    for match in rows:
        by_team[match["team"]].append(match)
    for matches in by_team.values():
        matches.sort(key=lambda m: (m["date"], m["opponent"]), reverse=True)

    results = {}
    for team in teams:
        fixtures = by_team.get(team, [])[:games]
        evaluated = []
        for fixture in fixtures:
            prior = [
                m for m in by_team.get(fixture["opponent"], [])
                if m["date"] < fixture["date"]
            ][:previous_games]
            enough = len(prior) >= minimum_history
            evaluated.append({
                "date": fixture["date"], "opponent": fixture["opponent"],
                "venue": fixture["venue"], "goals_conceded": fixture["ga"],
                "prior_games": len(prior),
                "opponent_prior_ppg": sum(m["points"] for m in prior) / len(prior)
                if enough else None,
                "opponent_prior_gf_pg": sum(m["gf"] for m in prior) / len(prior)
                if enough else None,
            })
        rated = [r for r in evaluated if r["opponent_prior_ppg"] is not None]
        baseline = sum(r["opponent_prior_gf_pg"] for r in rated)
        conceded = sum(r["goals_conceded"] for r in rated)
        def ratio(observed, expected):
            return round(observed / expected, 3) if expected > 0 else None
        venue_rows = {}
        for venue in ("Home", "Away"):
            subset = [r for r in rated if r["venue"] == venue]
            venue_baseline = sum(r["opponent_prior_gf_pg"] for r in subset)
            venue_conceded = sum(r["goals_conceded"] for r in subset)
            venue_rows[venue] = {
                "rated": len(subset), "conceded": venue_conceded,
                "opponent_prior_goals": round(venue_baseline, 2),
                "observed_to_prior_goals_ratio": ratio(venue_conceded, venue_baseline),
            }
        results[team] = {
            "played": len(fixtures), "rated": len(rated),
            "mean_opponent_prior_ppg": round(
                sum(r["opponent_prior_ppg"] for r in rated) / len(rated), 3
            ) if rated else None,
            "mean_opponent_prior_gf_pg": round(
                baseline / len(rated), 3
            ) if rated else None,
            "goals_conceded_rated": conceded,
            "opponent_prior_goals_total": round(baseline, 2),
            "observed_to_prior_goals_ratio": ratio(conceded, baseline),
            "venues": venue_rows, "fixtures": evaluated,
        }
    return results


def compare_opponent_strength(team1: str, team2: str, games: int = 10):
    if team1 == team2:
        raise ValueError("Select two different teams")
    if not team1.strip() or not team2.strip():
        raise ValueError("Both teams are required")
    rows = _history()
    return {
        "type": "opponent_strength", "data": rate_opponents(
            rows, (team1, team2), games
        ),
        "source": ("read-only historical DuckDB team_match_stats and verified "
                   "2026/27 completed gameweek snapshots"),
        "method": ("Opponent's preceding five completed Premier League matches "
                   "strictly before each fixture; at least three required. "
                   "Prior season included when needed. No future-match leakage. "
                   "Points per match and goals scored per match are descriptive "
                   "proxies, not validated strength estimates."),
    }
