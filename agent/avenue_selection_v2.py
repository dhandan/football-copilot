"""Experimental choice between two genuinely different context investigations.

This module is NOT wired to the production copilot or the existing lab.
The model chooses one avenue from two grounded options; Python executes and
checks results. Passing a choice task is not evidence of causal explanations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from agent import groq_v2
from agent.investigator_v2 import _execute, parse_scope
from agent import langgraph_v2 as v2


@dataclass(frozen=True)
class AvenueCase:
    label: str
    question: str
    expected: str


CASES = (
    AvenueCase("arsenal_liverpool_venue",
               "Across Arsenal and Liverpool's last seven matches, could their "
               "home versus away defensive splits be relevant to the difference?",
               "venue"),
    AvenueCase("brighton_brentford_opposition",
               "For Brighton and Brentford's last seven games, compare the "
               "strength of the opponents they faced when discussing scoring.",
               "opposition"),
    AvenueCase("chelsea_tottenham_venue",
               "Does a home versus away breakdown shed light on Chelsea and "
               "Tottenham's goals conceded in their last eight matches?",
               "venue"),
    AvenueCase("villa_newcastle_opposition",
               "For Aston Villa and Newcastle over their last six games, "
               "did one side face a tougher set of opponents?",
               "opposition"),
)

CHOICES = ("venue", "opposition", "none")
STRICT_CHOICE = {
    "type": "json_schema",
    "json_schema": {
        "name": "investigation_avenue_choice",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "avenue": {"type": "string", "enum": list(CHOICES)},
            },
            "required": ["avenue"],
            "additionalProperties": False,
        },
    },
}


def observed_form(question: str):
    scope = parse_scope(question)
    if (len(scope.teams) != 2 or scope.games is None
            or not 3 <= scope.games <= 30):
        raise ValueError("Requires exactly two clubs and a 3-30 game window")
    calls = [{"name": "form_summary",
              "arguments": {"team": team, "games": scope.games}}
             for team in scope.teams]
    evidence = [_execute(call, v2.ALLOWED) for call in calls]
    if any("result" not in e for e in evidence):
        raise ValueError("Verified first-round form unavailable")
    rows = [e["result"]["data"][0] for e in evidence]
    if any(int(r["Played"]) != scope.games for r in rows):
        raise ValueError("Incomplete form coverage")
    return scope, evidence


def venue_evidence(scope, evidence):
    """Check existing independent home/away measurements, never infer missing."""
    rows = [e["result"]["data"][0] for e in evidence]
    metric = "GoalsFor" if scope.focus == "attack" else "GoalsAgainst"
    result = []
    for team, row in zip(scope.teams, rows):
        splits = row.get("VenueSplit")
        if row.get("VenueKnown") != row["Played"] or not isinstance(splits, dict):
            raise ValueError("Unverified venue coverage")
        totals = []
        for venue in ("Home", "Away"):
            value = splits.get(venue)
            if not isinstance(value, dict):
                raise ValueError("Missing venue data")
            n, goals = int(value["Played"]), int(value[metric])
            if n <= 0 or goals < 0:
                raise ValueError("Unrated venue evidence")
            totals.append((venue, n, goals))
        if (sum(n for _, n, _ in totals) != row["Played"]
                or sum(g for _, _, g in totals) != int(row[metric])):
            raise ValueError("Unreconciled venue values")
        result.append({"team": team, "metric": metric,
                       "venue_rates": [
                           {"venue": venue, "games": n, "goals": g,
                            "per_game": round(g / n, 3)}
                           for venue, n, g in totals]})
    return {"type": "verified_venue_context",
            "source": "same completed-match records as first-round form",
            "data": result}


def opponent_evidence(scope):
    call = {"name": "compare_opponent_strength", "arguments": {
        "team1": scope.teams[0], "team2": scope.teams[1], "games": scope.games}}
    item = _execute(call, v2.ALLOWED)
    if "result" not in item:
        raise ValueError("Insufficient verified pre-match opponent context")
    return item["result"]


def choice_prompt(question: str, scope, first: list[dict]) -> str:
    metric = "GoalsFor" if scope.focus == "attack" else "GoalsAgainst"
    rows = [e["result"]["data"][0] for e in first]
    facts = [f"{team}: {int(row[metric])} {metric}, "
             f"{int(row['Points'])} points in {int(row['Played'])} matches"
             for team, row in zip(scope.teams, rows)]
    return ("User question: " + question + "\nVerified first-round evidence: " +
            "; ".join(facts) +
            "\nSelect ONE most relevant further evidence avenue, or none. "
            "Options: venue = home/away goals-per-match splits from completed "
            "matches; opposition = opponents' pre-match points-per-match proxy. "
            "Choose based on what the user wants investigated, not which team "
            "you prefer. Neither avenue establishes tactical causes. "
            "Return only the JSON selection.")


def select_live(question, scope, first):
    """One explicitly opted-in Groq request; no extra retries."""
    if not groq_v2.groq_ready():
        raise ValueError("Groq Free/ZDR/key opt-in required")
    system = ("Select a read-only evidence investigation avenue. "
              "Return a strict JSON object with just 'avenue', choosing "
              "venue, opposition, or none. No football claims. No tools or "
              "other text. You cannot claim causation.")
    response = groq_v2._chat(system, choice_prompt(question, scope, first),
                             STRICT_CHOICE)
    try:
        data = json.loads(response)
        if isinstance(data, dict) and set(data) == {"avenue"}:
            if data["avenue"] in CHOICES:
                return data["avenue"]
    except (ValueError, TypeError):
        pass
    raise ValueError("Selection contract rejected")


def select_keyword(question):
    """Simple deterministic comparator; deliberately no extra information."""
    text = question.lower()
    if "home" in text or "away" in text or "venue" in text:
        return "venue"
    if "opponent" in text or "tougher" in text or "fixture difficulty" in text:
        return "opposition"
    return "none"


def investigate(question: str, avenue: str):
    scope, first = observed_form(question)
    if avenue not in CHOICES:
        raise ValueError("Unapproved selection")
    if avenue == "venue":
        extra = venue_evidence(scope, first)
    elif avenue == "opposition":
        extra = opponent_evidence(scope)
    else:
        extra = None
    return {"scope": scope, "form": first, "avenue": avenue,
            "extra": extra,
            "caveat": ("These are descriptive results, not proof of why either "
                       "team performed differently.")}
