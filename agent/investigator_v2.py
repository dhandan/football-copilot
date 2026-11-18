"""Experimental bounded investigator for Football Copilot V2.

Unlike the existing one-shot router, this prototype proposes a read-only
tool plan, checks coverage, and permits ONE further planning round for
missing evidence. Python tools remain the only sources of football facts.

This is NOT connected to the production or private Streamlit preview.
No Groq request occurs unless a caller explicitly passes a Groq planner.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from agent import langgraph_v2 as v2

MAX_ROUNDS = 2
MAX_TOOLS = 6
SAFE_TOOLS = frozenset(("form_summary", "head_to_head", "compare_opponent_strength"))
_WINDOW = re.compile(
    r"\b(?:last|past|previous|those|these)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"(?:\s+(?:Premier League|league|PL))?(?:\s+(?:matches|games))?\b", re.I
)
_H2H = re.compile(r"\b(?:head[- ]to[- ]head|h2h)\b", re.I)
_OPPOSITION = re.compile(
    r"\b(?:opponents?|opposition|fixture difficulty|strength of schedule|"
    r"weaker (?:teams|opposition)|stronger (?:teams|opposition))\b", re.I
)
_REJECT = re.compile(r"\b(?:delete|erase|wipe|overwrite|retrain|destroy|drop)\b", re.I)
_OFFTOPIC = re.compile(r"\b(?:weather|temperature|rainfall|currency|stock price)\b", re.I)
_REFERENTIAL = re.compile(
    r"\b(?:their|they|them|both|those|these|that|which of|what about|"
    r"how about|opponents faced|the difference|why is that)\b", re.I
)


@dataclass(frozen=True)
class Scope:
    teams: tuple[str, ...]
    games: int | None
    h2h: bool
    opposition: bool
    venues: bool


def _teams(question: str) -> tuple[str, ...]:
    # Preserve the speaker's team order rather than the catalogue's sort
    # order. Prefer longer aliases when they overlap at the same position.
    mentions = []
    for label in v2._available_team_names():
        for match in re.finditer(
            r"(?<!\w)" + re.escape(label) + r"(?!\w)", question, re.I
        ):
            canonical = v2.TEAM_ALIASES.get(label.lower(), label)
            mentions.append((match.start(), -len(label), canonical))
    seen = []
    for _, _, team in sorted(mentions):
        if team not in seen:
            seen.append(team)
    return tuple(seen)


def parse_scope(question: str, previous: Scope | None = None) -> Scope:
    teams = _teams(question)
    match = _WINDOW.search(question)
    games = None
    if match:
        token = match.group(1).lower()
        games = {"one": 1, "two": 2, "three": 3, "four": 4,
                 "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10}.get(token)
        if games is None:
            games = int(token)
    # Explicit new entities always take precedence. Never silently merge teams
    # from an unrelated previous question.
    if previous and _REFERENTIAL.search(question):
        if not teams:
            teams = previous.teams
        if games is None and teams == previous.teams:
            games = previous.games
    return Scope(teams, games, bool(_H2H.search(question)),
                 bool(_OPPOSITION.search(question)),
                 bool(re.search(r"\b(?:home|away)\b", question, re.I)))


def requirements(scope: Scope) -> list[dict]:
    if not (1 <= len(scope.teams) <= 2):
        return []
    needed = []
    if scope.games is not None and 1 <= scope.games <= 30:
        needed.extend({"name": "form_summary",
                       "arguments": {"team": team, "games": scope.games}}
                      for team in scope.teams)
    if len(scope.teams) == 2:
        if scope.h2h:
            needed.append({"name": "head_to_head", "arguments": {
                "team1": scope.teams[0], "team2": scope.teams[1]}})
        if scope.opposition and scope.games is not None and 1 <= scope.games <= 30:
            needed.append({"name": "compare_opponent_strength", "arguments": {
                "team1": scope.teams[0], "team2": scope.teams[1],
                "games": scope.games}})
    # No fabricated default last-N window for unqualified open questions.
    return needed[:MAX_TOOLS]


def call_key(call: dict) -> tuple:
    return call["name"], tuple(sorted(call["arguments"].items()))


def validated_proposals(raw, needed: list[dict]) -> list[dict]:
    """The model can choose evidence, never gain access to other tools."""
    try:
        parsed = v2._Plan.model_validate({"calls": raw})
    except (ValueError, TypeError):
        return []
    authorised = {call_key(item) for item in needed}
    result = []
    for tool in parsed.calls:
        call = tool.model_dump()
        if (call["name"] in SAFE_TOOLS and call_key(call) in authorised
                and call not in result):
            result.append(call)
    return result


def _execute(call: dict, tools: dict) -> dict:
    name, args = call["name"], call["arguments"]
    try:
        if name == "form_summary" and tools[name] is v2.form_summary:
            result = tools[name](**args, season="2026/27")
        else:
            result = tools[name](**args)
        if not isinstance(result, dict) or result.get("error"):
            raise ValueError("Unavailable tool evidence")
        if name in ("form_summary", "head_to_head") and not result.get("data"):
            raise ValueError("Missing verified rows")
        if name == "form_summary" and (
            len(result["data"]) != 1 or
            int(result["data"][0].get("Played", 0)) < args["games"]
        ):
            raise ValueError("Insufficient completed match coverage")
        if name == "compare_opponent_strength":
            rows = result.get("data", {})
            if not isinstance(rows, dict) or any(
                rows.get(team, {}).get("rated", 0) < 3
                for team in (args["team1"], args["team2"])
            ):
                raise ValueError("Insufficient opponent-history coverage")
        return {"tool": name, "arguments": args, "result": result}
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return {"tool": name, "arguments": args,
                "error": "Verified evidence unavailable: " + type(exc).__name__}
    except Exception as exc:
        # Never expose exception messages, filesystem paths or credentials.
        return {"tool": name, "arguments": args,
                "error": "Tool failed: " + type(exc).__name__}


def render_evidence(scope: Scope, evidence: list[dict], gaps: list[dict]) -> str:
    """Only deterministic tool values become factual claims."""
    lines = ["### Verified football evidence"]
    forms = []
    for item in evidence:
        if "result" not in item:
            continue
        result = item["result"]
        if item["tool"] == "form_summary":
            row = result["data"][0]
            if not all(k in row for k in ("Played", "Points", "PointsPerGame",
                                          "GoalsFor", "GoalsAgainst")):
                continue
            forms.append((item["arguments"]["team"], row, result.get(
                "source", "verified form tool")))
    if forms:
        lines += ["", "| Team | Played | Points | PPG | Goals for | Goals against |",
                  "|:--|--:|--:|--:|--:|--:|"]
        for team, row, _ in forms:
            lines.append(
                f"| {team} | {int(row['Played'])} | {int(row['Points'])} | "
                f"{float(row['PointsPerGame']):.2f} | {int(row['GoalsFor'])} | "
                f"{int(row['GoalsAgainst'])} |"
            )
        if len(forms) == 2:
            (ta, a, _), (tb, b, _) = forms
            if int(a["Played"]) == int(b["Played"]):
                diff = abs(int(a["GoalsAgainst"]) - int(b["GoalsAgainst"]))
                if diff:
                    better = ta if a["GoalsAgainst"] < b["GoalsAgainst"] else tb
                    lines.append(f"\n**{better} conceded {diff} fewer goals** "
                                 "in the measured window. Descriptive, not causal.")
                else:
                    lines.append("\nBoth teams conceded the same number of goals.")
        if scope.venues:
            for team, row, _ in forms:
                if (row.get("VenueKnown") != row.get("Played")
                        or not isinstance(row.get("VenueSplit"), dict)):
                    lines.append(f"\n{team}: verified venue breakdown unavailable.")
                    continue
                for venue in ("Home", "Away"):
                    venue_row = row["VenueSplit"].get(venue, {})
                    if venue_row.get("Played"):
                        n = int(venue_row["Played"])
                        ga = int(venue_row["GoalsAgainst"])
                        lines.append(f"\n{team}, {venue.lower()}: {ga} goals "
                                     f"conceded in {n} matches ({ga/n:.2f} per match).")
        sources = "; ".join(dict.fromkeys(source for _, _, source in forms))
        lines.append(f"\n*Form source: {sources}. Completed results only.*")
    for item in evidence:
        if "result" not in item:
            continue
        result = item["result"]
        if item["tool"] == "head_to_head":
            rows = result["data"]
            lines += ["", "**Historical head-to-head results**",
                      "| Date | Venue (first team) | Goals for-against (first team) |",
                      "|:--|:--|:--|"]
            n = min(scope.games or 5, 30)
            for row in rows[:n]:
                lines.append(f"| {str(row['Date'])[:10]} | {row['Venue']} | "
                             f"{int(row['GoalsFor'])}-{int(row['GoalsAgainst'])} |")
            lines.append("\n*Source: historical Premier League DuckDB H2H, "
                         "not verified current-season snapshots.*")
        if item["tool"] == "compare_opponent_strength":
            data = result["data"]
            lines += ["", "**Pre-match opponent form**",
                      "| Team | Rated fixtures | Opponents' prior PPG |",
                      "|:--|--:|--:|"]
            for team in scope.teams:
                row = data.get(team, {})
                ppg = row.get("mean_opponent_prior_ppg")
                ppg_text = f"{float(ppg):.2f}" if ppg is not None else "unavailable"
                lines.append(f"| {team} | {row.get('rated', 0)}/{row.get('played', 0)} "
                             f"| {ppg_text} |")
            lines.append("\n*Opponent PPG is an observational pre-match proxy, "
                         "not a validated defensive adjustment or causal finding. "
                         f"Source: {result.get('source', 'read-only opponent tool')}.*")
    if gaps:
        lines.append("\n**Unresolved evidence:** " + ", ".join(
            f"{c['name']} ({', '.join(str(v) for v in c['arguments'].values())})"
            for c in gaps
        ) + ". No unsupported conclusion is inferred.")
    for item in evidence:
        if "error" in item:
            lines.append(f"\n**Unavailable:** {item['tool']} — {item['error']}.")
    if not evidence:
        lines.append("\nNo verified football evidence obtained.")
    return "\n".join(lines)


class Investigator:
    """Plan -> execute -> assess coverage -> optionally plan missing evidence."""

    def __init__(self, planner: Callable[[str], list[dict]],
                 tools: dict | None = None):
        self.planner = planner
        source = tools if tools is not None else v2.ALLOWED
        self.tools = {name: source[name] for name in SAFE_TOOLS if name in source}
        self.context: dict[str, Scope] = {}

    def ask(self, question: str, thread_id: str = "default") -> dict:
        if not question.strip() or not thread_id.strip():
            raise ValueError("Question and thread_id must be non-empty")
        if _REJECT.search(question) or _OFFTOPIC.search(question):
            return {"answer": "Outside read-only football investigation scope.",
                    "evidence": [], "gaps": [], "trace": ["scope_rejected"],
                    "complete": False, "rounds": 0}
        scope = parse_scope(question, self.context.get(thread_id))
        needed = requirements(scope)
        if not needed:
            return {"answer": "Cannot resolve the requested evidence. Specify "
                    "one or two Premier League teams and a last-N match window "
                    "(1–30), or ask for historical head-to-head results.",
                    "evidence": [], "gaps": [], "trace": ["scope_unresolved"],
                    "complete": False, "rounds": 0}
        self.context[thread_id] = scope
        tried, evidence, trace, rounds = set(), [], [], 0
        for round_idx in range(MAX_ROUNDS):
            missing = [c for c in needed if call_key(c) not in tried]
            if not missing or len(tried) >= MAX_TOOLS:
                break
            rounds += 1
            instruction = (
                "Select read-only Premier League evidence tools. Return only "
                "a list of {name,arguments} calls (max 3). Do not answer, invent "
                "statistics, request destructive tools or change source data. "
                f"User question: {question!r}. Teams: {scope.teams}. "
                f"Last-N window: {scope.games}. "
                f"Required evidence categories still missing: "
                f"{[c['name'] for c in missing]}. "
                f"Previously successful tools: "
                f"{[e['tool'] for e in evidence if 'result' in e]}. "
                "Use exact team names and match window."
            )
            try:
                proposals = validated_proposals(self.planner(instruction), missing)
            except (ValueError, TypeError, TimeoutError) as exc:
                proposals = []
                trace.append("planner_rejected:" + type(exc).__name__)
            except Exception as exc:
                proposals = []
                trace.append("planner_failed:" + type(exc).__name__)
            trace.append(f"round_{round_idx+1}_approved_{len(proposals)}")
            for call in proposals[:3]:
                key = call_key(call)
                if key in tried:
                    continue
                tried.add(key)
                item = _execute(call, self.tools)
                evidence.append(item)
                trace.append(f"executed_{item['tool']}" if "result" in item
                             else f"unavailable_{item['tool']}")
        complete_keys = {(e["tool"], tuple(sorted(e["arguments"].items())))
                         for e in evidence if "result" in e}
        gaps = [c for c in needed if call_key(c) not in complete_keys]
        trace.append("complete" if not gaps else "partial_evidence")
        return {"answer": render_evidence(scope, evidence, gaps),
                "evidence": evidence, "gaps": gaps, "trace": trace,
                "complete": not gaps, "rounds": rounds}
