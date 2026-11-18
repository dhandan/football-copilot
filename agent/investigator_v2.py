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
from agent.groq_v2 import GroqRateLimited

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
    focus: str = "defence"
    trend: bool = False


def _teams(question: str) -> tuple[str, ...]:
    # Preserve the speaker's team order rather than the catalogue's sort
    # order. Prefer longer aliases when they overlap at the same position.
    mentions = []
    catalog = v2._available_team_names()
    # A user says Nottingham Forest; historical data commonly abbreviates
    # this as Nott'm Forest. Resolve to the name actually in the dataset.
    alternate = {"Nottingham Forest": "Nott'm Forest",
                 "Manchester City": "Man City",
                 "Manchester United": "Man United",
                 "Tottenham Hotspur": "Tottenham"}
    for label in list(catalog) + list(alternate):
        for match in re.finditer(
            r"(?<!\w)" + re.escape(label) + r"(?!\w)", question, re.I
        ):
            canonical = alternate.get(label,
                                      v2.TEAM_ALIASES.get(label.lower(), label))
            mentions.append((match.start(), -len(label), canonical))
    seen = []
    for _, _, team in sorted(mentions):
        if team not in seen:
            seen.append(team)
    return tuple(seen)


def parse_scope(question: str, previous: Scope | None = None) -> Scope:
    # Explicit subject switches discard the old named teams before parsing.
    current = question
    if re.search(r"\b(?:forget|ignore|instead of|switch from)\b", question, re.I):
        switches = list(re.finditer(r"\b(?:compare|show|analyse|analyze)\b", question, re.I))
        if switches and _teams(question[switches[-1].start():]):
            current = question[switches[-1].start():]
    teams = _teams(current)
    # A recent-five versus preceding-five comparison needs TEN match records.
    # Do not treat "last five" alone as a trend request.
    trend = bool(re.search(
        r"\b(?:five before|previous five|preceding five|prior five|"
        r"five matches before|five games before|"
        r"compared (?:with|to) (?:the )?(?:five|previous five)|"
        r"(?:last|recent|latest) five .{0,70}(?:previous|preceding|before|prior) five)\b",
        current, re.I))
    match = _WINDOW.search(current)
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
    if trend:
        games = 10
    if previous and _REFERENTIAL.search(question):
        if not teams:
            teams = previous.teams
        if games is None and teams == previous.teams:
            games = previous.games
    attacking = bool(re.search(
        r"\b(?:attack(?:ing)?|goals? scored|goals? for|going forward|"
        r"scor(?:e|ed|ing)|offen[sc]e)\b", current, re.I))
    defensive = bool(re.search(
        r"\b(?:defen[sc](?:e|ive)|defend(?:ing)?|conced(?:e|ed|ing)|"
        r"goals? against)\b", current, re.I))
    focus = ("attack" if attacking and not defensive else
             "defence" if defensive else
             previous.focus if previous and not _teams(current) and _REFERENTIAL.search(question)
             else "defence")
    return Scope(teams, games, bool(_H2H.search(current)),
                 bool(_OPPOSITION.search(current)),
                 bool(re.search(r"\b(?:home|away)\b", current, re.I)), focus, trend)


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


def _execute(call: dict, tools: dict, require_venues: bool = False, focus: str = 'defence', require_trend: bool = False) -> dict:
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
        if name == "form_summary" and require_trend:
            row = result["data"][0]
            measure = "Scored" if focus == "attack" else "Conceded"
            keys = (f"{measure}Recent5", f"{measure}Previous5",
                    "PointsRecent5", "PointsPrevious5")
            if int(row["Played"]) != 10 or not all(k in row for k in keys):
                raise ValueError("Missing complete two-window trend evidence")
            recent, previous, points_a, points_b = (int(row[k]) for k in keys)
            overall = "GoalsFor" if focus == "attack" else "GoalsAgainst"
            if (min(recent, previous, points_a, points_b) < 0
                    or recent + previous != int(row[overall])
                    or points_a + points_b != int(row["Points"])):
                raise ValueError("Inconsistent two-window trend evidence")
        if name == "form_summary" and require_venues:
            row = result["data"][0]
            split = row.get("VenueSplit")
            played = int(row["Played"])
            if (row.get("VenueKnown") != played
                    or not isinstance(split, dict)
                    or any(not isinstance(split.get(venue), dict)
                           for venue in ("Home", "Away"))):
                raise ValueError("Incomplete home/away evidence")
            try:
                home, away = (split[venue] for venue in ("Home", "Away"))
                home_n, away_n = int(home["Played"]), int(away["Played"])
                metric = "GoalsFor" if focus == "attack" else "GoalsAgainst"
                home_ga, away_ga = int(home[metric]), int(away[metric])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Missing venue measurements") from exc
            if (home_n <= 0 or away_n <= 0 or home_n + away_n != played
                    or home_ga < 0 or away_ga < 0
                    or home_ga + away_ga != int(row[metric])):
                raise ValueError("Insufficient or inconsistent venue evidence")
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


def _defensive_conclusions(scope: Scope, evidence: list[dict]) -> list[str]:
    """Derived comparisons only; never ask an LLM to infer missing facts."""
    if len(scope.teams) != 2:
        return []
    forms = {}
    opponent = None
    for item in evidence:
        if "result" not in item:
            continue
        if item["tool"] == "form_summary":
            forms[item["arguments"]["team"]] = item["result"]["data"][0]
        elif item["tool"] == "compare_opponent_strength":
            opponent = item["result"].get("data", {})
    a, b = scope.teams
    lines = []
    if scope.focus == "attack":
        if a not in forms or b not in forms:
            return lines
        ra, rb = forms[a], forms[b]
        if int(ra["Played"]) != int(rb["Played"]):
            return ["**Attacking verdict:** Unequal match windows; comparison unavailable."]
        gf_a, gf_b = int(ra["GoalsFor"]), int(rb["GoalsFor"])
        pt_a, pt_b = int(ra["Points"]), int(rb["Points"])
        if gf_a == gf_b:
            lines.append("**Attacking verdict:** Both teams scored the same number of goals.")
        else:
            winner = a if gf_a > gf_b else b
            lines.append(f"**Attacking verdict:** {winner} scored more goals "
                         f"({gf_a} versus {gf_b}).")
        if pt_a == pt_b:
            lines.append("**Points verdict:** Both teams earned the same number of points.")
        else:
            winner = a if pt_a > pt_b else b
            lines.append(f"**Points verdict:** {winner} earned more points "
                         f"({pt_a} versus {pt_b}).")
        if scope.venues:
            for venue in ("Home", "Away"):
                try:
                    va, vb = ra["VenueSplit"][venue], rb["VenueSplit"][venue]
                    na, nb = int(va["Played"]), int(vb["Played"])
                    fa, fb = int(va["GoalsFor"]), int(vb["GoalsFor"])
                    if na <= 0 or nb <= 0:
                        raise ValueError("No rated venue matches")
                    winner = (a if fa * nb > fb * na else
                              b if fa * nb < fb * na else None)
                    desc = (f"{winner} scored at a higher rate" if winner
                            else "Both teams scored at the same rate")
                    lines.append(f"**{venue} attacking verdict:** {desc} "
                                 f"({fa/na:.2f} versus {fb/nb:.2f} per match).")
                except (KeyError, TypeError, ValueError):
                    lines.append(f"**{venue} attacking verdict:** Goals-scored "
                                 "venue evidence unavailable; no conclusion.")
        return lines
    if a in forms and b in forms:
        ra, rb = forms[a], forms[b]
        if int(ra["Played"]) == int(rb["Played"]):
            ga_a, ga_b = int(ra["GoalsAgainst"]), int(rb["GoalsAgainst"])
            if ga_a == ga_b:
                lines.append("**Overall verdict:** The teams conceded equally across the measured matches.")
            else:
                better = a if ga_a < ga_b else b
                lines.append(f"**Overall verdict:** {better} conceded fewer goals "
                             f"({ga_a} versus {ga_b}); this is a descriptive result.")
        if scope.venues:
            for venue in ("Home", "Away"):
                try:
                    va = ra["VenueSplit"][venue]
                    vb = rb["VenueSplit"][venue]
                    na, nb = int(va["Played"]), int(vb["Played"])
                    ga_a, ga_b = int(va["GoalsAgainst"]), int(vb["GoalsAgainst"])
                    if not (na > 0 and nb > 0 and
                            ra.get("VenueKnown") == ra["Played"] and
                            rb.get("VenueKnown") == rb["Played"]):
                        raise ValueError("Incomplete venue breakdown")
                except (KeyError, TypeError, ValueError):
                    lines.append(f"**{venue} verdict:** Insufficient verified venue evidence.")
                    continue
                # Compare exact rational rates, not rounded display figures.
                left, right = ga_a * nb, ga_b * na
                if left == right:
                    lines.append(f"**{venue} verdict:** Both teams conceded at the same "
                                 f"rate ({ga_a/na:.2f} per match).")
                else:
                    better = a if left < right else b
                    lines.append(f"**{venue} verdict:** {better} conceded at a lower "
                                 f"rate ({ga_a/na:.2f} versus {ga_b/nb:.2f} "
                                 "goals per match).")
    if scope.venues and a in forms and b in forms:
        ra, rb = forms[a], forms[b]
        if int(ra["Played"]) == int(rb["Played"]):
            total_a, total_b = int(ra["GoalsAgainst"]), int(rb["GoalsAgainst"])
            if total_a != total_b:
                overall = a if total_a < total_b else b
                for venue in ("Home", "Away"):
                    try:
                        va, vb = ra["VenueSplit"][venue], rb["VenueSplit"][venue]
                        na, nb = int(va["Played"]), int(vb["Played"])
                        ga, gb = int(va["GoalsAgainst"]), int(vb["GoalsAgainst"])
                        if na <= 0 or nb <= 0:
                            continue
                        lhs, rhs = ga * nb, gb * na
                        if lhs == rhs:
                            lines.append(f"**{venue} versus overall:** The venue rates "
                                         "are tied, despite the overall difference.")
                        else:
                            venue_leader = a if lhs < rhs else b
                            comparison = ("The overall advantage also holds"
                                          if overall == venue_leader
                                          else "The overall advantage reverses")
                            lines.append(f"**{venue} versus overall:** {comparison} "
                                         f"for {venue.lower()} concession rate.")
                    except (KeyError, TypeError, ValueError):
                        continue
    if scope.opposition:
        if not isinstance(opponent, dict):
            lines.append("**Opposition verdict:** Verified opponent-strength evidence unavailable.")
        else:
            aa, bb = opponent.get(a, {}), opponent.get(b, {})
            pa, pb = aa.get("mean_opponent_prior_ppg"), bb.get("mean_opponent_prior_ppg")
            if pa is None or pb is None:
                lines.append("**Opposition verdict:** Insufficient rated opponent evidence.")
            else:
                pa, pb = float(pa), float(pb)
                if pa == pb:
                    lines.append("**Opposition verdict:** The measured opponents had "
                                 "equal average prior PPG.")
                else:
                    tougher = a if pa > pb else b
                    lines.append(f"**Opposition verdict:** {tougher} faced opponents "
                                 f"with higher average prior PPG "
                                 f"({pa:.2f} versus {pb:.2f}; difference "
                                 f"{abs(pa-pb):.2f}). This proxy does not establish "
                                 "that the defensive difference was caused by opposition.")
    return lines


def _trend_facts(scope: Scope, evidence: list[dict]) -> list[str]:
    """Compare two adjacent five-match windows from 10 completed results."""
    if not scope.trend:
        return []
    rows = {e["arguments"]["team"]: e["result"]["data"][0]
            for e in evidence if e.get("tool") == "form_summary" and "result" in e}
    recent_key, previous_key = (("ScoredRecent5", "ScoredPrevious5")
                                if scope.focus == "attack" else
                                ("ConcededRecent5", "ConcededPrevious5"))
    lines = []
    for team in scope.teams:
        row = rows.get(team)
        if not row or int(row.get("Played", 0)) != 10:
            continue
        if not all(k in row for k in (recent_key, previous_key,
                                      "PointsRecent5", "PointsPrevious5")):
            continue
        recent, previous = int(row[recent_key]), int(row[previous_key])
        recent_pts = int(row["PointsRecent5"])
        previous_pts = int(row["PointsPrevious5"])
        if min(recent, previous, recent_pts, previous_pts) < 0:
            continue
        # Reconcile window totals against the overall verified form row.
        overall = "GoalsFor" if scope.focus == "attack" else "GoalsAgainst"
        if recent + previous != int(row[overall]) or recent_pts + previous_pts != int(row["Points"]):
            continue
        direction = ("unchanged" if recent == previous else
                     "improved" if (recent > previous if scope.focus == "attack"
                                    else recent < previous) else "worsened")
        metric = "goals scored" if scope.focus == "attack" else "goals conceded"
        lines.append(f"**{team} five-match trend:** {direction.capitalize()} on {metric}: "
                     f"{recent} in the most recent five versus {previous} in the "
                     f"preceding five. Points: {recent_pts} versus {previous_pts}. "
                     "Descriptive only; not proof of a persistent trend.")
    return lines


def _answer_lead(scope: Scope, evidence: list[dict], gaps: list[dict]) -> str | None:
    """Concise, deterministic lead. No unsupported causal or tactical inference."""
    if gaps:
        return None
    if scope.trend:
        trend_lines = _trend_facts(scope, evidence)
        return " ".join(line.replace("**", "") for line in trend_lines) if (
            len(trend_lines) == len(scope.teams)) else None
    if len(scope.teams) != 2:
        return None
    forms = {}
    opponents = None
    for item in evidence:
        if "result" not in item:
            continue
        if item["tool"] == "form_summary":
            forms[item["arguments"]["team"]] = item["result"]["data"][0]
        elif item["tool"] == "compare_opponent_strength":
            opponents = item["result"].get("data", {})
    a, b = scope.teams
    if a not in forms or b not in forms:
        return None
    ra, rb = forms[a], forms[b]
    if int(ra["Played"]) != int(rb["Played"]):
        return None
    if scope.focus == "attack":
        fa, fb = int(ra["GoalsFor"]), int(rb["GoalsFor"])
        pa, pb = int(ra["Points"]), int(rb["Points"])
        lead = (f"{a} and {b} scored {fa} goals each" if fa == fb else
                f"{a if fa > fb else b} scored more goals "
                f"({fa} versus {fb})")
        text = (f"{lead} across their last {scope.games} matches. "
                f"{a} earned {pa} points and {b} earned {pb}.")
        if scope.venues:
            text += (" The venue-specific attacking comparison is shown "
                     "below where goals-scored splits are verified.")
        return text
    ga, gb = int(ra["GoalsAgainst"]), int(rb["GoalsAgainst"])
    if ga == gb:
        text = (f"{a} and {b} conceded the same number of goals "
                f"({ga} each) in their last {scope.games} matches.")
        overall = None
    else:
        overall = a if ga < gb else b
        text = (f"{overall} had the better goals-conceded record in their "
                f"last {scope.games} matches: {a} conceded {ga}, "
                f"compared with {gb} for {b}.")
    if scope.venues:
        for venue in ("Home", "Away"):
            try:
                va, vb = ra["VenueSplit"][venue], rb["VenueSplit"][venue]
                na, nb = int(va["Played"]), int(vb["Played"])
                ga_v, gb_v = int(va["GoalsAgainst"]), int(vb["GoalsAgainst"])
                if min(na, nb) <= 0:
                    continue
                left, right = ga_v * nb, gb_v * na
                verdict = ("the teams were level on concession rate"
                           if left == right else
                           f"{a if left < right else b} had the lower concession rate")
                text += (f" {venue} matches: {verdict} "
                         f"({ga_v/na:.2f} versus {gb_v/nb:.2f} per match).")
            except (KeyError, TypeError, ValueError):
                pass
    if scope.opposition and isinstance(opponents, dict):
        oa, ob = opponents.get(a, {}), opponents.get(b, {})
        pa, pb = oa.get("mean_opponent_prior_ppg"), ob.get("mean_opponent_prior_ppg")
        if pa is not None and pb is not None:
            difference = abs(float(pa) - float(pb))
            rated_a, rated_b = oa.get("rated", 0), ob.get("rated", 0)
            played_a, played_b = oa.get("played", 0), ob.get("played", 0)
            text += (f" Opponents' average prior PPG was {float(pa):.2f} "
                     f"versus {float(pb):.2f} (difference {difference:.2f}; "
                     f"rated coverage {rated_a}/{played_a} versus "
                     f"{rated_b}/{played_b}).")
            if difference < 0.10:
                text += (" This is a small numerical difference on this proxy, "
                         "not proof of materially tougher opposition.")
            text += " It does not explain the defensive gap causally."
    return text


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
        if len(forms) == 2 and scope.focus == "attack":
            (ta, a, _), (tb, b, _) = forms
            if a["Played"] == b["Played"]:
                gf_a, gf_b = int(a["GoalsFor"]), int(b["GoalsFor"])
                pt_a, pt_b = int(a["Points"]), int(b["Points"])
                if gf_a != gf_b:
                    winner = ta if gf_a > gf_b else tb
                    lines.append(f"\n**{winner} scored {abs(gf_a-gf_b)} more goals** "
                                 "in the measured window.")
                else:
                    lines.append("\nBoth teams scored the same number of goals.")
                lines.append(f"\nPoints earned: {ta} {pt_a}, {tb} {pt_b}.")
        if len(forms) == 2 and scope.focus != "attack":
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
                        metric = "GoalsFor" if scope.focus == "attack" else "GoalsAgainst"
                        if metric not in venue_row:
                            lines.append(f"\n{team}, {venue.lower()}: verified {metric} unavailable.")
                            continue
                        goals = int(venue_row[metric])
                        direction = "scored" if scope.focus == "attack" else "conceded"
                        lines.append(f"\n{team}, {venue.lower()}: {goals} goals "
                                     f"{direction} in {n} matches ({goals/n:.2f} per match).")
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
    if scope.trend:
        trend_lines = _trend_facts(scope, evidence)
        if trend_lines:
            lines += ["", "### Two-window form investigation", *trend_lines]
        if len(trend_lines) != len(scope.teams):
            lines.append("\\n**Trend evidence incomplete:** cannot verify all "
                         "recent-five and preceding-five comparisons.")
    conclusions = [] if scope.trend else _defensive_conclusions(scope, evidence)
    if conclusions:
        lines += ["", "### What the evidence shows", *conclusions]
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
    details = "\n".join(lines)
    lead = _answer_lead(scope, evidence, gaps)
    return ("### Answer\n" + lead + "\n\n" + details) if lead else details


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
        provider_blocked = False
        for round_idx in range(MAX_ROUNDS):
            missing = [c for c in needed if call_key(c) not in tried]
            if not missing or len(tried) >= MAX_TOOLS:
                break
            rounds += 1
            # Do not leak the expected answer/tool list to the first
            # planner pass: otherwise the apparent AI planning score is
            # largely a test of following a supplied solution.
            review = (
                "First pass: independently select the read-only tools "
                "necessary to answer every component of the question."
                if round_idx == 0 else
                f"Second and FINAL pass. Already verified tools: "
                f"{[e['tool'] for e in evidence if 'result' in e]}. "
                f"Missing evidence requirements (repair hints): {missing!r}. "
                "Select ONLY missing calls; do not repeat completed tools."
            )
            instruction = (
                "Select read-only Premier League evidence tools. Return only "
                "a list of {name,arguments} calls (max 3). Do not answer, invent "
                "statistics, request destructive tools or change source data. "
                f"User question: {question!r}. Teams: {scope.teams}. "
                f"Last-N window: {scope.games}. "
                + review + " Use exact team names and match window."
            )
            try:
                proposals = validated_proposals(self.planner(instruction), missing)
            except GroqRateLimited:
                trace.append("provider_blocked:GroqRateLimited")
                provider_blocked = True
                break
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
                item = _execute(call, self.tools, require_venues=scope.venues,
                                focus=scope.focus, require_trend=scope.trend)
                evidence.append(item)
                trace.append(f"executed_{item['tool']}" if "result" in item
                             else f"unavailable_{item['tool']}")
        complete_keys = {(e["tool"], tuple(sorted(e["arguments"].items())))
                         for e in evidence if "result" in e}
        gaps = [c for c in needed if call_key(c) not in complete_keys]
        trace.append("complete" if not gaps else "partial_evidence")
        answer = render_evidence(scope, evidence, gaps)
        if provider_blocked:
            answer += ("\n\n**Provider rate limit:** Groq blocked further planning. "
                       "This is an availability failure, not evidence of incorrect "
                       "model reasoning; missing evidence remains unresolved.")
        return {"answer": answer,
                "evidence": evidence, "gaps": gaps, "trace": trace,
                "complete": not gaps, "rounds": rounds,
                "provider_blocked": provider_blocked}
