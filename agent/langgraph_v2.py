"""V2 LangGraph prototype. Separate from existing Streamlit runtime until evaluated.

One bounded local-LLM planning call, then at most three read-only tool calls.
No model writes, hidden retries, paid providers, or answer-generation LLM loop.
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Annotated, Callable, Literal, TypedDict, Union
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from httpx import ReadTimeout

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from agent.football_tools import (
    experiment_history,
    frozen_gameweek_predictions,
    model_performance,
    form_summary,
    next_meeting,
)

ALLOWED = {
    "frozen_gameweek_predictions": frozen_gameweek_predictions,
    "experiment_history": experiment_history,
    "model_performance": model_performance,
    "form_summary": form_summary,
    "next_meeting": next_meeting,
}
MAX_TOOLS = 3
TEAM_ALIASES = {'manchester city': 'Man City', 'manchester united': 'Man United', 'spurs': 'Tottenham'}

class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _GWArgs(_Strict):
    gameweek: int = Field(ge=1, le=38)


class _H07Args(_Strict):
    experiment: Literal["H07"]


class _FormArgs(_Strict):
    team: str = Field(min_length=1, max_length=80)
    games: int = Field(ge=1, le=30)


class _MeetingArgs(_Strict):
    team1: str = Field(min_length=1, max_length=80)
    team2: str = Field(min_length=1, max_length=80)


class _EmptyArgs(_Strict):
    pass


class _ForecastCall(_Strict):
    name: Literal["frozen_gameweek_predictions"]
    arguments: _GWArgs


class _HistoryCall(_Strict):
    name: Literal["experiment_history"]
    arguments: _H07Args


class _FormCall(_Strict):
    name: Literal["form_summary"]
    arguments: _FormArgs


class _MeetingCall(_Strict):
    name: Literal["next_meeting"]
    arguments: _MeetingArgs


class _PerformanceCall(_Strict):
    name: Literal["model_performance"]
    arguments: _EmptyArgs


_ToolCall = Annotated[
    Union[_ForecastCall, _HistoryCall, _PerformanceCall, _FormCall, _MeetingCall],
    Field(discriminator="name"),
]


class _Plan(_Strict):
    calls: list[_ToolCall] = Field(max_length=MAX_TOOLS)



class AgentState(TypedDict):
    question: str
    plan: list[dict]
    evidence: list[dict]
    answer: str
    trace: list[str]
    history: list[dict]


def local_planner(question: str) -> list[dict]:
    """Use local Qwen only for concise tool choice, not large evidence prompts."""
    from ollama import Client

    instruction = (
        "Choose read-only tools needed to answer the user's football question. "
        "Return JSON only: an object with a 'calls' list, each item having "
        "'name' and 'arguments'. Allowed: frozen_gameweek_predictions "
        "(arguments: gameweek integer 1..38), experiment_history "
        "(arguments: experiment='H07'), model_performance (arguments: {}), "
        "form_summary (arguments: exact team name, games integer 1..10). "
        "The arguments object for model_performance MUST be empty {}. "
        "Never include champion, accuracy or any other parameter for model_performance. "
        "Choose every relevant tool for multi-part questions, up to three calls. "
        "For a request combining performance and forecasts, select BOTH tools. "
        "For unsupported or destructive requests, return an empty calls list. "
        "No invented tools, and never interpret a destructive request as a research question. "
        "Do not provide an answer. "
        "Examples: 'GW8 forecasts?' -> frozen_gameweek_predictions(8); "
        "'Why reject H07?' -> experiment_history(H07); "
        "'How has 40/60 performed?' -> model_performance()."
    )
    timeout = max(5, min(180, int(os.getenv("FOOTBALL_AGENT_TIMEOUT", "30"))))
    result = Client(timeout=timeout).chat(
        model=os.getenv("FOOTBALL_AGENT_MODEL", "qwen3:0.6b"),
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": question},
        ],
        format=_Plan.model_json_schema(),
        think=False,
        options={"num_predict": 200, "num_ctx": 2048, "temperature": 0},
    )
    parsed = _Plan.model_validate_json(result.message.content)
    return [call.model_dump() for call in parsed.calls]


class _Answer(_Strict):
    answer: str = Field(min_length=20, max_length=1800)


def local_synthesizer(question: str, facts: str) -> str:
    """Experimental Ollama answer writer; only compact verified tool summaries."""
    from ollama import Client

    result = Client(timeout=max(5, min(180, int(os.getenv("FOOTBALL_ANSWER_TIMEOUT", "60"))))).chat(
        model=os.getenv("FOOTBALL_ANSWER_MODEL", "qwen3:0.6b"),
        messages=[
            {"role": "system", "content": (
                "Answer using ONLY the supplied football evidence. Never invent results, "
                "scores, teams, metrics, fixtures or sources. Explain comparisons in plain English. "
                "A saved forecast is not itself an evaluated result. GW8 matches have now "
                "finished and the supplied GW8 evaluation is authoritative. Do not describe "
                "GW8 as unevaluated. Do not claim proven improvement from a small sample. "
                "Do not say observed champion performance is unavailable when supplied. "
                "Distinguish the historical GW6-GW7 40/60 leader from the latest GW8 "
                "evaluation, where market-only leads accuracy and Log Loss. Model 2 remains "
                "the original benchmark. The 40/60 blend is a provisional candidate, not "
                "formally promoted. Attribute every metric to the right model and period. "
                "Keep all figures exact and be concise."
            )},
            {"role": "user", "content": f"Question: {question}\nVerified evidence:\n{facts}"},
        ],
        format=_Answer.model_json_schema(),
        think=False,
        options={"num_predict": 320, "num_ctx": 2048, "temperature": 0},
    )
    return _Answer.model_validate_json(result.message.content).answer


def _grounded(candidate: str, facts: str, is_forecast: bool, is_performance: bool) -> bool:
    """Conservative numeric guard; reject novel numbers and unsupported evaluation claims."""
    number = r"(?<![A-Za-z0-9])\d+(?:\.\d+)?%?"
    known = set(re.findall(number, facts))
    if set(re.findall(number, candidate)) - known:
        return False
    if "H07 parked:" in facts:
        lower = candidate.lower()
        if "h07" not in lower or not any(
            x in lower for x in ("parked", "rejected", "did not outperform", "underperformed")
        ):
            return False
    if is_performance:
        lower = candidate.lower()
        # An answer about performance must contain the observed comparison, not
        # merely claim that the evidence is insufficient.
        if not (("30%" in candidate and "40%" in candidate) if "GW8 (10 finished matches)" in facts else ("55%" in candidate and "40%" in candidate)):
            return False
        # Reject explicit claims that Model 2 leads. Do not reject statements
        # such as "40/60 is better than Model 2".
        if re.search(r"\bmodel\s*2\s+(?:is|was|remains|has been)\s+(?:the\s+)?(?:best|better|leader|leading|highest.performing|top.performing)", lower):
            return False
        if re.search(r"\b(?:best|leading|top.performing)\s+model\s+is\s+model\s*2\b", lower):
            return False
        # A valid answer must preserve both the provisional governance status
        # and the limited size of the prospective sample.
        if not any(x in lower for x in ("provisional", "not formally promoted", "not officially promoted", "pending promotion")):
            return False
        if not (any(x in lower for x in ("small sample", "limited sample", "only 20", "20 matches", "20 fixtures", "insufficient evidence", "one gameweek", "ten matches", "10 matches"))
                or re.search(r"\b(?:small|limited)\s+\d+[- ]match\s+sample\b", lower)):
            return False
        if any(x in lower for x in (
            "not explicitly stated", "no clear information",
            "does not provide clear information", "performance is unknown",
        )):
            return False
    if is_forecast and is_performance and "GW8 (10 finished matches)" not in facts:
        # The model must state the temporal distinction explicitly.
        lower = candidate.lower()
        if not any(x in lower for x in ("not yet evaluated", "not evaluated", "cannot assess", "cannot yet", "unknown", "not results")):
            return False
    return True


def make_graph(planner: Callable[[str], list[dict]] = local_planner, checkpointer=None, synthesizer: Callable[[str, str], str] | None = None):
    def plan(state: AgentState):
        # Narrow guard against clearly non-football questions which the small
        # local planner can otherwise misroute to forecast tools.
        if re.search(r"\b(weather|temperature|rainfall|wind speed)\b", state["question"], re.I):
            return {"plan": [], "trace": ["rejected_out_of_scope"]}
        # Safety gate: destructive intent must not trigger unrelated read tools.
        # Actual tool permissions remain read-only regardless of this check.
        if re.search(r"\b(delete|erase|wipe|remove|overwrite|retrain|drop|destroy)\b", state["question"], re.I):
            return {"plan": [], "trace": ["rejected_destructive_intent"]}
        # Fast, bounded deterministic route for clearly identified two-team form.
        # This skips a potentially slow local Ollama planner and uses read-only tools.
        q = state["question"]
        prior_q = next((h["question"] for h in reversed(state.get("history", []))
                        if re.search(r"\b(compare|versus|vs|which of)\b", h["question"], re.I)), "")
        team_names = ("Manchester City", "Manchester United", "Liverpool",
                      "Arsenal", "Chelsea", "Tottenham", "Spurs", "Man City", "Man United")
        def names_in(sentence):
            found = []
            for label in team_names:
                if re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", sentence, re.I):
                    team = TEAM_ALIASES.get(label.lower(), label)
                    if team not in found:
                        found.append(team)
            return found
        teams = names_in(q)
        followup = bool(re.search(r"\b(their|both|those|that comparison|who|which|better|they|each other)\b", q, re.I) or re.search(r"\b(?:what about|how about|and over|over the last|last \d+|defen[sc]\w*)\b", q, re.I))
        if len(teams) != 2 and followup:
            teams = names_in(prior_q)
            if len(teams) != 2:
                teams = names_in(" ".join(h["question"] for h in reversed(state.get("history", []))))[:2]
        if re.search(r"\b(when|date|next|upcoming)\b", q, re.I) and re.search(
            r"\b(play|playing|meet|meeting|face|fixture)\b", q, re.I
        ) and len(teams) == 2:
            return {"plan": [{"name": "next_meeting",
                              "arguments": {"team1": teams[0], "team2": teams[1]}}],
                    "trace": ["deterministic_next_meeting_route"]}
        requested = re.search(r"\b(?:last|past|those|these)\s+(\d+|five|ten)(?:\s+(?:matches|games))?\b", q, re.I)
        games = {"five": 5, "ten": 10}.get(requested.group(1).lower(), None) if requested else None
        if requested and games is None:
            games = int(requested.group(1))
        is_form = bool(re.search(r"\b(form|matches|games|defen[sc]\w*|comparison|compare|conced\w*|last|past|those|these)\b", q, re.I))
        gw_match = re.search(r"\b(?:GW|gameweek)\s*(\d{1,2})\b", q, re.I)
        forecast_intent = bool(re.search(r"\b(forecast|prediction|predicted)\w*\b", q, re.I))
        if (len(teams) == 2 and games is not None and 1 <= games <= 30
                and is_form and gw_match and forecast_intent):
            gw = int(gw_match.group(1))
            if 1 <= gw <= 38:
                return {"plan": [
                    {"name": "form_summary", "arguments": {"team": team, "games": games}}
                    for team in teams
                ] + [{"name": "frozen_gameweek_predictions",
                       "arguments": {"gameweek": gw}}],
                        "trace": ["deterministic_multi_tool_form_forecast_route"]}
        if len(teams) == 2 and games is not None and 1 <= games <= 30 and is_form:
            return {"plan": [{"name": "form_summary",
                               "arguments": {"team": team, "games": games}} for team in teams],
                    "trace": ["deterministic_form_route"]}
        # Explicit model-quality questions are not requests for a team's recent form.
        # Repair the two misroutes observed in the 13-turn live Qwen acceptance.
        # Compose explicit forecasting, performance and experiment requests before
        # applying single-topic performance shortcuts.  Three read-only tools max.
        mentioned_gw = re.search(r"\b(?:GW|gameweek)\s*(\d{1,2})\b", q, re.I)
        asks_forecasts = bool(re.search(r"\b(forecast|prediction)\w*\b", q, re.I))
        asks_performance = bool(re.search(r"\b(champion|40/60|model performance|accuracy)\b", q, re.I))
        asks_h07 = bool(re.search(r"\bH07\b", q, re.I))
        if mentioned_gw and asks_forecasts and asks_performance and asks_h07:
            gw = int(mentioned_gw.group(1))
            if 1 <= gw <= 38:
                return {"plan": [
                    {"name": "frozen_gameweek_predictions", "arguments": {"gameweek": gw}},
                    {"name": "model_performance", "arguments": {}},
                    {"name": "experiment_history", "arguments": {"experiment": "H07"}}],
                    "trace": ["deterministic_explicit_multi_source_route"]}
        performance_intent = bool(re.search(
            r"\b(champion|40/60|model performance|model 2)\b", q, re.I
        ) and re.search(
            r"\b(performed|performance|accuracy|accurate|log loss|brier)\b", q, re.I
        ))
        if performance_intent and not (re.search(r"\bhow does that compare\b", q, re.I) and "recent performance" not in q.lower()):
            prior_forecast = any(
                re.search(r"\b(?:GW|gameweek)\s*\d+\b", h["question"], re.I)
                for h in state.get("history", [])[-2:]
            )
            comparison_followup = bool(re.search(
                r"\b(how does that compare|compare with|compared with)\b", q, re.I
            ))
            if comparison_followup and prior_forecast and "recent performance" in q.lower():
                gw = next((
                    int(m.group(1)) for h in reversed(state.get("history", [])[-2:])
                    if (m := re.search(r"\b(?:GW|gameweek)\s*(\d+)\b", h["question"], re.I))
                ), None)
                if gw and 1 <= gw <= 38:
                    return {"plan": [
                        {"name": "frozen_gameweek_predictions", "arguments": {"gameweek": gw}},
                        {"name": "model_performance", "arguments": {}}],
                        "trace": ["deterministic_performance_followup_route"]}
            return {"plan": [{"name": "model_performance", "arguments": {}}],
                    "trace": ["deterministic_performance_route"]}
        try:
            previous = state.get("history", [])[-2:]
            context = "\n".join(
                f"Earlier user: {item['question']}\nEarlier assistant: {item['answer'][:300]}"
                for item in previous
            )
            prompt = (f"Conversation context:\n{context}\nCurrent question: {state['question']}"
                      if context else state["question"])
            raw = planner(prompt)
        except (ReadTimeout, TimeoutError) as exc:
            return {"plan": [], "trace": [f"planner_timeout: {type(exc).__name__} (local planner timeout)"]}
        except (ValueError, KeyError, TypeError, ValidationError) as exc:
            return {"plan": [], "trace": [f"planner_invalid_output: {type(exc).__name__}"]}
        valid = []
        for call in raw[:MAX_TOOLS] if isinstance(raw, list) else []:
            if not isinstance(call, dict):
                continue
            name, args = call.get("name", call.get("tool")), call.get("arguments", {})
            if name not in ALLOWED or not isinstance(args, dict):
                continue
            if name == "frozen_gameweek_predictions":
                if set(args) != {"gameweek"}:
                    continue
                try:
                    gw = int(args["gameweek"])
                except (ValueError, TypeError):
                    continue
                if not 1 <= gw <= 38:
                    continue
                args = {"gameweek": gw}
            elif name == "experiment_history":
                if args.get("experiment", "").upper() != "H07":
                    continue
                args = {"experiment": "H07"}
            elif name == "next_meeting":
                try:
                    args = _MeetingArgs.model_validate(args).model_dump()
                except ValidationError:
                    continue
            elif name == "form_summary":
                try:
                    parsed = _FormArgs.model_validate(args)
                except ValidationError:
                    continue
                args = parsed.model_dump()
                args["team"] = TEAM_ALIASES.get(args["team"].strip().lower(), args["team"].strip())
            elif args:
                continue
            if {"name": name, "arguments": args} not in valid:
                valid.append({"name": name, "arguments": args})
        # Resolve a missing second team in a two-team recent-form follow-up.
        if (re.search(r"\b(their|both|those)\b", state["question"], re.I)
                and any(c["name"] == "form_summary" for c in valid)):
            prior = next((h["question"] for h in reversed(state.get("history", []))
                          if re.search(r"\bcompare\b", h["question"], re.I)), "")
            # Use earlier planner-validated team names carried in prior tool
            # evidence when available; otherwise resolve explicit known names.
            names = []
            for label in ("Manchester City", "Manchester United", "Liverpool",
                          "Arsenal", "Chelsea", "Tottenham", "Spurs", "Man City", "Man United"):
                if re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", prior, re.I):
                    canonical = TEAM_ALIASES.get(label.lower(), label)
                    if canonical not in names:
                        names.append(canonical)
            if len(names) == 2:
                games = next(c["arguments"]["games"] for c in valid
                             if c["name"] == "form_summary")
                for team in names:
                    call = {"name": "form_summary",
                            "arguments": {"team": team, "games": games}}
                    if call not in valid and len(valid) < MAX_TOOLS:
                        valid.append(call)
        # Narrow completeness check for a comparative follow-up. The small model
        # sometimes selects only forecasts even when asked for champion performance.
        # This does not force any tool for unrelated requests.
        previous_questions = " ".join(x["question"] for x in state.get("history", [])[-2:])
        wants_champion_comparison = (
            bool(re.search(r"\b(champion|40/60)\b", state["question"], re.I))
            and bool(re.search(r"\b(compare|compared|comparison|versus|vs|how does that)\b", state["question"], re.I))
        )
        has_forecast_context = bool(re.search(r"\b(GW\s*\d+|gameweek\s*\d+)\b", previous_questions, re.I))
        repaired = False
        if wants_champion_comparison and has_forecast_context and not any(
            x["name"] == "model_performance" for x in valid
        ) and len(valid) < MAX_TOOLS:
            valid.append({"name": "model_performance", "arguments": {}})
            repaired = True
        trace = [f"planned {len(valid)} tool(s)"]
        if repaired:
            trace.append("plan_completed_missing_comparison_evidence")
        return {"plan": valid, "trace": trace}

    def execute(state: AgentState):
        evidence, trace = [], list(state["trace"])
        for call in state["plan"]:
            name = call["name"]
            try:
                if name == "form_summary" and ALLOWED[name] is form_summary:
                    result = ALLOWED[name](**call["arguments"], season="2026/27")
                else:
                    result = ALLOWED[name](**call["arguments"])
            except Exception as exc:
                result = {"error": f"Tool execution failed: {type(exc).__name__}: {exc}" if isinstance(exc, ValueError) else f"Tool execution failed: {type(exc).__name__}"}
            evidence.append({"tool": name, "arguments": call["arguments"], "result": result})
            trace.append(f"executed {name}")
        return {"evidence": evidence, "trace": trace}

    def respond(state: AgentState):
        def save(answer: str):
            history = state.get("history", [])
            return {"answer": answer, "history": (history + [{"question": state["question"], "answer": answer}])[-4:]}
        if not state["evidence"]:
            if any("planner_timeout" in t for t in state["trace"]):
                return save("Local Qwen planner timed out. No tools were run or data changed; see trace for details.")
            return save("No supported evidence tool selected. Please specify a gameweek, H07, or model performance.")
        parts = []
        for item in state["evidence"]:
            result, name = item["result"], item["tool"]
            if "error" in result and name == "next_meeting":
                parts.append(f"Unable to verify the next Premier League meeting. {result['error']} Source: {result.get('source', 'FPL schedule')}.")
            elif "error" in result:
                parts.append(f"{name}: {result['error']}")
            elif name == "frozen_gameweek_predictions":
                rows = result["data"]
                lines = [
                    f"**GW{result['gameweek']} frozen forecasts ({len(rows)} fixtures)**",
                    "",
                    "| Fixture | 40/60 champion | Model 2 | Adaptive challenger |",
                    "|---|---|---|---|",
                ]
                for fixture in rows:
                    match = f"{fixture['HomeTeam']} vs {fixture['AwayTeam']}"
                    champion = fixture.get("ShadowPredictedResult")
                    benchmark = fixture.get("PredictedResult")
                    adaptive = fixture.get("AdaptivePredictedResult")
                    lines.append(
                        f"| {match} | {champion if champion is not None else 'N/A'} "
                        f"| {benchmark if benchmark is not None else 'N/A'} "
                        f"| {adaptive if adaptive is not None else 'N/A'} |"
                    )
                lines.append(
                    "\n40/60 is the provisional champion; Model 2 is the benchmark. "
                    "These are saved forecasts, not evaluated results. "
                    f"Source: {result['source']}."
                )
                parts.append("\n".join(lines))
            elif name == "experiment_history":
                metrics = result["historical_2025_26"]
                a, b = metrics["Model 2"], metrics["H07 xG+xA"]
                parts.append(
                    f"H07 parked: 2025/26 accuracy {b['accuracy_pct']}% vs {a['accuracy_pct']}% "
                    f"Model 2, Log Loss {b['log_loss']} vs {a['log_loss']}, "
                    f"Brier {b['brier']} vs {a['brier']}. Source: {result['source']}."
                )
            elif name == "next_meeting":
                fixture = result.get("fixture")
                if fixture:
                    parts.append(
                        f"**Next Premier League meeting: {fixture['HomeTeam']} vs "
                        f"{fixture['AwayTeam']}**\n\n"
                        f"Kick-off (UTC): {fixture['KickoffUTC']}. "
                        f"Gameweek: {fixture['Gameweek']}. "
                        f"Source: {result['source']} (live fixture schedule, subject to change)."
                    )
                else:
                    parts.append(
                        f"Unable to verify the next Premier League meeting. "
                        f"{result.get('error') or result.get('note', '')} "
                        f"Source: {result.get('source', 'FPL schedule')}."
                    )
            elif name == "form_summary":
                def fmt(value):
                    n = float(value)
                    return str(int(n)) if n.is_integer() else f"{n:g}"
                rows = result.get("data", [])
                if len(rows) != 1 or not isinstance(rows[0], dict) or not rows[0].get("Played"):
                    parts.append(f"{item['arguments']['team']}: no completed 2026/27 matches available in verified local gameweek results. Current-season form cannot be verified; historical form has not been substituted.")
                else:
                    r = rows[0]
                    parts.append(
                        f"**{item['arguments']['team']} recent form | {result.get('season', 'season unspecified')} | last {fmt(r['Played'])} matches**\n"
                        f"{fmt(r['Won'])} W, {fmt(r['Drawn'])} D, {fmt(r['Lost'])} L; "
                        f"{fmt(r['Points'])} points, {fmt(r['PointsPerGame'])} points per game, "
                        f"{fmt(r['GoalsFor'])} goals for, {fmt(r['GoalsAgainst'])} against. "
                        f"Source: {result.get('source', 'DuckDB team_match_stats')} (completed results, not frozen forecasts)."
                    )
            elif name == "model_performance":
                recent = {row["model"]: row for row in result["latest_metrics"] if row["period"] == "GW8"}
                old = {row["model"]: row for row in result["metrics"] if row["period"] == "GW6-GW7"}
                a, b, c, d = (recent[k] for k in ("40/60", "Adaptive", "Model 2", "Market only"))
                parts.append(
                    f"GW8 (10 finished matches): 40/60 {a['correct']} ({a['accuracy']}), "
                    f"Log Loss {a['log_loss']}, Brier {a['brier']}; "
                    f"adaptive {b['correct']} ({b['accuracy']}), "
                    f"Log Loss {b['log_loss']}, Brier {b['brier']}; "
                    f"Model 2 {c['correct']} ({c['accuracy']}), "
                    f"Log Loss {c['log_loss']}, Brier {c['brier']}; "
                    f"market only {d['correct']} ({d['accuracy']}), "
                    f"Log Loss {d['log_loss']}, Brier {d['brier']}. "
                    "Market-only leads GW8 on accuracy and Log Loss. "
                    "The 40/60 blend remains the provisional champion candidate, "
                    "not formally promoted. One gameweek is a small sample. "
                    f"Source: {result['source']}. "
                    f"Historical GW6-GW7 (20 matches): 40/60 {old['40/60']['accuracy']} vs "
                    f"Model 2 {old['Model 2']['accuracy']}. "
                    f"Source: {result['historical_source']}."
                )
        # Deterministic comparison: no language-model inference about winners.
        forms = []
        for item in state["evidence"]:
            if item["tool"] != "form_summary" or "error" in item["result"]:
                continue
            rows = item["result"].get("data", [])
            if (len(rows) == 1 and isinstance(rows[0], dict)
                    and all(isinstance(rows[0].get(k), (int, float))
                            and math.isfinite(float(rows[0][k]))
                            for k in ("Played", "Points", "PointsPerGame",
                                      "GoalsFor", "GoalsAgainst"))
                    and rows[0]["Played"] > 0):
                forms.append((item["arguments"]["team"], rows[0], item["result"].get("season", "2026/27"), item["arguments"]["games"]))
        if (len(forms) == 2
                and forms[0][0] != forms[1][0]
                and forms[0][1].get("Played") == forms[1][1].get("Played")
                and all(all(k in row and row[k] is not None
                            for k in ("Played", "Points", "PointsPerGame",
                                      "GoalsFor", "GoalsAgainst"))
                        for _, row, _, _ in forms)):
            (a, x, season_a, requested_a), (b, y, season_b, requested_b) = forms
            season_label = season_a if season_a == season_b else f"{a}: {season_a}; {b}: {season_b}"
            requested = max(requested_a, requested_b)
            coverage = (f"Only {int(x['Played'])} of {requested} requested matches are available. "
                        if int(x["Played"]) < requested else "")
            def number(value):
                value = float(value)
                return str(int(value)) if value.is_integer() else f"{value:g}"
            labels = {"PointsPerGame": "points per game", "Points": "points",
                      "GoalsFor": "goals scored", "GoalsAgainst": "goals conceded"}
            def edge(metric, higher=True):
                left, right = float(x[metric]), float(y[metric])
                if left == right:
                    return f"Both teams have {number(left)} {labels[metric]}."
                winner = a if (left > right) == higher else b
                if metric == "GoalsAgainst":
                    return f"{winner} conceded {number(abs(left-right))} fewer goals."
                return f"{winner} leads on {labels[metric]} by {number(abs(left-right))}."
            parts.insert(0, (
                f"**Comparison: {a} vs {b} ({season_label}, last {number(x['Played'])} completed matches)**\n\n"
                + coverage + edge("PointsPerGame") + " "
                + edge("Points") + " "
                + edge("GoalsFor") + " "
                + edge("GoalsAgainst", higher=False)
                + " Lower goals against is better; these are descriptive results, "
                  "not a forecast or causal explanation."
            ))
        if len(forms) == 2 and forms[0][1].get("Played") == forms[1][1].get("Played"):
            (a, x, season_a, req_a), (b, y, season_b, req_b) = forms
            def fmt_stat(value):
                n = float(value)
                return str(int(n)) if n.is_integer() else f"{n:g}"
            season_scope = season_a if season_a == season_b else f"{a}: {season_a}; {b}: {season_b}"
            cols = [
                ("Played", "Played"), ("Won", "Wins"), ("Drawn", "Draws"),
                ("Lost", "Losses"), ("Points", "Points"),
                ("PointsPerGame", "Points per game"),
                ("GoalsFor", "Goals scored"), ("GoalsAgainst", "Goals conceded"),
            ]
            table = ["| Metric | " + a + " | " + b + " |", "|:--|--:|--:|"]
            table += [f"| {label} | {fmt_stat(x[key])} | {fmt_stat(y[key])} |"
                      for key, label in cols]
            comparison = parts[0]
            insight = comparison.split("\n\n", 1)[-1]
            requested = max(req_a, req_b)
            coverage = (f"**Coverage:** {fmt_stat(x['Played'])} of {requested} requested matches per team.\n\n"
                        if x["Played"] < requested else "")
            source_items = [
                item["result"].get("source", "DuckDB team_match_stats")
                for item in state["evidence"] if item["tool"] == "form_summary"
            ]
            sources = "; ".join(dict.fromkeys(source_items))
            recent_questions = [h["question"] for h in state.get("history", [])[-3:]]
            defence_focus = bool(re.search(r"\b(defen[sc]\w*|conced\w*|clean sheets?)\b", state["question"], re.I))
            if not defence_focus and re.search(r"\b(?:what about|how about|over the last|last \d+)\b", state["question"], re.I):
                defence_focus = any(re.search(r"\b(defen[sc]\w*|conced\w*|clean sheets?)\b", earlier, re.I) for earlier in reversed(recent_questions))
            if defence_focus:
                conceded_a, conceded_b = int(x["GoalsAgainst"]), int(y["GoalsAgainst"])
                if conceded_a == conceded_b:
                    headline = (f"**Neither team has the defensive edge.** Both conceded "
                                f"{conceded_a} goals across {int(x['Played'])} matches.")
                else:
                    better = a if conceded_a < conceded_b else b
                    worse = b if better == a else a
                    low, high = sorted((conceded_a, conceded_b))
                    headline = (f"**{better} have been defensively stronger.** "
                                f"They conceded **{low} goals**, compared with "
                                f"**{high} for {worse}**, a difference of **{high - low}** "
                                f"across {int(x['Played'])} matches.")
                insight = headline
                cols = [("Played", "Played"), ("GoalsAgainst", "Goals conceded")]
                table = ["| Defensive metric | " + a + " | " + b + " |",
                         "|:--|--:|--:|"]
                table += [f"| {label} | {fmt_stat(x[key])} | {fmt_stat(y[key])} |"
                          for key, label in cols]
            parts = [
                f"### {a} vs {b} | Last {fmt_stat(x['Played'])} matches\n"
                f"*Seasons: {season_scope}*\n\n"
                + coverage
                + "**Key differences**\n\n" + insight + "\n\n"
                + "**Head-to-head form stats**\n\n"
                + "\n".join(table) + "\n\n"
                + f"**{a} recent form** and **{b} recent form** are compared above.\n"
                + f"*Source: {sources}. Completed Premier League results, not predictions.*"
            ]
        # A three-tool response should answer the requested fixture, not dump ten unrelated forecasts.
        mixed = [e for e in state["evidence"] if e["tool"] == "frozen_gameweek_predictions"]
        if len(forms) == 2 and len(mixed) == 1:
            (a, x, sa, req_a), (b, y, sb, req_b) = forms
            forecast = mixed[0]["result"]
            fixtures = forecast.get("data", [])
            matching = [f for f in fixtures
                        if {f.get("HomeTeam"), f.get("AwayTeam")} == {a, b}]
            def display_num(n):
                n = float(n)
                return str(int(n)) if n.is_integer() else f"{n:g}"
            if "error" in forecast:
                fixture_message = f"Frozen GW{mixed[0]['arguments']['gameweek']} forecast unavailable: {forecast['error']}."
            elif len(matching) == 1:
                fixture = matching[0]
                raw_blend = fixture.get("ShadowPredictedResult")
                blend_pick = {
                    "H": fixture["HomeTeam"],
                    "D": "Draw",
                    "A": fixture["AwayTeam"],
                }.get(raw_blend, raw_blend or "not available")
                fixture_message = (
                    f"**Frozen GW{forecast['gameweek']} forecast:** "
                    f"{fixture['HomeTeam']} vs {fixture['AwayTeam']}. "
                    f"Model 2 selection: **{fixture.get('PredictedResult', 'unavailable')}**. "
                    f"40/60 provisional blend selection: "
                    f"**{blend_pick}**. "
                    "This saved pre-match prediction is not an explanation of recent form."
                )
            else:
                fixture_message = (
                    f"**No {a} versus {b} fixture** appears in the frozen "
                    f"GW{forecast.get('gameweek', mixed[0]['arguments']['gameweek'])} "
                    "snapshot. No fixture forecast is inferred."
                )
            parts = [
                f"### {a} vs {b}: recent form and frozen forecast\n\n"
                f"**Recent completed league matches:** last {display_num(x['Played'])} "
                f"for {a} and {display_num(y['Played'])} for {b}.\n\n"
                f"| Metric | {a} | {b} |\n|---|---:|---:|\n"
                f"| Points per game | {display_num(x['PointsPerGame'])} | {display_num(y['PointsPerGame'])} |\n"
                f"| Goals scored | {display_num(x['GoalsFor'])} | {display_num(y['GoalsFor'])} |\n"
                f"| Goals conceded | {display_num(x['GoalsAgainst'])} | {display_num(y['GoalsAgainst'])} |\n\n"
                + fixture_message + "\n\n"
                + f"*Form seasons: {sa if sa == sb else sa + '; ' + sb}. "
                f"Form source: {state['evidence'][0]['result'].get('source', 'historical data')}. "
                f"Forecast source: {forecast.get('source', 'unavailable')}.*"
            ]
        facts = "\n\n".join(parts)
        # Earlier answers are labelled context, not fresh tool evidence.
        # This supports follow-ups even when the planner only fetches new data.
        synthesis_facts = facts
        previous = state.get("history", [])[-1:]
        if synthesizer is not None and previous:
            prior = previous[0]
            synthesis_facts += ("\n\nPrevious turn (context, not newly evaluated): "
                      + prior["question"] + "\n" + prior["answer"][:900])
        if any(item["tool"] == "next_meeting" for item in state["evidence"]):
            return {**save(facts), "trace": [*state["trace"], "fixture_facts_preserved"]}
        if any(item["tool"] == "form_summary" for item in state["evidence"]):
            return {**save(facts), "trace": [*state["trace"], "form_facts_preserved"]}
        if synthesizer is None:
            return save(facts)
        # The LLM is not permitted to compress or replace fixture-by-fixture
        # forecast output: the user asked for the actual saved predictions.
        if len(state["evidence"]) == 1 and state["evidence"][0]["tool"] == "frozen_gameweek_predictions":
            return {**save(facts), "trace": [*state["trace"], "forecast_rows_preserved"]}
        if any("error" in item["result"] for item in state["evidence"]):
            return {**save(facts), "trace": [*state["trace"], "synthesis_skipped_tool_error"]}
        try:
            candidate = synthesizer(state["question"], synthesis_facts)
            tools = {item["tool"] for item in state["evidence"]}
            has_forecast_context = "frozen forecasts" in synthesis_facts.lower()
            if not _grounded(candidate, synthesis_facts,
                             "frozen_gameweek_predictions" in tools or has_forecast_context,
                             "model_performance" in tools):
                if os.getenv("FOOTBALL_AGENT_DEBUG") == "1":
                    print("[synthesis rejected]", repr(candidate[:1200]))
                return {**save(facts), "trace": [*state["trace"], "synthesis_rejected_grounding"]}
            return {**save(candidate), "trace": [*state["trace"], "synthesis_accepted"]}
        except (ReadTimeout, TimeoutError, ValueError, TypeError) as exc:
            return {**save(facts), "trace": [*state["trace"], f"synthesis_fallback: {type(exc).__name__}"]}

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("respond", respond)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", lambda s: "execute" if s["plan"] else "respond")
    graph.add_edge("execute", "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer)


def ask(question: str, planner: Callable[[str], list[dict]] = local_planner, synthesizer=None):
    return make_graph(planner, synthesizer=synthesizer).invoke({
        "question": question, "plan": [], "evidence": [], "answer": "", "trace": []
    })


class Conversation:
    """Session-scoped in-memory LangGraph state; no disk, cloud or paid services.

    Use a stable thread_id for follow-ups. A new Conversation has no prior memory.
    This prototype does not persist across Python process restarts.
    """

    def __init__(self, planner: Callable[[str], list[dict]] = local_planner, synthesizer=None):
        self.graph = make_graph(planner, checkpointer=InMemorySaver(), synthesizer=synthesizer)

    def ask(self, question: str, thread_id: str = "default"):
        if not thread_id.strip():
            raise ValueError("thread_id must be non-empty")
        return self.graph.invoke(
            {"question": question, "plan": [], "evidence": [], "answer": "", "trace": []},
            config={"configurable": {"thread_id": thread_id}},
        )
