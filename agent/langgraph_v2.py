"""V2 LangGraph prototype. Separate from existing Streamlit runtime until evaluated.

One bounded local-LLM planning call, then at most three read-only tool calls.
No model writes, hidden retries, paid providers, or answer-generation LLM loop.
"""
from __future__ import annotations

import json
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
)

ALLOWED = {
    "frozen_gameweek_predictions": frozen_gameweek_predictions,
    "experiment_history": experiment_history,
    "model_performance": model_performance,
    "form_summary": form_summary,
}
MAX_TOOLS = 3

class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _GWArgs(_Strict):
    gameweek: int = Field(ge=1, le=38)


class _H07Args(_Strict):
    experiment: Literal["H07"]


class _FormArgs(_Strict):
    team: str = Field(min_length=1, max_length=80)
    games: int = Field(ge=1, le=10)


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


class _PerformanceCall(_Strict):
    name: Literal["model_performance"]
    arguments: _EmptyArgs


_ToolCall = Annotated[
    Union[_ForecastCall, _HistoryCall, _PerformanceCall, _FormCall],
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
            elif name == "form_summary":
                try:
                    parsed = _FormArgs.model_validate(args)
                except ValidationError:
                    continue
                args = parsed.model_dump()
            elif args:
                continue
            if {"name": name, "arguments": args} not in valid:
                valid.append({"name": name, "arguments": args})
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
            if "error" in result:
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
            elif name == "form_summary":
                rows = result.get("data", [])
                if len(rows) != 1 or not isinstance(rows[0], dict) or not rows[0].get("Played"):
                    parts.append(f"{item['arguments']['team']}: no completed-match form summary available.")
                else:
                    r = rows[0]
                    parts.append(
                        f"{item['arguments']['team']} recent form (last {r['Played']} matches): "
                        f"{r['Won']} wins, {r['Drawn']} draws, {r['Lost']} losses; "
                        f"{r['Points']} points, {r['PointsPerGame']} points per game, "
                        f"{r['GoalsFor']} goals for, {r['GoalsAgainst']} against. "
                        "Source: DuckDB team_match_stats (most recent matches; not a frozen pre-match snapshot)."
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
        facts = "\n\n".join(parts)
        # Earlier answers are labelled context, not fresh tool evidence.
        # This supports follow-ups even when the planner only fetches new data.
        synthesis_facts = facts
        previous = state.get("history", [])[-1:]
        if synthesizer is not None and previous:
            prior = previous[0]
            synthesis_facts += ("\n\nPrevious turn (context, not newly evaluated): "
                      + prior["question"] + "\n" + prior["answer"][:900])
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
