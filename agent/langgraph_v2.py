"""V2 LangGraph prototype. Separate from existing Streamlit runtime until evaluated.

One bounded local-LLM planning call, then at most three read-only tool calls.
No model writes, hidden retries, paid providers, or answer-generation LLM loop.
"""
from __future__ import annotations

import json
import re
from typing import Callable, TypedDict

from httpx import ReadTimeout

from langgraph.graph import END, START, StateGraph

from agent.football_tools import (
    experiment_history,
    frozen_gameweek_predictions,
    model_performance,
)

ALLOWED = {
    "frozen_gameweek_predictions": frozen_gameweek_predictions,
    "experiment_history": experiment_history,
    "model_performance": model_performance,
}
MAX_TOOLS = 3


class AgentState(TypedDict):
    question: str
    plan: list[dict]
    evidence: list[dict]
    answer: str
    trace: list[str]


def local_planner(question: str) -> list[dict]:
    """Use local Qwen only for concise tool choice, not large evidence prompts."""
    from ollama import Client

    instruction = (
        "Choose read-only tools needed to answer the user's football question. "
        "Return JSON only: an object with a 'calls' list, each item having "
        "'name' and 'arguments'. Allowed: frozen_gameweek_predictions "
        "(arguments: gameweek integer 1..38), experiment_history "
        "(arguments: experiment='H07'), model_performance (arguments: {}). "
        "Choose at most three calls, only when relevant. No invented tools. "
        "Do not provide an answer. "
        "Examples: 'GW8 forecasts?' -> frozen_gameweek_predictions(8); "
        "'Why reject H07?' -> experiment_history(H07); "
        "'How has 40/60 performed?' -> model_performance()."
    )
    result = Client(timeout=60).chat(
        model="qwen3:4b",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": question},
        ],
        format="json",
        think=False,
        options={"num_predict": 120, "num_ctx": 1024, "temperature": 0},
    )
    parsed = json.loads(result.message.content)
    return parsed.get("calls", [])


def make_graph(planner: Callable[[str], list[dict]] = local_planner):
    def plan(state: AgentState):
        try:
            raw = planner(state["question"])
        except (ReadTimeout, TimeoutError) as exc:
            return {"plan": [], "trace": [f"planner_timeout: {type(exc).__name__}"]}
        except (ValueError, KeyError, TypeError) as exc:
            return {"plan": [], "trace": [f"planner_invalid_output: {type(exc).__name__}"]}
        valid = []
        for call in raw[:MAX_TOOLS] if isinstance(raw, list) else []:
            if not isinstance(call, dict):
                continue
            name, args = call.get("name"), call.get("arguments", {})
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
            elif args:
                continue
            if {"name": name, "arguments": args} not in valid:
                valid.append({"name": name, "arguments": args})
        return {"plan": valid, "trace": [f"planned {len(valid)} tool(s)"]}

    def execute(state: AgentState):
        evidence, trace = [], list(state["trace"])
        for call in state["plan"]:
            name = call["name"]
            try:
                result = ALLOWED[name](**call["arguments"])
            except Exception as exc:
                result = {"error": f"Tool execution failed: {type(exc).__name__}"}
            evidence.append({"tool": name, "arguments": call["arguments"], "result": result})
            trace.append(f"executed {name}")
        return {"evidence": evidence, "trace": trace}

    def respond(state: AgentState):
        if not state["evidence"]:
            if any("planner_timeout" in t for t in state["trace"]):
                return {"answer": "Local Qwen planner timed out. No tools were run or data changed; see trace for details."}
            return {"answer": "No supported evidence tool selected. Please specify a gameweek, H07, or model performance."}
        parts = []
        for item in state["evidence"]:
            result, name = item["result"], item["tool"]
            if "error" in result:
                parts.append(f"{name}: {result['error']}")
            elif name == "frozen_gameweek_predictions":
                parts.append(
                    f"GW{result['gameweek']} frozen forecasts: {len(result['data'])} fixtures. "
                    "Model 2 benchmark, 40/60 provisional champion and adaptive challenger "
                    "(where available). The forecasts are saved, not regenerated."
                )
            elif name == "experiment_history":
                metrics = result["historical_2025_26"]
                a, b = metrics["Model 2"], metrics["H07 xG+xA"]
                parts.append(
                    f"H07 parked: 2025/26 accuracy {b['accuracy_pct']}% vs {a['accuracy_pct']}% "
                    f"Model 2, Log Loss {b['log_loss']} vs {a['log_loss']}, "
                    f"Brier {b['brier']} vs {a['brier']}. Source: {result['source']}."
                )
            elif name == "model_performance":
                match = {row["model"]: row for row in result["metrics"] if row["period"] == "GW6-GW7"}
                a, b = match["Model 2"], match["40/60"]
                parts.append(
                    f"GW6-GW7 (20 matches): 40/60 {b['correct']} ({b['accuracy']}), "
                    f"Log Loss {b['log_loss']}, Brier {b['brier']}; "
                    f"Model 2 {a['correct']} ({a['accuracy']}), "
                    f"Log Loss {a['log_loss']}, Brier {a['brier']}. "
                    "Small prospective sample, not evidence of a confirmed model improvement. "
                    f"Source: {result['source']}."
                )
        return {"answer": "\n\n".join(parts)}

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("respond", respond)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", lambda s: "execute" if s["plan"] else "respond")
    graph.add_edge("execute", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


def ask(question: str, planner: Callable[[str], list[dict]] = local_planner):
    return make_graph(planner).invoke({
        "question": question, "plan": [], "evidence": [], "answer": "", "trace": []
    })
