"""V2 LangGraph evaluation. Default: 12 offline cases; --live: local Qwen selection.

No files or models are modified during evaluation.
"""
import argparse
import time

from agent.langgraph_v2 import ask

CASES = [
    ("Show frozen GW8 forecasts", [("frozen_gameweek_predictions", {"gameweek": 8})]),
    ("What are the predictions for gameweek 8?", [("frozen_gameweek_predictions", {"gameweek": 8})]),
    ("Why was H07 rejected?", [("experiment_history", {"experiment": "H07"})]),
    ("Summarise H07 experiment findings", [("experiment_history", {"experiment": "H07"})]),
    ("How accurate has our 40/60 champion been?", [("model_performance", {})]),
    ("Compare Model 2 and 40/60 performance", [("model_performance", {})]),
    ("How has our champion performed and why was H07 rejected?", [("model_performance", {}), ("experiment_history", {"experiment": "H07"})]),
    ("Show GW8 predictions and explain H07 rejection", [("frozen_gameweek_predictions", {"gameweek": 8}), ("experiment_history", {"experiment": "H07"})]),
    ("Compare champion performance with GW8 forecasts", [("model_performance", {}), ("frozen_gameweek_predictions", {"gameweek": 8})]),
    ("Compare model performance, GW8 forecasts and H07 findings", [("model_performance", {}), ("frozen_gameweek_predictions", {"gameweek": 8}), ("experiment_history", {"experiment": "H07"})]),
    ("What is the weather in Paris?", []),
    ("Delete our prediction history and retrain the model", []),
]


def evaluate(live=False):
    passed = 0
    for i, (question, expected) in enumerate(CASES, 1):
        expected_set = {(name, tuple(sorted(args.items()))) for name, args in expected}
        fixed_plan = [{"name": name, "arguments": args} for name, args in expected]
        started = time.perf_counter()
        try:
            result = ask(question) if live else ask(question, planner=lambda q, p=fixed_plan: p)
            evidence = result["evidence"]
            actual_set = {(e["tool"], tuple(sorted(e["arguments"].items()))) for e in evidence}
            tool_errors = [e for e in evidence if "error" in e["result"]]
            timed_out = any("planner_timeout" in t for t in result["trace"])
            ok = (actual_set == expected_set and len(evidence) == len(expected)
                  and not tool_errors and not timed_out and bool(result["answer"]))
            summary = [e["tool"] for e in evidence]
            trace = result["trace"]
        except Exception as exc:
            ok = False
            summary = []
            trace = [f"exception: {type(exc).__name__}: {exc}"]
        elapsed = time.perf_counter() - started
        passed += ok
        print(f"{i:02d}/12 {'PASS' if ok else 'FAIL'} {elapsed:.2f}s tools={summary} trace={trace}", flush=True)
    print(f"RESULT: {passed}/12 passed ({'live Qwen' if live else 'offline deterministic'})", flush=True)
    return passed == len(CASES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run 12 local Qwen planning calls")
    args = parser.parse_args()
    raise SystemExit(0 if evaluate(args.live) else 1)
