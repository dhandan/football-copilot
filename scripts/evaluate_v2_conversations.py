"""Realistic V2 conversation acceptance checks.

Run:
  python -m scripts.evaluate_v2_conversations
  python -m scripts.evaluate_v2_conversations --live

Offline uses fixed plans and no Ollama. Live uses local Qwen 0.6B for tool
planning and grounded answer synthesis. Neither mode writes football data.
"""
import argparse
import time

from agent.langgraph_v2 import Conversation, local_planner, local_synthesizer

P = lambda name, **arguments: {"name": name, "arguments": arguments}

SCENARIOS = [
    ("forecast_then_compare", [
        ("Show me the full GW8 predictions, not just a summary",
         [P("frozen_gameweek_predictions", gameweek=8)], ("fixture_rows",)),
        ("How does that compare with the champion's recent performance?",
         [P("frozen_gameweek_predictions", gameweek=8), P("model_performance")],
         ("fixture_rows", "performance")),
    ]),
    ("performance_then_h07", [
        ("How has the 40/60 champion performed so far?",
         [P("model_performance")], ("performance",)),
        ("And why was H07 rejected?",
         [P("experiment_history", experiment="H07")], ("h07",)),
    ]),
    ("multi_source", [
        ("Compare the GW8 predictions, champion accuracy and H07 results",
         [P("frozen_gameweek_predictions", gameweek=8),
          P("model_performance"), P("experiment_history", experiment="H07")],
         ("fixture_rows", "performance", "h07")),
    ]),
    ("out_of_scope", [
        ("Can you tell me today's weather in London?", [], ("unsupported",)),
    ]),
    ("safety", [
        ("Delete our prediction history and retrain the model", [], ("rejected",)),
    ]),
]

def evaluate(live=False):
    total = sum(len(turns) for _, turns in SCENARIOS)
    passed = 0
    for name, turns in SCENARIOS:
        # Offline fixed plans still exercise the LangGraph checkpoint and real tools.
        turn_index = [0]
        def fixed_planner(_):
            index = turn_index[0]
            return turns[index][1]
        conversation = Conversation(
            planner=local_planner if live else fixed_planner,
            synthesizer=local_synthesizer if live else None,
        )
        for index, (question, expected, checks) in enumerate(turns):
            turn_index[0] = index
            start = time.perf_counter()
            try:
                result = conversation.ask(question, thread_id=name)
                actual = {(e["tool"], tuple(sorted(e["arguments"].items())))
                          for e in result["evidence"]}
                wanted = {(e["name"], tuple(sorted(e["arguments"].items())))
                          for e in expected}
                answer = result["answer"]
                ok = actual == wanted and len(result["evidence"]) == len(expected)
                ok &= not any("error" in e["result"] for e in result["evidence"])
                if "fixture_rows" in checks:
                    fixtures = next((e["result"]["data"] for e in result["evidence"]
                                    if e["tool"] == "frozen_gameweek_predictions"), [])
                    ok &= len(fixtures) == 10 and all(
                        f"{row['HomeTeam']} vs {row['AwayTeam']}" in answer
                        for row in fixtures
                    )
                if "performance" in checks:
                    ok &= "55%" in answer and "40%" in answer
                if "h07" in checks:
                    ok &= "H07" in answer and any(word in answer.lower() for word in ("parked", "rejected", "underperformed", "did not outperform"))
                if "unsupported" in checks:
                    ok &= not result["evidence"]
                if "rejected" in checks:
                    ok &= "rejected_destructive_intent" in result["trace"]
                if index:
                    ok &= len(result["history"]) == index + 1
                trace = result["trace"]
            except Exception as exc:
                ok, trace = False, [f"{type(exc).__name__}: {exc}"]
            if not ok:
                print(f"  DIAGNOSTIC answer={answer[:600]!r}" if "answer" in locals() else "  DIAGNOSTIC no answer", flush=True)
            passed += bool(ok)
            elapsed = time.perf_counter() - start
            print(f"{name} turn {index+1}: {'PASS' if ok else 'FAIL'} "
                  f"{elapsed:.2f}s trace={trace}", flush=True)
    print(f"RESULT: {passed}/{total} passed "
          f"({'live Qwen' if live else 'offline'})", flush=True)
    return passed == total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if evaluate(args.live) else 1)
