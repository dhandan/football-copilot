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
    ("cross_season_defensive_followup", [
        ("Compare Liverpool and Arsenal over their last 10 matches",
         [P("form_summary", team="Liverpool", games=10),
          P("form_summary", team="Arsenal", games=10)], ("form_pair",)),
        ("Who has been defensively better over those 10 matches?",
         [P("form_summary", team="Liverpool", games=10),
          P("form_summary", team="Arsenal", games=10)], ("form_pair", "defence")),
        ("What about over the last 15 matches?",
         [P("form_summary", team="Liverpool", games=15),
          P("form_summary", team="Arsenal", games=15)], ("form_pair", "defence")),
        ("What about over the last 5?",
         [P("form_summary", team="Liverpool", games=5),
          P("form_summary", team="Arsenal", games=5)], ("form_pair", "defence")),
    ]),
    ("matching_frozen_fixture", [
        ("Compare Chelsea and Tottenham's last 5 matches with the GW8 frozen prediction",
         [P("form_summary", team="Chelsea", games=5),
          P("form_summary", team="Tottenham", games=5),
          P("frozen_gameweek_predictions", gameweek=8)],
         ("form_pair", "matching_fixture")),
    ]),
    ("missing_frozen_fixture", [
        ("Compare Liverpool and Arsenal's last 5 matches with the GW8 frozen prediction",
         [P("form_summary", team="Liverpool", games=5),
          P("form_summary", team="Arsenal", games=5),
          P("frozen_gameweek_predictions", gameweek=8)],
         ("form_pair", "missing_fixture")),
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
    durations = []
    for name, turns in SCENARIOS:
        # Offline fixed plans still exercise the LangGraph checkpoint and real tools.
        turn_index = [0]
        def fixed_planner(_):
            index = turn_index[0]
            return turns[index][1]
        # Measure the two local LLM stages independently without changing
        # the agent, its tool selection or its grounding behaviour.
        stage = {"planning": 0.0, "synthesis": 0.0, "synthesis_calls": 0}

        def timed_planner(question):
            started = time.perf_counter()
            try:
                return local_planner(question)
            finally:
                stage["planning"] += time.perf_counter() - started

        def timed_synthesizer(question, facts):
            started = time.perf_counter()
            stage["synthesis_calls"] += 1
            try:
                return local_synthesizer(question, facts)
            finally:
                stage["synthesis"] += time.perf_counter() - started

        conversation = Conversation(
            planner=timed_planner if live else fixed_planner,
            synthesizer=timed_synthesizer if live else None,
        )
        for index, (question, expected, checks) in enumerate(turns):
            turn_index[0] = index
            stage.update(planning=0.0, synthesis=0.0, synthesis_calls=0)
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
                if "form_pair" in checks:
                    form = [e for e in result["evidence"] if e["tool"] == "form_summary"]
                    ok &= len(form) == 2
                    if len(form) == 2:
                        ok &= all(e["result"].get("data") and
                                  e["result"]["data"][0].get("Played") == e["arguments"]["games"]
                                  for e in form)
                        ok &= all(e["result"].get("source") for e in form)
                        if "defence" in checks and all(e["result"].get("data") for e in form):
                            a, b = (e["result"]["data"][0]["GoalsAgainst"] for e in form)
                            expected_winner = ("Neither team" if a == b else
                                               form[0]["arguments"]["team"] if a < b else
                                               form[1]["arguments"]["team"])
                            ok &= expected_winner in answer
                            ok &= str(abs(a - b)) in answer or a == b
                            ok &= "Goals conceded" in answer
                if "matching_fixture" in checks or "missing_fixture" in checks:
                    frozen = next((e["result"] for e in result["evidence"]
                                   if e["tool"] == "frozen_gameweek_predictions"), {})
                    teams = [e["arguments"]["team"] for e in result["evidence"]
                             if e["tool"] == "form_summary"]
                    matches = [row for row in frozen.get("data", [])
                               if {row.get("HomeTeam"), row.get("AwayTeam")} == set(teams)]
                    if "matching_fixture" in checks:
                        ok &= len(matches) == 1
                        if len(matches) == 1:
                            fixture = matches[0]
                            expected_blend = {"H": fixture["HomeTeam"], "A": fixture["AwayTeam"],
                                              "D": "Draw"}.get(fixture.get("ShadowPredictedResult"),
                                                              fixture.get("ShadowPredictedResult"))
                            ok &= f"{fixture['HomeTeam']} vs {fixture['AwayTeam']}" in answer
                            ok &= f"Model 2 selection: **{fixture['PredictedResult']}**" in answer
                            if expected_blend:
                                ok &= f"blend selection: **{expected_blend}**" in answer
                            ok &= "No " + teams[0] + " versus " + teams[1] + " fixture" not in answer
                    else:
                        ok &= not matches
                        ok &= "No fixture forecast is inferred" in answer
                    ok &= "frozen forecasts (10 fixtures)" not in answer.lower()
                    ok &= frozen.get("source", "") in answer
                    ok &= "not an explanation of recent form" in answer.lower() or "No fixture forecast is inferred" in answer
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
            durations.append(elapsed)
            # Remaining time includes graph overhead and real tool execution;
            # it is not claimed as exclusive tool latency.
            other = max(0.0, elapsed - stage["planning"] - stage["synthesis"])
            synthesis_outcome = (
                "accepted" if "synthesis_accepted" in trace else
                "rejected_or_failed" if stage["synthesis_calls"] else
                "not_attempted"
            )
            print(f"{name} turn {index+1}: {'PASS' if ok else 'FAIL'} "
                  f"{elapsed:.2f}s "
                  f"stages=planner:{stage['planning']:.2f}s,"
                  f"synth:{stage['synthesis']:.2f}s,"
                  f"tools+graph:{other:.2f}s "
                  f"synthesis={synthesis_outcome} trace={trace}", flush=True)
    print(f"LATENCY: mean={sum(durations)/len(durations):.2f}s max={max(durations):.2f}s", flush=True)
    print(f"RESULT: {passed}/{total} passed "
          f"({'live Qwen' if live else 'offline'})", flush=True)
    return passed == total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if evaluate(args.live) else 1)
