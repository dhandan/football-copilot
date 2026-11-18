"""Independent, offline audit of the investigator's rendered football answers.

The guided planner is an engineering control, NOT an AI benchmark. This audit
checks rendered numbers against the underlying real-data tool responses rather
than treating a completed tool plan as an accurate answer. No Groq calls.
"""
from __future__ import annotations

import re

from agent.investigator_v2 import Investigator, requirements, parse_scope
from scripts.evaluate_v2_investigator import CASES, EXPECTED


def audit_answer(answer: str, scope, evidence: list[dict]) -> list[str]:
    problems = []
    by_tool = {}
    for item in evidence:
        if "result" in item:
            by_tool.setdefault(item["tool"], []).append(item)
    forms = by_tool.get("form_summary", [])
    if scope.games is not None:
        if len(forms) != len(scope.teams):
            problems.append("missing_form_evidence")
        for team in scope.teams:
            matching = [e for e in forms if e["arguments"]["team"] == team]
            if len(matching) != 1:
                problems.append("missing_or_duplicate_form:" + team)
                continue
            row = matching[0]["result"]["data"][0]
            expected = (f"| {team} | {int(row['Played'])} | "
                        f"{int(row['Points'])} | {float(row['PointsPerGame']):.2f} | "
                        f"{int(row['GoalsFor'])} | {int(row['GoalsAgainst'])} |")
            if expected not in answer:
                problems.append("wrong_or_missing_form_row:" + team)
            if scope.venues:
                split = row.get("VenueSplit", {})
                if row.get("VenueKnown") != row["Played"]:
                    problems.append("unverified_venue:" + team)
                for venue in ("Home", "Away"):
                    v = split.get(venue, {})
                    metric = "GoalsFor" if scope.focus == "attack" else "GoalsAgainst"
                    n, ga = v.get("Played", 0), v.get(metric)
                    if not isinstance(n, int) or n <= 0 or not isinstance(ga, int):
                        problems.append("missing_venue:" + team + ":" + venue)
                        continue
                    expected = (f"{team}, {venue.lower()}: {ga} goals "
                                f"{'scored' if scope.focus == 'attack' else 'conceded'} in {n} matches ({ga/n:.2f} per match).")
                    if expected not in answer:
                        problems.append("wrong_or_missing_venue:" + team + ":" + venue)
        if len(forms) == 2 and scope.focus != "attack":
            fa, fb = forms
            a, b = fa["result"]["data"][0], fb["result"]["data"][0]
            if a["Played"] == b["Played"]:
                ga_a, ga_b = int(a["GoalsAgainst"]), int(b["GoalsAgainst"])
                if ga_a == ga_b:
                    if "Both teams conceded the same number of goals." not in answer:
                        problems.append("missing_equal_defensive_conclusion")
                else:
                    leader = fa["arguments"]["team"] if ga_a < ga_b else fb["arguments"]["team"]
                    if f"**{leader} conceded {abs(ga_a-ga_b)} fewer goals**" not in answer:
                        problems.append("wrong_defensive_conclusion")
    if scope.opposition:
        items = by_tool.get("compare_opponent_strength", [])
        if len(items) != 1:
            problems.append("missing_opponent_evidence")
        else:
            data = items[0]["result"]["data"]
            for team in scope.teams:
                row = data.get(team, {})
                ppg = row.get("mean_opponent_prior_ppg")
                text = f"{float(ppg):.2f}" if ppg is not None else "unavailable"
                expected = (f"| {team} | {row.get('rated', 0)}/{row.get('played', 0)} "
                            f"| {text} |")
                if expected not in answer:
                    problems.append("wrong_or_missing_opponent_row:" + team)
            if "not a validated defensive adjustment or causal finding" not in answer:
                problems.append("missing_opponent_caveat")
    if scope.h2h:
        items = by_tool.get("head_to_head", [])
        if len(items) != 1 or not items[0]["result"].get("data"):
            problems.append("missing_h2h_evidence")
        else:
            rows = items[0]["result"]["data"]
            for row in rows[:min(scope.games or 5, 30)]:
                expected = (f"| {str(row['Date'])[:10]} | {row['Venue']} | "
                            f"{int(row['GoalsFor'])}-{int(row['GoalsAgainst'])} |")
                if expected not in answer:
                    problems.append("wrong_or_missing_h2h_row")
                    break
    # Direct conclusions must be present and agree with underlying rows.
    if len(forms) == 2 and scope.focus != "attack":
        fa, fb = forms
        a, b = fa["result"]["data"][0], fb["result"]["data"][0]
        if a["Played"] == b["Played"]:
            if a["GoalsAgainst"] == b["GoalsAgainst"]:
                expected = "**Overall verdict:** The teams conceded equally"
            else:
                leader = (fa["arguments"]["team"] if a["GoalsAgainst"] < b["GoalsAgainst"]
                          else fb["arguments"]["team"])
                expected = f"**Overall verdict:** {leader} conceded fewer goals"
            if expected not in answer:
                problems.append("missing_or_wrong_overall_verdict")
        if scope.venues:
            for venue in ("Home", "Away"):
                va, vb = a["VenueSplit"][venue], b["VenueSplit"][venue]
                left = int(va["GoalsAgainst"]) * int(vb["Played"])
                right = int(vb["GoalsAgainst"]) * int(va["Played"])
                if left == right:
                    expected = f"**{venue} verdict:** Both teams conceded at the same rate"
                else:
                    leader = (fa["arguments"]["team"] if left < right
                              else fb["arguments"]["team"])
                    expected = f"**{venue} verdict:** {leader} conceded at a lower rate"
                if expected not in answer:
                    problems.append("missing_or_wrong_venue_verdict:" + venue)
    if scope.opposition and len(forms) == 2:
        items = by_tool.get("compare_opponent_strength", [])
        if len(items) == 1:
            team_a, team_b = scope.teams
            data = items[0]["result"]["data"]
            pa = data.get(team_a, {}).get("mean_opponent_prior_ppg")
            pb = data.get(team_b, {}).get("mean_opponent_prior_ppg")
            if pa is not None and pb is not None:
                if float(pa) == float(pb):
                    expected = "**Opposition verdict:** The measured opponents had equal"
                else:
                    leader = team_a if float(pa) > float(pb) else team_b
                    expected = f"**Opposition verdict:** {leader} faced opponents"
                if expected not in answer:
                    problems.append("missing_or_wrong_opposition_verdict")
    # The direct answer must reflect verified figures, not merely restate a table.
    if len(forms) == 2 and not any(item.get("error") for item in evidence):
        fa, fb = forms
        a, b = fa["result"]["data"][0], fb["result"]["data"][0]
        ta, tb = fa["arguments"]["team"], fb["arguments"]["team"]
        if a["Played"] == b["Played"]:
            if scope.focus == "attack":
                gf_a, gf_b = int(a["GoalsFor"]), int(b["GoalsFor"])
                lead = (f"{ta} and {tb} scored {gf_a} goals each" if gf_a == gf_b
                        else f"{ta if gf_a > gf_b else tb} scored more goals")
            elif int(a["GoalsAgainst"]) == int(b["GoalsAgainst"]):
                lead = f"{ta} and {tb} conceded the same number of goals"
            else:
                winner = ta if int(a["GoalsAgainst"]) < int(b["GoalsAgainst"]) else tb
                lead = f"{winner} had the better goals-conceded record"
            if not answer.startswith("### Answer\n" + lead):
                problems.append("missing_or_wrong_lead")
            if scope.opposition:
                items = by_tool.get("compare_opponent_strength", [])
                if items:
                    data = items[0]["result"]["data"]
                    pa = data.get(ta, {}).get("mean_opponent_prior_ppg")
                    pb = data.get(tb, {}).get("mean_opponent_prior_ppg")
                    if pa is not None and pb is not None and abs(float(pa)-float(pb)) < 0.10:
                        if "not proof of materially tougher opposition" not in answer:
                            problems.append("small_opponent_gap_overstated_or_missing")
    if scope.focus == "attack" and len(forms) == 2:
        ta, tb = forms[0]["arguments"]["team"], forms[1]["arguments"]["team"]
        a, b = forms[0]["result"]["data"][0], forms[1]["result"]["data"][0]
        if a["Played"] == b["Played"]:
            if a["GoalsFor"] == b["GoalsFor"]:
                expected_attack = "Both teams scored the same number of goals."
            else:
                winner = ta if a["GoalsFor"] > b["GoalsFor"] else tb
                expected_attack = f"**{winner} scored {abs(a['GoalsFor']-b['GoalsFor'])} more goals**"
            if expected_attack not in answer:
                problems.append("missing_or_wrong_attacking_verdict")
            expected_points = f"Points earned: {ta} {int(a['Points'])}, {tb} {int(b['Points'])}."
            if expected_points not in answer:
                problems.append("missing_points_comparison")
    if scope.trend:
        for team in scope.teams:
            matching = [e for e in forms if e["arguments"]["team"] == team]
            if len(matching) != 1:
                problems.append("missing_trend_team:" + team)
                continue
            row = matching[0]["result"]["data"][0]
            measure = "Scored" if scope.focus == "attack" else "Conceded"
            keys = (f"{measure}Recent5", f"{measure}Previous5",
                    "PointsRecent5", "PointsPrevious5")
            if not all(k in row for k in keys):
                problems.append("missing_trend_metrics:" + team)
                continue
            recent, previous, points_a, points_b = (int(row[k]) for k in keys)
            direction = ("Unchanged" if recent == previous else
                         "Improved" if (recent > previous if scope.focus == "attack"
                                        else recent < previous) else "Worsened")
            metric = "goals scored" if scope.focus == "attack" else "goals conceded"
            expected_trend = (f"**{team} five-match trend:** {direction} on {metric}: "
                              f"{recent} in the most recent five versus {previous} in "
                              f"the preceding five. Points: {points_a} versus {points_b}.")
            if expected_trend not in answer:
                problems.append("missing_or_wrong_trend:" + team)
            if not answer.startswith("### Answer\\n") or (
                    f"{team} five-match trend:" not in answer.split(
                        "### Verified football evidence")[0]):
                problems.append("missing_trend_lead:" + team)
    if forms and "*Form source:" not in answer:
        problems.append("missing_form_attribution")
    if any(item.get("error") for item in evidence):
        problems.append("unavailable_evidence")
    if "Unresolved evidence:" in answer:
        problems.append("unresolved_evidence")
    return problems


def audit_offline() -> tuple[int, int]:
    passed = 0
    for name, question, followup in CASES:
        first_scope = parse_scope(question)
        follow_scope = parse_scope(followup, first_scope) if followup else None
        def planner(prompt, first_scope=first_scope, follow_scope=follow_scope):
            scope = (follow_scope if follow_scope and "User question: 'And what about"
                     in prompt else first_scope)
            needed = requirements(scope)
            if "Second and FINAL pass" in prompt:
                return needed[2:3]
            return needed[:2]
        agent = Investigator(planner)
        initial = agent.ask(question, thread_id=name)
        result = agent.ask(followup, thread_id=name) if followup else initial
        scope = agent.context[name]
        expected = EXPECTED[name]
        found = (scope.teams, scope.games, scope.h2h, scope.opposition, scope.venues)
        issues = audit_answer(result["answer"], scope, result["evidence"])
        if found != expected:
            issues.append("incorrect_scope")
        if not result["complete"]:
            issues.append("incomplete_evidence")
        success = not issues
        passed += int(success)
        print(f"ANSWER_AUDIT {name} {'PASS' if success else 'FAIL'} issues={issues}", flush=True)
    print(f"ANSWER_AUDIT_SUMMARY verified={passed}/{len(CASES)} "
          "guided_planner=True not_full_semantic_accuracy=True", flush=True)
    return passed, len(CASES)


if __name__ == "__main__":
    ok, total = audit_offline()
    raise SystemExit(0 if ok == total else 1)
