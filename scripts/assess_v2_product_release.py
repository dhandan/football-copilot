"""One bounded V2 product acceptance screen — no AI, secrets or new features.

20 independently specified questions across five real multi-turn journeys.
Comparison arm is the existing (pre-integration) V2 Conversation with a
disabled model planner. This measures the offline deterministic slice only,
NOT full original production or the live Groq-supported baseline.

Scores: appropriate route + verified tools + intended teams/window + an
answer that addresses the task. No invented outcome or automatic promotion.
A passing CI run means the assessment ran, NOT that V2 earned a GO.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent.product_conversation_v2 import ProductConversation
from agent.langgraph_v2 import Conversation
from agent.investigator_v2 import parse_scope


@dataclass(frozen=True)
class Case:
    label: str
    query: str
    teams: tuple[str, ...]
    games: int | None
    intent: str
    expected_tools: tuple[str, ...]


CASES = (
    # A: general comparisons and drilldowns
    Case("a1_overall", "Compare Arsenal and Liverpool over their last 10 Premier League matches",
         ("Arsenal", "Liverpool"), 10, "overall", ("form_summary", "form_summary")),
    Case("a2_attacking", "Compare Arsenal and Liverpool's goals scored over their last 10 matches",
         ("Arsenal", "Liverpool"), 10, "attack", ("form_summary", "form_summary")),
    Case("a3_venue_attack", "How does Arsenal versus Liverpool home and away scoring compare over their last 10 matches?",
         ("Arsenal", "Liverpool"), 10, "venue_attack", ("form_summary", "form_summary")),
    Case("a4_defending", "Who has been better defensively over those 10 matches?",
         ("Arsenal", "Liverpool"), 10, "defence", ("form_summary", "form_summary")),
    Case("a5_opponents", "Compare Arsenal and Liverpool's last 10 matches and the strength of opponents they faced",
         ("Arsenal", "Liverpool"), 10, "opposition", ("form_summary", "form_summary", "compare_opponent_strength")),
    # B: different teams, short window, reasoning and head-to-head
    Case("b1_overall", "Compare Brighton and Brentford over their last seven matches",
         ("Brighton", "Brentford"), 7, "overall", ("form_summary", "form_summary")),
    Case("b2_attack", "Who scored more goals over those seven matches?",
         ("Brighton", "Brentford"), 7, "attack", ("form_summary", "form_summary")),
    Case("b3_why", "Why did Brighton score more than Brentford over their last seven matches?",
         ("Brighton", "Brentford"), 7, "why", ("form_summary", "form_summary")),
    Case("b4_venues", "Compare Brighton and Brentford's home and away defending over their last seven matches",
         ("Brighton", "Brentford"), 7, "venue_defence", ("form_summary", "form_summary")),
    Case("b5_h2h", "Show Brighton and Brentford's historical head-to-head",
         ("Brighton", "Brentford"), None, "h2h", ("head_to_head",)),
    # C: subject switching and trend
    Case("c1_overall", "Compare Chelsea and Tottenham over their last eight matches",
         ("Chelsea", "Tottenham"), 8, "overall", ("form_summary", "form_summary")),
    Case("c2_defence", "Which team conceded fewer across those eight matches?",
         ("Chelsea", "Tottenham"), 8, "defence", ("form_summary", "form_summary")),
    Case("c3_attack", "Compare Chelsea and Tottenham scoring over their last eight matches",
         ("Chelsea", "Tottenham"), 8, "attack", ("form_summary", "form_summary")),
    Case("c4_switch", "Now compare Newcastle and Aston Villa over their last six matches",
         ("Newcastle", "Aston Villa"), 6, "overall", ("form_summary", "form_summary")),
    Case("c5_switch_defence", "How do their defensive records compare over those six matches?",
         ("Newcastle", "Aston Villa"), 6, "defence", ("form_summary", "form_summary")),
    # D: single-team change and safe scope limits
    Case("d1_trend_defence", "Has Arsenal's defence improved in its most recent five matches compared with the previous five?",
         ("Arsenal",), 10, "trend_defence", ("form_summary",)),
    Case("d2_trend_attack", "Has Arsenal's scoring improved in its most recent five matches compared with the previous five?",
         ("Arsenal",), 10, "trend_attack", ("form_summary",)),
    Case("d3_other_club", "Compare Everton and Bournemouth over their last five games",
         ("Everton", "Bournemouth"), 5, "overall", ("form_summary", "form_summary")),
    Case("d4_other_attack", "How do Everton and Bournemouth compare for goals scored over their last five matches?",
         ("Everton", "Bournemouth"), 5, "attack", ("form_summary", "form_summary")),
    Case("d5_unsupported", "What is the temperature in London right now?",
         (), None, "unsupported", ()),
)

assert len(CASES) == 20 and len({c.label for c in CASES}) == 20
assert len({c.intent for c in CASES}) >= 9


def judge(case: Case, result: dict, *, product: bool) -> tuple[bool, list[str]]:
    """Independent contract; use evidence and intent, not vendor-specific phrasing."""
    issues = []
    evidence = result.get("evidence", []) or []
    answer = result.get("answer", "")
    actual = [e.get("tool") for e in evidence if "result" in e]
    if case.intent == "unsupported":
        if product:
            if result.get("route") != "unsupported" or evidence:
                issues.append("unsupported_misrouted")
        elif evidence:
            issues.append("unsupported_evidence")
        return not issues, issues
    if product and (not result.get("complete")
                    or result.get("route") != "deterministic_investigation"):
        issues.append("not_complete_verified")
    if sorted(actual) != sorted(case.expected_tools):
        issues.append("wrong_verified_tools")
    forms = [e for e in evidence if e.get("tool") == "form_summary" and "result" in e]
    named = [e["arguments"]["team"] for e in forms]
    if forms and case.intent != "h2h" and set(named) != set(case.teams):
        issues.append("wrong_clubs")
    if forms and case.games is not None and any(
            e["arguments"]["games"] != case.games or
            int(e["result"]["data"][0]["Played"]) != case.games for e in forms):
        issues.append("wrong_window")
    if not answer or not any(str(e["result"]["data"][0].get("Points", ""))
                             in answer for e in forms):
        if case.intent != "h2h":
            issues.append("missing_verified_summary")
    lower = answer.lower()
    if case.intent == "overall":
        if product and not all(x in lower.split("verified football evidence")[0]
                               for x in ("points", "scored", "conceded")):
            issues.append("headline_not_broad")
    elif case.intent in ("attack", "venue_attack", "trend_attack"):
        if not any(x in lower for x in ("scored", "scoring", "attacking")):
            issues.append("missing_attack_response")
    elif case.intent in ("defence", "venue_defence", "trend_defence"):
        if not any(x in lower for x in ("conceded", "defensive", "defence")):
            issues.append("missing_defence_response")
    if case.intent.startswith("venue") and product:
        if "home" not in lower or "away" not in lower:
            issues.append("missing_venue_response")
    if case.intent.startswith("trend") and product:
        if "preceding five" not in lower:
            issues.append("missing_second_window")
    if case.intent == "opposition" and product:
        if "opponent" not in lower:
            issues.append("missing_opposition_context")
    if case.intent == "h2h" and "head-to-head" not in lower:
        issues.append("missing_h2h_response")
    if case.intent == "why" and product and "cannot establish" not in lower:
        issues.append("unqualified_causality")
    return not issues, issues


def run():
    current, existing = ProductConversation(), Conversation(planner=lambda _: [])
    numbers = {"product": 0, "prior_v2_offline": 0}
    details = []
    for case in CASES:
        # Each letter is a multi-turn user session.
        thread = case.label[0]
        candidate = current.ask(case.query, thread_id=thread)
        control = existing.ask(case.query, thread_id="legacy-" + thread)
        a_ok, a_errors = judge(case, candidate, product=True)
        b_ok, b_errors = judge(case, control, product=False)
        numbers["product"] += a_ok
        numbers["prior_v2_offline"] += b_ok
        details.append({"case": case.label, "v2": a_ok, "existing": b_ok,
                        "v2_issues": a_errors, "existing_issues": b_errors})
        print(f"PRODUCT_CASE {case.label} "
              f"v2={'PASS' if a_ok else 'FAIL'} issues={a_errors} "
              f"existing={'PASS' if b_ok else 'FAIL'} existing_issues={b_errors}",
              flush=True)
    total = len(CASES)
    gap = numbers["product"] - numbers["prior_v2_offline"]
    # A real GO also needs a manual quality review. Automated contracts alone
    # must never grant promotion.
    decision = ("NO-GO: material measured gap not established"
                if gap < 3 or numbers["product"] < 18 else
                "HOLD: automated screen promising; manual quality gate still required")
    print(f"PRODUCT_SUMMARY v2={numbers['product']}/{total} "
          f"prior_v2_offline={numbers['prior_v2_offline']}/{total} "
          f"absolute_case_gap={gap}/{total} decision={decision} "
          "baseline=existing_V2_deterministic_only not_live_production "
          "not_full_semantic_accuracy=True no_promotion=True", flush=True)
    return details, numbers, decision


if __name__ == "__main__":
    run()
