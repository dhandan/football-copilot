"""Integrated, facts-first product route for the private Football Copilot V2.

No new planning model. Supported completed-match questions use existing
verified read-only tools, conversational scope and audited conclusions.
Unsupported/prediction requests go to the existing V2 Conversation, if an
explicitly configured provider is available. This is NOT production promotion.
"""
from __future__ import annotations

from dataclasses import replace

from agent.investigator_v2 import Investigator, parse_scope, requirements
from scripts.audit_v2_investigator_answers import audit_answer


class _CompiledPlanner:
    """Choose exact read-only requirements without a remote model."""
    def __init__(self):
        self.first = []
        self.later = []
        self.calls = 0

    def configure(self, scope):
        compiled = requirements(scope)
        self.first = compiled[:3]
        self.later = compiled[3:]
        if scope.investigate_why and not scope.opposition and len(scope.teams) == 2:
            candidate = {"name": "compare_opponent_strength", "arguments": {
                "team1": scope.teams[0], "team2": scope.teams[1],
                "games": scope.games}}
            if scope.games is not None and candidate not in compiled:
                self.later.append(candidate)
        self.calls = 0

    def __call__(self, instruction):
        self.calls += 1
        if "Second and FINAL pass" in instruction:
            return self.later[:3]
        return self.first


def suggested_followups(scope) -> list[str]:
    """Two grounded next questions based only on the previous verified scope.

    These are suggestions, never hidden background tool calls.  A fresh query
    still goes through the normal deterministic parser and evidence audit.
    """
    if not scope or not scope.teams or scope.games is None:
        return []
    if not 1 <= scope.games <= 30:
        return []
    if len(scope.teams) == 2:
        a, b = scope.teams
        prefix = f"Compare {a} and {b}"
        if scope.focus == "attack":
            choices = [
                f"{prefix}'s defending over their last {scope.games} matches.",
                f"How does {a} versus {b} home and away scoring compare over their last {scope.games} matches?",
            ]
        else:
            choices = [
                f"{prefix}'s goals scored over their last {scope.games} matches.",
                f"How do {a} and {b} compare defensively at home and away over their last {scope.games} matches?",
            ]
        return choices
    team = scope.teams[0]
    if scope.trend:
        return [f"How has {team}'s attacking record changed over its most recent five matches compared with the previous five?"]
    return [
        f"Has {team}'s defence improved over its most recent five matches compared with the previous five?",
        f"How has {team} scored over its last {scope.games} matches?",
    ]


def general_comparison_lead(question: str, scope, evidence: list[dict]) -> str | None:
    """Answer open compare-form questions across points, scoring and defending.

    Specific goal/defensive/venue/opponent/H2H questions retain their existing
    specialist answer. Every stated number comes from the two verified rows.
    """
    import re
    if (len(scope.teams) != 2 or scope.games is None or scope.trend
            or scope.h2h or scope.opposition or scope.venues
            or scope.investigate_why
            or not re.search(r"\\b(?:compare|comparison)\\b", question, re.I)
            or re.search(
                r"\\b(?:defen[sc](?:e|ive)|defend|conced|attack|scor|"
                r"goals?|home|away|opponent|points?|ppg)\\b", question, re.I)):
        return None
    rows = {e["arguments"]["team"]: e["result"]["data"][0]
            for e in evidence if e.get("tool") == "form_summary"
            and "result" in e}
    a, b = scope.teams
    if set(rows) != {a, b}:
        return None
    ra, rb = rows[a], rows[b]
    if int(ra["Played"]) != scope.games or int(rb["Played"]) != scope.games:
        return None
    pa, pb = int(ra["Points"]), int(rb["Points"])
    fa, fb = int(ra["GoalsFor"]), int(rb["GoalsFor"])
    ga, gb = int(ra["GoalsAgainst"]), int(rb["GoalsAgainst"])
    points = (f"{a} and {b} earned {pa} points each" if pa == pb else
              f"{a if pa > pb else b} earned more points "
              f"({a} {pa}, {b} {pb})")
    attack = (f"both scored {fa} goals" if fa == fb else
              f"{a if fa > fb else b} scored more goals "
              f"({fa} versus {fb})")
    defence = (f"both conceded {ga}" if ga == gb else
               f"{a if ga < gb else b} conceded fewer "
               f"({ga} versus {gb})")
    return (f"Over their last {scope.games} matches, {points}; "
            f"{attack}; and {defence}. "
            "This compares observed results, not underlying causes.")


class ProductConversation:
    """A single session with deterministic investigation and legacy fallback."""

    def __init__(self, legacy=None):
        self.legacy = legacy
        self.planner = _CompiledPlanner()
        self.investigator = Investigator(self.planner)

    def ask(self, question: str, thread_id: str = "product") -> dict:
        previous = self.investigator.context.get(thread_id)
        scope = parse_scope(question, previous)
        # Never hijack prediction/H07/performance enquiries.  They belong to
        # the original V2 experience, which remains unchanged.
        eligible = bool(requirements(scope)) and (
            scope.games is not None or scope.h2h)
        if eligible:
            self.planner.configure(scope)
            result = self.investigator.ask(question, thread_id=thread_id)
            evidence = result.get("evidence", [])
            has_context = any(e.get("tool") == "compare_opponent_strength"
                              and "result" in e for e in evidence)
            render_scope = replace(scope, opposition=True) if has_context else scope
            issues = (audit_answer(result["answer"], render_scope, evidence)
                      if result["complete"] else [])
            if issues:
                # Never display an apparently complete but ungrounded response.
                return {
                    "answer": ("The match data was retrieved but the generated "
                               "comparison failed verification. No conclusion "
                               "will be presented."),
                    "evidence": evidence,
                    "trace": result["trace"] + ["answer_audit_failed"],
                    "complete": False,
                    "route": "verified_data_rejected",
                    "planner_requests": 0,
                }
            if result["complete"]:
                general = general_comparison_lead(question, scope, evidence)
                if general:
                    marker = "### Verified football evidence"
                    if result["answer"].startswith("### Answer\\n") and marker in result["answer"]:
                        result["answer"] = ("### Answer\\n" + general + "\\n\\n" + marker
                                            + result["answer"].split(marker, 1)[1])
            return {**result, "route": "deterministic_investigation",
                    "planner_requests": 0,
                    "followups": suggested_followups(scope) if result["complete"] else []}
        if self.legacy is None:
            return {
                "answer": ("That question isn't currently supported by the "
                           "verified form investigator. Ask about a named team "
                           "and a last-N match window, or enable the private "
                           "Groq preview for other supported V2 questions."),
                "evidence": [], "trace": ["unsupported_without_groq"],
                "complete": False, "route": "unsupported",
                "planner_requests": 0,
            }
        result = self.legacy.ask(question, thread_id=thread_id)
        return {**result, "route": "legacy_v2",
                "planner_requests": None}
