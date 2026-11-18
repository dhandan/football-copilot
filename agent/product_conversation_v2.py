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
            return {**result, "route": "deterministic_investigation",
                    "planner_requests": 0}
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
