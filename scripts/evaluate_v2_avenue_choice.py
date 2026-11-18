"""Two-avenue selection screen: independent intent labels, verified evidence.

Offline default compares a simple keyword baseline with fixed expected labels.
Live Groq is explicit opt-in, bounded to six single-request selections. A
selection pass is NOT proof of autonomous agent value or causal explanation.
"""
from __future__ import annotations

import argparse
import time
from agent.groq_v2 import GroqUnavailable
from agent.avenue_selection_v2 import (
    CASES, choice_prompt, investigate, observed_form, select_keyword, select_live,
)


def run(live_groq=False):
    if live_groq:
        from agent import groq_v2
        if not groq_v2.groq_ready():
            raise ValueError("Groq Free/ZDR/key preflight failed; no remote calls")
        if len(CASES) > 8:
            raise ValueError("Live screen exceeds one-off request budget")
    counts = {"baseline": 0, "groq": 0}
    for case in CASES:
        start = time.perf_counter()
        scope, first = observed_form(case.question)
        assert len(first) == 2
        baseline = select_keyword(case.question)
        counts["baseline"] += baseline == case.expected
        failure_type = None
        if live_groq:
            try:
                selected = select_live(case.question, scope, first)
                data = investigate(case.question, selected)
                if selected != "none" and data["extra"] is None:
                    raise ValueError("Selected avenue lacks verified evidence")
                counts["groq"] += selected == case.expected
            except (GroqUnavailable, ValueError) as exc:
                # One attempt per case. Record only the exception class,
                # never provider messages, response text, prompts or secrets.
                selected = None
                failure_type = type(exc).__name__
        else:
            selected = None
        # Validate each expected avenue against real evidence independently.
        oracle = investigate(case.question, case.expected)
        if case.expected != "none" and not oracle["extra"]:
            raise ValueError("Benchmark target cannot produce grounded evidence")
        print(f"AVENUE_CASE {case.label} "
              f"baseline={'PASS' if baseline == case.expected else 'FAIL'} "
              f"groq={'PASS' if selected == case.expected else 'FAIL' if live_groq else 'NOT_RUN'} "
              f"expected={case.expected} "
              f"selected={selected or 'unavailable' if live_groq else 'not_run'} "
              f"failure_type={failure_type or 'none'} "
              f"elapsed_s={time.perf_counter()-start:.3f}", flush=True)
    n = len(CASES)
    print(f"AVENUE_SUMMARY baseline={counts['baseline']}/{n} "
          f"live_groq={counts['groq'] if live_groq else 'NOT_RUN'}"
          f"{('/'+str(n)) if live_groq else ''} "
          "independent_labels=True two_avenues_plus_none=True "
          "not_causal_or_novel_reasoning=True no_production_promotion=True", flush=True)
    return counts, n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-groq", action="store_true")
    args = parser.parse_args()
    counts, total = run(args.live_groq)
    raise SystemExit(0 if (counts["groq"] == total if args.live_groq
                           else counts["baseline"] == total) else 1)
