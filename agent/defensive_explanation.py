"""Deterministic evidence for explaining observed Premier League defensive form.

This does not infer tactics, player availability, shot quality or causality.
Numbers are computed from completed result rows before any LLM receives them.
"""
from __future__ import annotations

import math


def _valid_count(row: dict, name: str, played: int) -> int | None:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)) or int(value) != float(value):
        return None
    return int(value) if 0 <= value <= played else None


def defensive_evidence(team_a: str, row_a: dict, team_b: str, row_b: dict) -> str:
    """Explain scoreline patterns, never the unmeasured causes behind them.

    Return an honest limited-data explanation if match-distribution inputs are
    missing (e.g. tests and older outputs without the new optional fields).
    """
    pa, pb = int(row_a["Played"]), int(row_b["Played"])
    if pa < 1 or pa != pb:
        return "The teams do not have a comparable completed-match window."
    ga, gb = float(row_a["GoalsAgainst"]), float(row_b["GoalsAgainst"])
    if not all(math.isfinite(v) and v >= 0 for v in (ga, gb)):
        return "Reliable defensive totals are unavailable."
    fmt = lambda x: str(int(x)) if float(x).is_integer() else f"{x:.2f}"
    report = (
        f"**Why the defensive records differ (observed results):** "
        f"{team_a} conceded {fmt(ga)} in {pa} games "
        f"({ga / pa:.2f} per match), while {team_b} conceded "
        f"{fmt(gb)} ({gb / pb:.2f} per match)."
    )
    ca, cb = (_valid_count(row_a, "CleanSheets", pa),
              _valid_count(row_b, "CleanSheets", pb))
    ma, mb = (_valid_count(row_a, "Conceded2Plus", pa),
              _valid_count(row_b, "Conceded2Plus", pb))
    if ca is not None and cb is not None and ma is not None and mb is not None:
        report += (
            f" {team_a} kept {ca} clean sheets versus {cb} for {team_b}; "
            f"{team_a} conceded two or more in {ma} matches versus {mb} "
            f"for {team_b}."
        )
    else:
        report += (" Match-by-match clean-sheet and heavy-concession counts "
                   "are not available for this comparison.")
    ra = _valid_count(row_a, "ConcededRecent5", pa * 100)
    rb = _valid_count(row_b, "ConcededRecent5", pb * 100)
    oa = _valid_count(row_a, "ConcededPrevious5", pa * 100)
    ob = _valid_count(row_b, "ConcededPrevious5", pb * 100)
    if pa >= 10 and all(v is not None for v in (ra, rb, oa, ob)):
        report += (
            f" Over the most recent five matches {team_a} conceded {ra} "
            f"(previous five: {oa}), versus {team_b} {rb} "
            f"(previous five: {ob})."
        )
    report += (
        " These are **descriptive patterns**, not proof of why they arose. "
        "Match results alone cannot establish tactics, goalkeeper performance, "
        "shot quality, opponent difficulty or injuries."
    )
    return report
