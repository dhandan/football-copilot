# scripts/validate_gameweek_blog.py

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate generated Football Copilot Gameweek blog output."
    )
    parser.add_argument(
        "--gameweek",
        type=int,
        required=True,
        help="Gameweek number to validate, e.g. 2",
    )
    args = parser.parse_args()

    gameweek = args.gameweek
    gw_tag = f"GW{gameweek:02d}"

    evaluation_path = (
        ROOT
        / "data"
        / "live"
        / "evaluations"
        / f"2026_27_gw{gameweek:02d}_evaluation.csv"
    )

    blog_path = (
        ROOT
        / "docs"
        / "gameweeks"
        / f"{gw_tag}.md"
    )

    if not evaluation_path.exists():
        fail(f"Evaluation file not found: {evaluation_path}")

    if not blog_path.exists():
        fail(f"Blog file not found: {blog_path}")

    evaluation = pd.read_csv(evaluation_path)
    blog = blog_path.read_text(encoding="utf-8")

    if evaluation.empty:
        fail("Evaluation CSV is empty.")

    # --------------------------------------------------
    # Expected live values
    # --------------------------------------------------

    matches = len(evaluation)

    correct_outcomes = int(
        evaluation["OutcomeCorrect"]
        .astype(bool)
        .sum()
    )

    modal_one_one = int(
        evaluation["MostLikelyScore"]
        .astype(str)
        .str.strip()
        .eq("1-1")
        .sum()
    )

    actual_one_one = int(
        evaluation["ActualScore"]
        .astype(str)
        .str.strip()
        .eq("1-1")
        .sum()
    )

    # --------------------------------------------------
    # Validate current Gameweek identity
    # --------------------------------------------------

    expected_heading = f"GW{gameweek} produced"
    if expected_heading not in blog:
        fail(
            f"Expected current Gameweek reference '{expected_heading}' "
            "was not found."
        )

    # --------------------------------------------------
    # Detect stale Gameweek references
    # --------------------------------------------------

    stale_gameweeks = []

    for match in re.finditer(r"\bGW(\d+)\b", blog):
        found_gw = int(match.group(1))

        if found_gw != gameweek:
            stale_gameweeks.append(found_gw)

    # Some cross-Gameweek references may be intentional elsewhere in a page,
    # so only fail on stale Gameweeks inside the 'What we learned' section.

    what_we_learned_match = re.search(
        r"## What we learned(.*?)(?:\n---|\Z)",
        blog,
        flags=re.DOTALL,
    )

    if not what_we_learned_match:
        fail("'What we learned' section not found.")

    learned_section = what_we_learned_match.group(1)

    for match in re.finditer(r"\bGW(\d+)\b", learned_section):
        found_gw = int(match.group(1))

        if found_gw not in {gameweek, 5}:
            fail(
                f"Possible stale Gameweek reference GW{found_gw} "
                "found in 'What we learned'."
            )

    # --------------------------------------------------
    # Validate dynamic metrics
    # --------------------------------------------------

    expected_outcome_text = (
        f"GW{gameweek} produced {correct_outcomes} correct "
        f"1X2 outcomes from {matches} matches."
    )

    if expected_outcome_text not in blog:
        fail(
            "Outcome summary does not match evaluation CSV.\n"
            f"Expected: {expected_outcome_text}"
        )

    expected_one_one_text = (
        f"{modal_one_one} of {matches} fixtures had 1-1 as the "
        f"single most likely scoreline, while {actual_one_one} "
        "actually finished 1-1."
    )

    if expected_one_one_text not in blog:
        fail(
            "1-1 diagnostic does not match evaluation CSV.\n"
            f"Expected: {expected_one_one_text}"
        )

    # --------------------------------------------------
    # Detect known stale phrases
    # --------------------------------------------------

    stale_phrases = [
        "GW1 contains only 10 matches",
        "Eight GW1 fixtures",
        "surprising GW1 results",
        "after one Gameweek",
        "seven of the ten GW1 fixtures",
        "five correct 1X2 outcomes from ten matches",
        "none actually finished 1-1",
        "Four areas are now being monitored",
    ]

    for phrase in stale_phrases:
        if phrase.lower() in blog.lower():
            fail(f"Known stale phrase found: '{phrase}'")

    # --------------------------------------------------
    # Structural checks
    # --------------------------------------------------

    if "Five areas are being monitored" not in learned_section:
        fail(
            "Expected five-area monitoring statement "
            "not found in 'What we learned'."
        )

    if "Model 2 remains frozen as the official live benchmark" not in learned_section:
        fail(
            "Frozen Model 2 benchmark statement not found."
        )

    # --------------------------------------------------
    # Success
    # --------------------------------------------------

    print("FOOTBALL COPILOT")
    print("GAMEWEEK BLOG VALIDATION")
    print("========================")
    print()
    print(f"Gameweek: {gameweek}")
    print(f"Matches: {matches}")
    print(f"Correct outcomes: {correct_outcomes}")
    print(f"Modal 1-1 predictions: {modal_one_one}")
    print(f"Actual 1-1 results: {actual_one_one}")
    print()
    print("PASS: Generated Gameweek blog matches evaluation data.")


if __name__ == "__main__":
    main()