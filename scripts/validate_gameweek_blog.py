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
        description="Validate Football Copilot Gameweek blog output."
    )
    parser.add_argument(
        "--gameweek",
        type=int,
        required=True,
        help="Gameweek number to validate, e.g. 4",
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

    accuracy = correct_outcomes / matches * 100

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

    actual_draws = int(
        evaluation["ActualResult"]
        .astype(str)
        .str.strip()
        .eq("Draw")
        .sum()
    )

    predicted_draws = int(
        evaluation["PredictedResult"]
        .astype(str)
        .str.strip()
        .eq("Draw")
        .sum()
    )

    # --------------------------------------------------
    # Structural checks
    # --------------------------------------------------

    required_sections = [
        "## Pre-match predictions",
        "## Actual results",
        "## Gameweek performance",
        "## Diagnostics",
        "## What we learned",
    ]

    for section in required_sections:
        if section not in blog:
            fail(f"Required section not found: '{section}'")

    # --------------------------------------------------
    # Validate current Gameweek metrics
    # --------------------------------------------------

    accuracy_text = f"**{accuracy:.1f}%**"

    if accuracy_text not in blog:
        fail(
            "Gameweek accuracy does not match evaluation CSV.\n"
            f"Expected to find: {accuracy_text}"
        )

    correct_text = f"**{correct_outcomes}**"

    actual_results_match = re.search(
        r"## Actual results(.*?)(?:\n## |\Z)",
        blog,
        flags=re.DOTALL,
    )

    if not actual_results_match:
        fail("'Actual results' section not found.")

    actual_results_section = actual_results_match.group(1)

    result_rows = [
        line
        for line in actual_results_section.splitlines()
        if line.startswith("| ")
        and not line.startswith("| Fixture")
        and not line.startswith("|---")
    ]

    if len(result_rows) != matches:
        fail(
            "Actual-results row count does not match evaluation CSV.\n"
            f"Expected: {matches}, found: {len(result_rows)}"
        )

    # --------------------------------------------------
    # Validate draw and 1-1 diagnostics
    # --------------------------------------------------

    draw_scoreline_match = re.search(
        r"## Draw and scoreline behaviour(.*?)(?:\n## |\Z)",
        blog,
        flags=re.DOTALL,
    )

    if not draw_scoreline_match:
        fail("'Draw and scoreline behaviour' section not found.")

    draw_scoreline_section = draw_scoreline_match.group(1)

    expected_draw_text = (
        f"Model 2 predicted **"
        f"{'zero' if predicted_draws == 0 else predicted_draws} draws** "
        f"from the {matches} fixtures."
    )

    if expected_draw_text not in draw_scoreline_section:
        fail(
            "Predicted-draw diagnostic does not match evaluation CSV.\n"
            f"Expected: {expected_draw_text}"
        )

    number_words = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
    }

    draw_count_variants = [
        str(actual_draws),
        number_words.get(actual_draws, str(actual_draws)),
    ]

    draw_count_found = any(
        f"{count} matches actually finished as draws".lower()
        in draw_scoreline_section.lower()
        for count in draw_count_variants
    )

    if not draw_count_found:
        fail(
            "Actual-draw diagnostic does not match evaluation CSV.\n"
            f"Expected draw count: {actual_draws}"
        )

    expected_modal_text = (
        f"All **{modal_one_one}/{matches}** fixtures had 1-1 "
        "as their modal predicted scoreline"
    )

    if expected_modal_text not in draw_scoreline_section:
        fail(
            "1-1 modal-score diagnostic does not match evaluation CSV.\n"
            f"Expected modal 1-1 count: {modal_one_one}/{matches}"
        )

    if actual_one_one == 0:
        expected_actual_one_one_text = "**none** finished 1-1"
    else:
        expected_actual_one_one_text = (
            f"**{actual_one_one}** finished 1-1"
        )

    if expected_actual_one_one_text not in draw_scoreline_section:
        fail(
            "Actual 1-1 diagnostic does not match evaluation CSV.\n"
            f"Expected actual 1-1 count: {actual_one_one}"
        )

    # --------------------------------------------------
    # Validate What we learned
    # --------------------------------------------------

    what_we_learned_match = re.search(
        r"## What we learned(.*?)(?:\n## |\n---|\Z)",
        blog,
        flags=re.DOTALL,
    )

    if not what_we_learned_match:
        fail("'What we learned' section not found.")

    learned_section = what_we_learned_match.group(1)

    if f"GW{gameweek}" not in learned_section:
        fail(
            f"Current Gameweek GW{gameweek} not referenced "
            "in 'What we learned'."
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
        "Four areas are now being monitored",
    ]

    for phrase in stale_phrases:
        if phrase.lower() in blog.lower():
            fail(f"Known stale phrase found: '{phrase}'")

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
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Predicted draws: {predicted_draws}")
    print(f"Actual draws: {actual_draws}")
    print(f"Modal 1-1 predictions: {modal_one_one}")
    print(f"Actual 1-1 results: {actual_one_one}")
    print()
    print("PASS: Gameweek blog matches evaluation data.")


if __name__ == "__main__":
    main()