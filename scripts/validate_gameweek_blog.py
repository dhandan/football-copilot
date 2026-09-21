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


def extract_section(blog: str, heading: str) -> str:
    match = re.search(
        rf"^{re.escape(heading)}\s*$\n(.*?)(?=^## |\Z)",
        blog,
        flags=re.DOTALL | re.MULTILINE,
    )

    if not match:
        fail(f"Required section not found: '{heading}'")

    return match.group(1)


def markdown_table_rows(section: str) -> list[str]:
    rows = []

    for line in section.splitlines():
        stripped = line.strip()

        if not stripped.startswith("|"):
            continue

        # Ignore Markdown separator rows such as:
        # | --- | ---: | :--- |
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]

        if cells and all(
            re.fullmatch(r":?-{3,}:?", cell) is not None
            for cell in cells
        ):
            continue

        rows.append(stripped)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Football Copilot Gameweek blog output."
    )
    parser.add_argument(
        "--gameweek",
        type=int,
        required=True,
        help="Gameweek number to validate, e.g. 5",
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
    # Expected values from authoritative evaluation CSV
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
    # Required structure
    # --------------------------------------------------

    required_sections = [
        "## Pre-match predictions",
        "## Actual results",
        "## Gameweek performance",
        "## Diagnostics",
        "## Draw and scoreline behaviour",
        "## What we learned",
    ]

    for section in required_sections:
        if section not in blog:
            fail(f"Required section not found: '{section}'")

    # --------------------------------------------------
    # Actual-results table
    # --------------------------------------------------

    actual_results_section = extract_section(
        blog,
        "## Actual results",
    )

    rows = markdown_table_rows(actual_results_section)

    if not rows:
        fail("Actual-results table not found.")

    # First non-separator Markdown row is the header.
    result_rows = rows[1:]

    if len(result_rows) != matches:
        fail(
            "Actual-results row count does not match evaluation CSV.\n"
            f"Expected: {matches}, found: {len(result_rows)}"
        )

    # --------------------------------------------------
    # Current Gameweek performance
    # --------------------------------------------------

    gameweek_performance_section = extract_section(
        blog,
        "## Gameweek performance",
    )

    expected_accuracy = f"{accuracy:.1f}%"

    if expected_accuracy not in gameweek_performance_section:
        fail(
            "Gameweek accuracy does not match evaluation CSV.\n"
            f"Expected: {expected_accuracy}"
        )

    # --------------------------------------------------
    # Draw / scoreline diagnostics
    # --------------------------------------------------

    draw_scoreline_section = extract_section(
        blog,
        "## Draw and scoreline behaviour",
    )

    diagnostic_patterns = {
        "Predicted 1X2 draws": (
            predicted_draws,
            r"Predicted 1X2 draws:\**\s*(\d+)",
        ),
        "Actual draws": (
            actual_draws,
            r"Actual draws:\**\s*(\d+)",
        ),
        "Modal 1-1 predictions": (
            modal_one_one,
            r"Modal 1-1 predictions:\**\s*(\d+)",
        ),
        "Actual 1-1 results": (
            actual_one_one,
            r"Actual 1-1 results:\**\s*(\d+)",
        ),
    }

    for label, (expected, pattern) in diagnostic_patterns.items():
        match = re.search(
            pattern,
            draw_scoreline_section,
            flags=re.IGNORECASE,
        )

        if not match:
            fail(
                f"{label} diagnostic not found in "
                "'Draw and scoreline behaviour'."
            )

        found = int(match.group(1))

        if found != expected:
            fail(
                f"{label} diagnostic does not match evaluation CSV.\n"
                f"Expected: {expected}, found: {found}"
            )

    # --------------------------------------------------
    # What we learned
    # --------------------------------------------------

    what_we_learned_section = extract_section(
        blog,
        "## What we learned",
    )

    if f"GW{gameweek}" not in what_we_learned_section:
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