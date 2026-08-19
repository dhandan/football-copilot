from pathlib import Path
import argparse

import pandas as pd


# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "predictions"
)

BLOG_DIR = (
    PROJECT_ROOT
    / "docs"
    / "gameweeks"
)


# ==================================================
# ARGUMENTS
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--gameweek",
    type=int,
    required=True,
    help="Premier League gameweek number.",
)

args = parser.parse_args()

gameweek = args.gameweek


# ==================================================
# FIND OFFICIAL SNAPSHOT
# ==================================================

prediction_file = (
    PREDICTION_DIR
    /
    f"2026_27_gw{gameweek:02d}_predictions.csv"
)


if not prediction_file.exists():

    raise FileNotFoundError(
        f"Official prediction snapshot not found: "
        f"{prediction_file}"
    )


# ==================================================
# LOAD DATA
# ==================================================

predictions = pd.read_csv(
    prediction_file
)


if predictions.empty:

    raise ValueError(
        "Prediction snapshot is empty."
    )


season = str(
    predictions.iloc[0][
        "Season"
    ]
)


timestamp = pd.to_datetime(
    predictions.iloc[0][
        "PredictionTimestamp"
    ]
)


formatted_timestamp = (
    timestamp.strftime(
        "%d %B %Y, %H:%M"
    )
)


model_version = str(
    predictions.iloc[0][
        "ModelVersion"
    ]
)


cold_start_count = int(
    predictions[
        "ColdStartUsed"
    ]
    .fillna(
        False
    )
    .astype(
        bool
    )
    .sum()
)


# ==================================================
# CONFIDENCE
# ==================================================

predictions[
    "MaximumProbability"
] = predictions[
    [
        "HomeWinProbability",
        "DrawProbability",
        "AwayWinProbability",
    ]
].max(
    axis=1
)


confidence_rows = (
    predictions
    .sort_values(
        "MaximumProbability",
        ascending=False,
    )
    .head(
        3
    )
)


# ==================================================
# BUILD MARKDOWN
# ==================================================

lines = []


lines.append(
    f"# Football Copilot {season}"
)

lines.append(
    f"## Gameweek {gameweek}"
)

lines.append("")


lines.append(
    f"**Prediction snapshot:** "
    f"{formatted_timestamp} UK time  "
)

lines.append(
    f"**Model:** {model_version}  "
)

lines.append(
    f"**Fixtures predicted:** "
    f"{len(predictions)}  "
)

lines.append(
    f"**Cold-start fixtures:** "
    f"{cold_start_count}"
)

lines.append("")


# ==================================================
# GAMEWEEK TABLE
# ==================================================

lines.append(
    "## Gameweek predictions"
)

lines.append("")


lines.append(
    "| Fixture | Predicted outcome | "
    "Most likely score | Home | Draw | Away |"
)

lines.append(
    "|---|---|---:|---:|---:|---:|"
)


for _, row in predictions.iterrows():

    fixture = (
        f"{row['HomeTeam']} v "
        f"{row['AwayTeam']}"
    )


    lines.append(
        f"| {fixture} "
        f"| **{row['PredictedResult']}** "
        f"| {row['MostLikelyScore']} "
        f"| {row['HomeWinProbability']:.1f}% "
        f"| {row['DrawProbability']:.1f}% "
        f"| {row['AwayWinProbability']:.1f}% |"
    )


lines.append("")


# ==================================================
# HIGHEST CONFIDENCE
# ==================================================

lines.append(
    "## Highest-confidence predictions"
)

lines.append("")


for _, row in confidence_rows.iterrows():

    lines.append(
        f"### {row['HomeTeam']} "
        f"v {row['AwayTeam']}"
    )

    lines.append("")

    lines.append(
        f"**Predicted outcome: "
        f"{row['PredictedResult']}**"
    )

    lines.append("")

    lines.append(
        f"{row['HomeTeam']} win: "
        f"{row['HomeWinProbability']:.1f}%  "
    )

    lines.append(
        f"Draw: "
        f"{row['DrawProbability']:.1f}%  "
    )

    lines.append(
        f"{row['AwayTeam']} win: "
        f"{row['AwayWinProbability']:.1f}%"
    )

    lines.append("")

    lines.append(
        f"Expected goals: "
        f"**{row['HomeTeam']} "
        f"{row['ExpectedHomeGoals']:.2f} "
        f"- "
        f"{row['ExpectedAwayGoals']:.2f} "
        f"{row['AwayTeam']}**"
    )

    lines.append("")

    lines.append(
        f"Most likely individual scoreline: "
        f"**{row['MostLikelyScore']}**"
    )

    lines.append("")


# ==================================================
# COLD START
# ==================================================

if cold_start_count > 0:

    lines.append(
        "## Promoted-team cold start"
    )

    lines.append("")

    lines.append(
        "Some promoted teams do not yet have "
        "sufficient current Premier League history "
        "for the standard feature pipeline."
    )

    lines.append("")

    lines.append(
        "Football Copilot therefore uses a "
        "validated promoted-team cold-start prior."
    )

    lines.append("")

    lines.append(
        "The current methodology combines:"
    )

    lines.append("")

    lines.append(
        "- **75% translated Championship performance**"
    )

    lines.append(
        "- **25% historical Premier League baseline**"
    )

    lines.append("")

    lines.append(
        "The weighting was selected using a "
        "historical promoted-team backtest."
    )

    lines.append("")


# ==================================================
# MODELLING NOTE
# ==================================================

lines.append(
    "## A modelling distinction worth noting"
)

lines.append("")

lines.append(
    "The predicted outcome and the most likely "
    "individual scoreline do not always agree."
)

lines.append("")

lines.append(
    "The outcome probability is the combined "
    "probability of every scoreline producing that "
    "result, while the most likely scoreline refers "
    "to one individual score combination."
)

lines.append("")


# ==================================================
# NEXT STEP
# ==================================================

lines.append(
    "## What happens next"
)

lines.append("")

lines.append(
    "These predictions were frozen before the "
    "gameweek and will not be retrospectively changed."
)

lines.append("")

lines.append(
    "After the fixtures are completed, Football "
    "Copilot will evaluate:"
)

lines.append("")

lines.append(
    "- 1X2 prediction accuracy"
)

lines.append(
    "- Log Loss"
)

lines.append(
    "- Brier score"
)

lines.append(
    "- exact-score accuracy"
)

lines.append(
    "- expected versus actual goals"
)

lines.append(
    "- cold-start performance"
)

lines.append(
    "- cumulative 2026/27 model performance"
)

lines.append("")

lines.append("---")

lines.append("")

lines.append(
    "*Football Copilot is an AI and analytics "
    "learning experiment.  Model probabilities "
    "are statistical estimates, not betting advice.*"
)


# ==================================================
# SAVE BLOG
# ==================================================

BLOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


output_file = (
    BLOG_DIR
    /
    f"GW{gameweek:02d}.md"
)


output_file.write_text(
    "\n".join(
        lines
    ),
    encoding="utf-8",
)


print()
print("FOOTBALL COPILOT")
print("GAMEWEEK BLOG GENERATED")
print("=======================")

print()
print(
    f"Gameweek: "
    f"{gameweek}"
)

print(
    f"Predictions: "
    f"{len(predictions)}"
)

print(
    f"Cold starts: "
    f"{cold_start_count}"
)

print()
print(
    f"Saved: "
    f"{output_file}"
)