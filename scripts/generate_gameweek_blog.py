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

EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "evaluations"
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
)

args = parser.parse_args()

gameweek = args.gameweek


# ==================================================
# FILES
# ==================================================

prediction_file = (
    PREDICTION_DIR
    /
    f"2026_27_gw{gameweek:02d}_predictions.csv"
)

evaluation_file = (
    EVALUATION_DIR
    /
    f"2026_27_gw{gameweek:02d}_evaluation.csv"
)

baseline_file = (
    EVALUATION_DIR
    /
    f"2026_27_gw{gameweek:02d}_baselines.csv"
)


if not prediction_file.exists():
    raise FileNotFoundError(
        f"Prediction file not found: {prediction_file}"
    )


predictions = pd.read_csv(
    prediction_file
)


evaluation_exists = (
    evaluation_file.exists()
)


if evaluation_exists:

    evaluation = pd.read_csv(
        evaluation_file
    )

else:

    evaluation = None

# ==================================================
# BASELINE EVALUATION
# ==================================================

if baseline_file.exists():

    baselines = pd.read_csv(
        baseline_file
    )

else:

    baselines = None

# ==================================================
# HEADER DATA
# ==================================================

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
    .fillna(False)
    .astype(bool)
    .sum()
)


# ==================================================
# MARKDOWN
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
# PRE-MATCH PREDICTIONS
# ==================================================

lines.append(
    "## Pre-match predictions"
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
# RESULTS / EVALUATION
# ==================================================

if evaluation_exists:

    matches = len(
        evaluation
    )

    correct = int(
        evaluation[
            "OutcomeCorrect"
        ]
        .astype(bool)
        .sum()
    )

    accuracy = (
        correct
        /
        matches
        *
        100
    )

    exact_scores = int(
        evaluation[
            "ExactScoreCorrect"
        ]
        .astype(bool)
        .sum()
    )

    log_loss = float(
        evaluation[
            "LogLoss"
        ]
        .mean()
    )

    brier = float(
        evaluation[
            "Brier"
        ]
        .mean()
    )

    home_goal_mae = float(
        evaluation[
            "HomeGoalAbsoluteError"
        ]
        .mean()
    )

    away_goal_mae = float(
        evaluation[
            "AwayGoalAbsoluteError"
        ]
        .mean()
    )

    total_goal_mae = float(
        evaluation[
            "TotalGoalsAbsoluteError"
        ]
        .mean()
    )


    lines.append(
        "## Actual results"
    )

    lines.append("")

    lines.append(
        "| Fixture | Prediction | "
        "Predicted score | Actual score | Correct? |"
    )

    lines.append(
        "|---|---|---:|---:|---:|"
    )


    for _, row in evaluation.iterrows():

        symbol = (
            "✅"
            if bool(
                row[
                    "OutcomeCorrect"
                ]
            )
            else
            "❌"
        )

        fixture = (
            f"{row['HomeTeam']} v "
            f"{row['AwayTeam']}"
        )

        lines.append(
            f"| {fixture} "
            f"| {row['PredictedResult']} "
            f"| {row['MostLikelyScore']} "
            f"| {row['ActualScore']} "
            f"| {symbol} |"
        )


    lines.append("")


    # ==================================================
    # PERFORMANCE
    # ==================================================

    lines.append(
        "## Gameweek performance"
    )

    lines.append("")

    lines.append(
        "| Metric | GW result | Historical Model 2 |"
    )

    lines.append(
        "|---|---:|---:|"
    )

    lines.append(
        f"| 1X2 Accuracy "
        f"| **{accuracy:.1f}%** "
        f"| 52.69% |"
    )

    lines.append(
        f"| Log Loss "
        f"| **{log_loss:.4f}** "
        f"| 0.9927 |"
    )

    lines.append(
        f"| Brier Score "
        f"| **{brier:.4f}** "
        f"| 0.5927 |"
    )

    lines.append(
        f"| Exact score hits "
        f"| **{exact_scores}/{matches}** "
        f"| - |"
    )

    lines.append("")

    lines.append(
        "### Goal prediction error"
    )

    lines.append("")

    lines.append(
        f"- Home goals MAE: **{home_goal_mae:.3f}**"
    )

    lines.append(
        f"- Away goals MAE: **{away_goal_mae:.3f}**"
    )

    lines.append(
        f"- Total goals MAE: **{total_goal_mae:.3f}**"
    )

    lines.append("")

        # ==================================================
    # BASELINE COMPARISON
    # ==================================================

    if baselines is not None:

        lines.append(
            "## Baseline comparison"
        )

        lines.append("")

        lines.append(
            "To understand whether Model 2 added value "
            "beyond simple football heuristics, GW1 was "
            "also compared with several naive baselines."
        )

        lines.append("")

        lines.append(
            "| Method | Accuracy | Log Loss | Brier Score |"
        )

        lines.append(
            "|---|---:|---:|---:|"
        )


        for _, baseline_row in baselines.iterrows():

            name = (
                baseline_row[
                    "Baseline"
                ]
            )

            accuracy_pct = float(
                baseline_row[
                    "AccuracyPct"
                ]
            )


            if pd.notna(
                baseline_row[
                    "LogLoss"
                ]
            ):

                log_loss_text = (
                    f"{float(baseline_row['LogLoss']):.4f}"
                )

            else:

                log_loss_text = "N/A"


            if pd.notna(
                baseline_row[
                    "Brier"
                ]
            ):

                brier_text = (
                    f"{float(baseline_row['Brier']):.4f}"
                )

            else:

                brier_text = "N/A"


            lines.append(
                f"| {name} "
                f"| {accuracy_pct:.1f}% "
                f"| {log_loss_text} "
                f"| {brier_text} |"
            )


        lines.append("")

        lines.append(
            "In GW1, Model 2 did not outperform the "
            "simple historical baselines."
        )

        lines.append("")

        lines.append(
            "The Always Home and Historical Majority "
            "baselines achieved 70.0% accuracy because "
            "seven of the ten GW1 fixtures were won by "
            "the home team."
        )

        lines.append("")

        lines.append(
            "The Historical Outcome Frequency baseline "
            "also produced a lower Log Loss and Brier "
            "Score than Model 2 in GW1."
        )

        lines.append("")

        lines.append(
            "This should not be interpreted as evidence "
            "that the naive baseline is a better model "
            "after only ten matches.  The comparison will "
            "be tracked cumulatively through the season."
        )

        lines.append("")


    # ==================================================
    # BIGGEST MISS / BEST CALL
    # ==================================================

    worst = (
        evaluation
        .sort_values(
            "ActualOutcomeProbability",
            ascending=True,
        )
        .iloc[0]
    )

    best = (
        evaluation
        .sort_values(
            "ActualOutcomeProbability",
            ascending=False,
        )
        .iloc[0]
    )


    lines.append(
        "## Diagnostics"
    )

    lines.append("")

    lines.append(
        "### Biggest model surprise"
    )

    lines.append("")

    lines.append(
        f"**{worst['HomeTeam']} v "
        f"{worst['AwayTeam']}**"
    )

    lines.append("")

    lines.append(
        f"Actual outcome: "
        f"**{worst['ActualResult']}**"
    )

    lines.append(
        f"Probability assigned to that outcome: "
        f"**{worst['ActualOutcomeProbability'] * 100:.1f}%**"
    )

    lines.append("")

    lines.append(
        "### Highest-confidence success"
    )

    lines.append("")

    lines.append(
        f"**{best['HomeTeam']} v "
        f"{best['AwayTeam']}**"
    )

    lines.append("")

    lines.append(
        f"Actual outcome: "
        f"**{best['ActualResult']}**"
    )

    lines.append(
        f"Probability assigned to that outcome: "
        f"**{best['ActualOutcomeProbability'] * 100:.1f}%**"
    )

    lines.append("")


    # ==================================================
    # COLD START
    # ==================================================

    if (
        "ColdStartUsed"
        in evaluation.columns
    ):

        cold_start = (
            evaluation[
                evaluation[
                    "ColdStartUsed"
                ]
                .astype(str)
                .str.lower()
                .isin(
                    [
                        "true",
                        "1",
                        "yes",
                    ]
                )
            ]
        )

        if not cold_start.empty:

            cold_accuracy = (
                cold_start[
                    "OutcomeCorrect"
                ]
                .astype(bool)
                .mean()
                *
                100
            )

            cold_log_loss = (
                cold_start[
                    "LogLoss"
                ]
                .mean()
            )

            cold_brier = (
                cold_start[
                    "Brier"
                ]
                .mean()
            )


            lines.append(
                "## Promoted-team cold-start performance"
            )

            lines.append("")

            lines.append(
                f"- Fixtures: **{len(cold_start)}**"
            )

            lines.append(
                f"- Accuracy: **{cold_accuracy:.1f}%**"
            )

            lines.append(
                f"- Log Loss: **{cold_log_loss:.4f}**"
            )

            lines.append(
                f"- Brier Score: **{cold_brier:.4f}**"
            )

            lines.append("")


    # ==================================================
    # WHAT WE LEARNED
    # ==================================================

    lines.append(
        "## What we learned"
    )

    lines.append("")

    lines.append(
        "GW1 produced five correct 1X2 outcomes from "
        "ten matches.  This is broadly close to the "
        "historical Model 2 accuracy of 52.69%, but "
        "ten matches are far too few to draw firm "
        "conclusions about model performance."
    )

    lines.append("")

    lines.append(
        "Four areas are now being monitored through "
        "GW5:"
    )

    lines.append("")

    lines.append(
        "1. **xG compression** — the model may be "
        "pulling team-strength estimates too strongly "
        "towards league-average scoring levels."
    )

    lines.append(
        "2. **1-1 modal scoreline concentration** — "
        "eight of ten fixtures had 1-1 as the single "
        "most likely scoreline, while none actually "
        "finished 1-1."
    )

    lines.append(
        "3. **Promoted-team pessimism** — the cold-start "
        "framework underestimated Hull and Ipswich in GW1."
    )

    lines.append(
        "4. **Upset calibration** — lower-probability "
        "results will be tracked to determine whether GW1 "
        "was normal football variance or a systematic bias."
    )

    lines.append("")

    lines.append(
        "5. **Incremental predictive value** — "
        "Model 2 will be compared with simple historical "
        "benchmarks to determine whether its fixture-specific "
        "features consistently improve prediction quality."
    )

    lines.append("")

    lines.append(
        "No changes will be made to Model 2 during "
        "the initial five-Gameweek live monitoring period."
    )

    lines.append("")


# ==================================================
# FOOTER
# ==================================================

lines.append("---")

lines.append("")

lines.append(
    "*Football Copilot is an AI and analytics "
    "learning experiment.  Model probabilities "
    "are statistical estimates, not betting advice.*"
)


# ==================================================
# SAVE
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
    f"Gameweek: {gameweek}"
)

print(
    f"Evaluation included: "
    f"{evaluation_exists}"
)

print()
print(
    f"Saved: {output_file}"
)