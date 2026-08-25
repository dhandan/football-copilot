from pathlib import Path
import argparse
import math

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

HISTORICAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "premier_league_matches.csv"
)

EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "evaluations"
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

evaluation_file = (
    EVALUATION_DIR
    /
    f"2026_27_gw{gameweek:02d}_evaluation.csv"
)

output_file = (
    EVALUATION_DIR
    /
    f"2026_27_gw{gameweek:02d}_baselines.csv"
)


if not HISTORICAL_FILE.exists():
    raise FileNotFoundError(
        f"Historical file not found: {HISTORICAL_FILE}"
    )


if not evaluation_file.exists():
    raise FileNotFoundError(
        f"Evaluation file not found: {evaluation_file}"
    )


# ==================================================
# LOAD DATA
# ==================================================

history = pd.read_csv(
    HISTORICAL_FILE
)

evaluation = pd.read_csv(
    evaluation_file
)


# ==================================================
# VALIDATE HISTORICAL RESULTS
# ==================================================

history = history[
    history["FTR"].isin(
        [
            "H",
            "D",
            "A",
        ]
    )
].copy()


if history.empty:
    raise ValueError(
        "No valid historical H/D/A results found."
    )


# ==================================================
# HISTORICAL OUTCOME FREQUENCIES
# ==================================================

historical_counts = (
    history[
        "FTR"
    ]
    .value_counts()
)


historical_total = len(
    history
)


home_probability = (
    historical_counts.get(
        "H",
        0,
    )
    /
    historical_total
)


draw_probability = (
    historical_counts.get(
        "D",
        0,
    )
    /
    historical_total
)


away_probability = (
    historical_counts.get(
        "A",
        0,
    )
    /
    historical_total
)


frequency_probabilities = {
    "H": home_probability,
    "D": draw_probability,
    "A": away_probability,
}


majority_class = max(
    frequency_probabilities,
    key=frequency_probabilities.get,
)


# ==================================================
# NORMALISE ACTUAL RESULTS
# ==================================================

def actual_result_code(
    row,
):

    actual = str(
        row[
            "ActualResult"
        ]
    ).strip()


    if actual == str(
        row[
            "HomeTeam"
        ]
    ).strip():

        return "H"


    if actual == str(
        row[
            "AwayTeam"
        ]
    ).strip():

        return "A"


    if actual.lower() == "draw":

        return "D"


    raise ValueError(
        "Could not determine actual result for "
        f"{row['HomeTeam']} vs {row['AwayTeam']}: "
        f"{actual}"
    )


evaluation[
    "ActualCode"
] = evaluation.apply(
    actual_result_code,
    axis=1,
)


# ==================================================
# METRIC HELPERS
# ==================================================

def accuracy(
    predictions,
    actuals,
):

    correct = sum(
        predicted == actual
        for predicted, actual
        in zip(
            predictions,
            actuals,
        )
    )

    return (
        correct
        /
        len(
            actuals
        )
    )


def log_loss(
    probability_rows,
    actuals,
):

    losses = []

    for probabilities, actual in zip(
        probability_rows,
        actuals,
    ):

        probability = probabilities[
            actual
        ]

        probability = max(
            probability,
            1e-15,
        )

        losses.append(
            -math.log(
                probability
            )
        )

    return sum(
        losses
    ) / len(
        losses
    )


def brier_score(
    probability_rows,
    actuals,
):

    scores = []

    for probabilities, actual in zip(
        probability_rows,
        actuals,
    ):

        score = 0.0

        for outcome in [
            "H",
            "D",
            "A",
        ]:

            observed = (
                1.0
                if outcome == actual
                else 0.0
            )

            score += (
                probabilities[
                    outcome
                ]
                -
                observed
            ) ** 2

        scores.append(
            score
        )

    return sum(
        scores
    ) / len(
        scores
    )


# ==================================================
# ACTUAL RESULTS
# ==================================================

actuals = (
    evaluation[
        "ActualCode"
    ]
    .tolist()
)


# ==================================================
# MODEL 2
# ==================================================

model_predictions = []

model_probabilities = []


for _, row in evaluation.iterrows():

    probabilities = {
        "H":
            float(
                row[
                    "HomeWinProbability"
                ]
            )
            / 100.0,

        "D":
            float(
                row[
                    "DrawProbability"
                ]
            )
            / 100.0,

        "A":
            float(
                row[
                    "AwayWinProbability"
                ]
            )
            / 100.0,
    }


    model_probabilities.append(
        probabilities
    )


    model_predictions.append(
        max(
            probabilities,
            key=probabilities.get,
        )
    )


model_accuracy = accuracy(
    model_predictions,
    actuals,
)


model_log_loss = log_loss(
    model_probabilities,
    actuals,
)


model_brier = brier_score(
    model_probabilities,
    actuals,
)


# ==================================================
# ALWAYS HOME
# ==================================================

always_home_predictions = [
    "H"
    for _ in actuals
]


always_home_accuracy = accuracy(
    always_home_predictions,
    actuals,
)


# ==================================================
# HISTORICAL MAJORITY CLASS
# ==================================================

majority_predictions = [
    majority_class
    for _ in actuals
]


majority_accuracy = accuracy(
    majority_predictions,
    actuals,
)


# ==================================================
# HISTORICAL FREQUENCY BASELINE
# ==================================================

frequency_rows = [
    frequency_probabilities.copy()
    for _ in actuals
]


frequency_predictions = [
    majority_class
    for _ in actuals
]


frequency_accuracy = accuracy(
    frequency_predictions,
    actuals,
)


frequency_log_loss = log_loss(
    frequency_rows,
    actuals,
)


frequency_brier = brier_score(
    frequency_rows,
    actuals,
)


# ==================================================
# RESULTS
# ==================================================

results = pd.DataFrame(
    [
        {
            "Gameweek":
                gameweek,

            "Baseline":
                "Model 2",

            "Accuracy":
                model_accuracy,

            "AccuracyPct":
                model_accuracy
                * 100,

            "LogLoss":
                model_log_loss,

            "Brier":
                model_brier,
        },

        {
            "Gameweek":
                gameweek,

            "Baseline":
                "Always Home",

            "Accuracy":
                always_home_accuracy,

            "AccuracyPct":
                always_home_accuracy
                * 100,

            "LogLoss":
                None,

            "Brier":
                None,
        },

        {
            "Gameweek":
                gameweek,

            "Baseline":
                "Historical Majority Class",

            "Accuracy":
                majority_accuracy,

            "AccuracyPct":
                majority_accuracy
                * 100,

            "LogLoss":
                None,

            "Brier":
                None,
        },

        {
            "Gameweek":
                gameweek,

            "Baseline":
                "Historical Outcome Frequency",

            "Accuracy":
                frequency_accuracy,

            "AccuracyPct":
                frequency_accuracy
                * 100,

            "LogLoss":
                frequency_log_loss,

            "Brier":
                frequency_brier,
        },
    ]
)


# ==================================================
# SAVE
# ==================================================

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


results.to_csv(
    output_file,
    index=False,
)


# ==================================================
# OUTPUT
# ==================================================

print()
print("FOOTBALL COPILOT")
print(
    f"2026/27 GAMEWEEK {gameweek} BASELINE EVALUATION"
)
print("=" * 65)

print()
print("HISTORICAL PREMIER LEAGUE SAMPLE")
print("=" * 65)

print(
    f"Matches: {historical_total}"
)

print()

print(
    f"Home wins: "
    f"{home_probability * 100:.2f}%"
)

print(
    f"Draws:     "
    f"{draw_probability * 100:.2f}%"
)

print(
    f"Away wins: "
    f"{away_probability * 100:.2f}%"
)

print()

print(
    "Historical majority class:",
    {
        "H": "Home",
        "D": "Draw",
        "A": "Away",
    }[
        majority_class
    ],
)


print()
print("=" * 65)
print("GAMEWEEK BASELINE COMPARISON")
print("=" * 65)
print()


for _, row in results.iterrows():

    print(
        row[
            "Baseline"
        ]
    )

    print(
        f"  Accuracy: "
        f"{row['AccuracyPct']:.1f}%"
    )

    if pd.notna(
        row[
            "LogLoss"
        ]
    ):

        print(
            f"  Log Loss: "
            f"{row['LogLoss']:.4f}"
        )

    else:

        print(
            "  Log Loss: N/A"
        )


    if pd.notna(
        row[
            "Brier"
        ]
    ):

        print(
            f"  Brier:    "
            f"{row['Brier']:.4f}"
        )

    else:

        print(
            "  Brier:    N/A"
        )

    print()


print("=" * 65)

print(
    f"Saved to: {output_file}"
)