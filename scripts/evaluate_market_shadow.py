from pathlib import Path
import argparse
import math

import pandas as pd


# ==================================================
# PROJECT SETUP
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SHADOW_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "challenger_predictions"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "results"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "challenger_evaluations"
)


# ==================================================
# CONFIG
# ==================================================

CHALLENGER_VERSION = "Model2_OpeningMarket_40_60_v1.0"


# ==================================================
# METRICS
# ==================================================

def actual_probability(row):

    if row["ActualResult"] == "H":
        return row["ShadowHomeProbability"]

    if row["ActualResult"] == "D":
        return row["ShadowDrawProbability"]

    if row["ActualResult"] == "A":
        return row["ShadowAwayProbability"]

    raise RuntimeError(
        f"Invalid ActualResult: {row['ActualResult']}"
    )


def calculate_brier(row):

    actual = {
        "H": 0.0,
        "D": 0.0,
        "A": 0.0,
    }

    actual[row["ActualResult"]] = 1.0

    return (
        (row["ShadowHomeProbability"] - actual["H"]) ** 2
        + (row["ShadowDrawProbability"] - actual["D"]) ** 2
        + (row["ShadowAwayProbability"] - actual["A"]) ** 2
    )


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gameweek",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    print()
    print("FOOTBALL COPILOT")
    print("MARKET SHADOW EVALUATION")
    print("========================")
    print()

    # ----------------------------------------------
    # FILES
    # ----------------------------------------------

    shadow_file = (
        SHADOW_DIR
        / (
            f"2026_27_gw{args.gameweek:02d}_"
            f"market_shadow_predictions.csv"
        )
    )

    results_file = (
        RESULTS_DIR
        / (
            f"2026_27_gw{args.gameweek:02d}_"
            f"results.csv"
        )
    )

    if not shadow_file.exists():
        raise RuntimeError(
            f"Shadow prediction snapshot not found: "
            f"{shadow_file}"
        )

    if not results_file.exists():
        raise RuntimeError(
            f"Results file not found: "
            f"{results_file}"
        )

    # ----------------------------------------------
    # LOAD
    # ----------------------------------------------

    shadow = pd.read_csv(shadow_file)
    results = pd.read_csv(results_file)

    if len(shadow) != 10:
        raise RuntimeError(
            f"Expected 10 shadow predictions, "
            f"found {len(shadow)}."
        )

    if len(results) != 10:
        raise RuntimeError(
            f"Expected 10 results, found {len(results)}."
        )

    if not (
        shadow["ChallengerVersion"]
        == CHALLENGER_VERSION
    ).all():
        raise RuntimeError(
            "Unexpected challenger version."
        )

    if not (
        shadow["SnapshotType"] == "OFFICIAL"
    ).all():
        raise RuntimeError(
            "Refusing to evaluate a non-OFFICIAL "
            "shadow snapshot."
        )

    # ----------------------------------------------
    # RESULTS
    # ----------------------------------------------

    required_result_columns = [
        "FixtureId",
        "ActualHomeGoals",
        "ActualAwayGoals",
        "ActualScore",
        "ActualResult",
        "ActualResultCode",
    ]

    missing = [
        column
        for column in required_result_columns
        if column not in results.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing result columns: {missing}"
        )

    # ----------------------------------------------
    # MERGE
    # ----------------------------------------------

    result_subset = results[
        required_result_columns
    ].copy()

    merged = shadow.merge(
        result_subset,
        on="FixtureId",
        how="left",
        validate="one_to_one",
    )

    if merged["ActualResult"].isna().any():
        raise RuntimeError(
            "Not all shadow predictions matched "
            "to a result."
        )

    # ----------------------------------------------
    # VALIDATE PROBABILITIES
    # ----------------------------------------------

    probability_columns = [
        "ShadowHomeProbability",
        "ShadowDrawProbability",
        "ShadowAwayProbability",
    ]

    for column in probability_columns:
        merged[column] = pd.to_numeric(
            merged[column],
            errors="coerce",
        )

    if merged[probability_columns].isna().any().any():
        raise RuntimeError(
            "Missing shadow probabilities."
        )

    probability_sum = (
        merged["ShadowHomeProbability"]
        + merged["ShadowDrawProbability"]
        + merged["ShadowAwayProbability"]
    )

    if not probability_sum.between(
        0.999999,
        1.000001,
    ).all():
        raise RuntimeError(
            "Shadow probabilities do not sum to 1."
        )

    # ----------------------------------------------
    # EVALUATE
    # ----------------------------------------------

    merged["OutcomeCorrect"] = (
        merged["ShadowPredictedResult"]
        == merged["ActualResult"]
    )

    merged["ActualOutcomeProbability"] = (
        merged.apply(
            actual_probability,
            axis=1,
        )
    )

    merged["LogLoss"] = (
        merged["ActualOutcomeProbability"]
        .apply(
            lambda p:
                -math.log(max(p, 1e-15))
        )
    )

    merged["Brier"] = merged.apply(
        calculate_brier,
        axis=1,
    )

    # ----------------------------------------------
    # SUMMARY
    # ----------------------------------------------

    accuracy = merged["OutcomeCorrect"].mean()

    log_loss = merged["LogLoss"].mean()

    brier = merged["Brier"].mean()

    correct = int(
        merged["OutcomeCorrect"].sum()
    )

    # ----------------------------------------------
    # SAVE
    # ----------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR
        / (
            f"2026_27_gw{args.gameweek:02d}_"
            f"market_shadow_evaluation.csv"
        )
    )

    if output_file.exists():
        raise RuntimeError(
            "Shadow evaluation already exists. "
            "Refusing to overwrite."
        )

    merged.to_csv(
        output_file,
        index=False,
    )

    # ----------------------------------------------
    # DISPLAY
    # ----------------------------------------------

    print(
        merged[
            [
                "HomeTeam",
                "AwayTeam",
                "ShadowPredictedResult",
                "ActualResult",
                "OutcomeCorrect",
                "ActualOutcomeProbability",
                "LogLoss",
                "Brier",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 60)
    print("SHADOW CHALLENGER RESULTS")
    print("=" * 60)

    print(
        f"Correct:   {correct}/10"
    )

    print(
        f"Accuracy:  {accuracy:.1%}"
    )

    print(
        f"Log Loss:  {log_loss:.4f}"
    )

    print(
        f"Brier:     {brier:.4f}"
    )

    print()
    print(
        f"Saved to: {output_file}"
    )