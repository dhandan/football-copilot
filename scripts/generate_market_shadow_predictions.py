from pathlib import Path
from datetime import datetime
import argparse

import pandas as pd


# ==================================================
# PROJECT SETUP
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "predictions"
)

MARKET_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "market"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "challenger_predictions"
)


# ==================================================
# FROZEN CHALLENGER CONFIG
# ==================================================

CHALLENGER_VERSION = "Model2_OpeningMarket_40_60_v1.0"

MODEL_VERSION = "Model2_v1.0"

MODEL_WEIGHT = 0.40
MARKET_WEIGHT = 0.60


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

    parser.add_argument(
        "--test",
        action="store_true",
        help="Allow use of a TEST market snapshot.",
    )

    args = parser.parse_args()

    print()
    print("FOOTBALL COPILOT")
    print("MARKET SHADOW CHALLENGER")
    print("========================")
    print()

    # ----------------------------------------------
    # OFFICIAL MODEL 2 SNAPSHOT
    # ----------------------------------------------

    prediction_file = (
        PREDICTION_DIR
        / f"2026_27_gw{args.gameweek:02d}_predictions.csv"
    )

    if not prediction_file.exists():
        raise RuntimeError(
            "Official Model 2 prediction snapshot does not exist: "
            f"{prediction_file}"
        )

    model = pd.read_csv(prediction_file)

    if len(model) != 10:
        raise RuntimeError(
            f"Expected 10 Model 2 predictions, found {len(model)}."
        )

    if not (model["ModelVersion"] == MODEL_VERSION).all():
        raise RuntimeError(
            f"Expected ModelVersion {MODEL_VERSION}."
        )

    if not (model["PredictionStatus"] == "Available").all():
        raise RuntimeError(
            "Not all Model 2 predictions are available."
        )

    # ----------------------------------------------
    # MARKET SNAPSHOT
    # ----------------------------------------------

    snapshot_type = "test" if args.test else "official"

    market_files = sorted(
        MARKET_DIR.glob(
            f"2026_27_gw{args.gameweek:02d}_"
            f"opening_market_{snapshot_type}_*.csv"
        )
    )

    if not market_files:
        raise RuntimeError(
            f"No {snapshot_type.upper()} market snapshot "
            f"found for GW{args.gameweek}."
        )

    if len(market_files) > 1:
        raise RuntimeError(
            f"Multiple {snapshot_type.upper()} market snapshots "
            f"found for GW{args.gameweek}. "
            "Refusing to choose one automatically."
        )

    market_file = market_files[0]

    market = pd.read_csv(market_file)

    if len(market) != 10:
        raise RuntimeError(
            f"Expected 10 market fixtures, found {len(market)}."
        )

    # ----------------------------------------------
    # JOIN
    # ----------------------------------------------

    model_columns = [
        "FixtureId",
        "FixtureDate",
        "FixtureTime",
        "HomeTeam",
        "AwayTeam",
        "ModelVersion",
        "HomeWinProbability",
        "DrawProbability",
        "AwayWinProbability",
        "ColdStartUsed",
        "ColdStartTeams",
        "ColdStartMethod",
    ]

    market_columns = [
        "FixtureDate",
        "FixtureTime",
        "HomeTeam",
        "AwayTeam",
        "SnapshotTimestamp",
        "MarketSource",
        "BookmakerCount",
        "AvgH",
        "AvgD",
        "AvgA",
        "MarketHomeProbability",
        "MarketDrawProbability",
        "MarketAwayProbability",
    ]

    merged = model[model_columns].merge(
        market[market_columns],
        on=[
            "FixtureDate",
            "FixtureTime",
            "HomeTeam",
            "AwayTeam",
        ],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    unmatched = merged[
        merged["_merge"] != "both"
    ]

    if not unmatched.empty:
        print()
        print("UNMATCHED FIXTURES")
        print(
            unmatched[
                [
                    "FixtureDate",
                    "FixtureTime",
                    "HomeTeam",
                    "AwayTeam",
                    "_merge",
                ]
            ].to_string(index=False)
        )

        raise RuntimeError(
            "Model 2 and market fixtures do not match exactly."
        )

    merged = merged.drop(columns="_merge")

    # ----------------------------------------------
    # MODEL PROBABILITIES
    #
    # Model 2 stores percentages.
    # Market snapshot stores decimals.
    # ----------------------------------------------

    model_h = merged["HomeWinProbability"] / 100.0
    model_d = merged["DrawProbability"] / 100.0
    model_a = merged["AwayWinProbability"] / 100.0

    model_total = model_h + model_d + model_a

    model_h = model_h / model_total
    model_d = model_d / model_total
    model_a = model_a / model_total

    # ----------------------------------------------
    # FROZEN 40 / 60 BLEND
    # ----------------------------------------------

    merged["ShadowHomeProbability"] = (
        MODEL_WEIGHT * model_h
        + MARKET_WEIGHT * merged["MarketHomeProbability"]
    )

    merged["ShadowDrawProbability"] = (
        MODEL_WEIGHT * model_d
        + MARKET_WEIGHT * merged["MarketDrawProbability"]
    )

    merged["ShadowAwayProbability"] = (
        MODEL_WEIGHT * model_a
        + MARKET_WEIGHT * merged["MarketAwayProbability"]
    )

    shadow_total = (
        merged["ShadowHomeProbability"]
        + merged["ShadowDrawProbability"]
        + merged["ShadowAwayProbability"]
    )

    if not shadow_total.between(
        0.999999,
        1.000001,
    ).all():
        raise RuntimeError(
            "Shadow probabilities do not sum to 1."
        )

    # ----------------------------------------------
    # SHADOW PICK
    # ----------------------------------------------

    def shadow_pick(row):

        probabilities = {
            "H": row["ShadowHomeProbability"],
            "D": row["ShadowDrawProbability"],
            "A": row["ShadowAwayProbability"],
        }

        return max(
            probabilities,
            key=probabilities.get,
        )

    merged["ShadowPredictedResult"] = merged.apply(
        shadow_pick,
        axis=1,
    )

    # ----------------------------------------------
    # METADATA
    # ----------------------------------------------

    merged.insert(
        0,
        "PredictionTimestamp",
        datetime.now()
        .astimezone()
        .isoformat(timespec="seconds"),
    )

    merged.insert(
        1,
        "Season",
        "2026/27",
    )

    merged.insert(
        2,
        "Gameweek",
        args.gameweek,
    )

    merged.insert(
        3,
        "ChallengerVersion",
        CHALLENGER_VERSION,
    )

    merged.insert(
        4,
        "ModelWeight",
        MODEL_WEIGHT,
    )

    merged.insert(
        5,
        "MarketWeight",
        MARKET_WEIGHT,
    )

    merged.insert(
        6,
        "SnapshotType",
        snapshot_type.upper(),
    )

    # ----------------------------------------------
    # DISPLAY
    # ----------------------------------------------

    print(
        f"Model snapshot:  {prediction_file.name}"
    )

    print(
        f"Market snapshot: {market_file.name}"
    )

    print(
        f"Mode:            {snapshot_type.upper()}"
    )

    print()

    display = merged[
        [
            "HomeTeam",
            "AwayTeam",
            "HomeWinProbability",
            "DrawProbability",
            "AwayWinProbability",
            "MarketHomeProbability",
            "MarketDrawProbability",
            "MarketAwayProbability",
            "ShadowHomeProbability",
            "ShadowDrawProbability",
            "ShadowAwayProbability",
            "ShadowPredictedResult",
        ]
    ].copy()

    print(
        display.to_string(index=False)
    )

    # ----------------------------------------------
    # SAVE
    # ----------------------------------------------

    if args.test:

        print()
        print(
            "TEST MODE - shadow prediction file "
            "will NOT be saved."
        )

        raise SystemExit

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR
        / (
            f"2026_27_gw{args.gameweek:02d}_"
            f"market_shadow_predictions.csv"
        )
    )

    if output_file.exists():
        raise RuntimeError(
            "OFFICIAL SHADOW SNAPSHOT ALREADY EXISTS. "
            "Refusing to overwrite."
        )

    merged.to_csv(
        output_file,
        index=False,
    )

    print()
    print(
        f"OFFICIAL SHADOW SNAPSHOT SAVED: "
        f"{output_file}"
    )

    print(
        "DO NOT MODIFY OR OVERWRITE THIS FILE."
    )