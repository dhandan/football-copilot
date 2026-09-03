from pathlib import Path

import numpy as np
import pandas as pd


FILE = Path(
    "data/processed/premier_league_matches_xg_enriched.csv"
)


def check_close(actual, expected, label, tolerance=1e-10):
    if pd.isna(actual) and pd.isna(expected):
        print(f"PASS  {label}")
        return True

    if pd.isna(actual) or pd.isna(expected):
        print(
            f"FAIL  {label}: "
            f"actual={actual}, expected={expected}"
        )
        return False

    if np.isclose(
        actual,
        expected,
        atol=tolerance,
        rtol=0,
    ):
        print(f"PASS  {label}")
        return True

    print(
        f"FAIL  {label}: "
        f"actual={actual:.6f}, "
        f"expected={expected:.6f}"
    )
    return False


def build_team_history(df):
    home = pd.DataFrame({
        "Date": df["Date"],
        "Team": df["HomeTeam"],
        "XGFor": df["HomeXG"],
        "XGAgainst": df["AwayXG"],
    })

    away = pd.DataFrame({
        "Date": df["Date"],
        "Team": df["AwayTeam"],
        "XGFor": df["AwayXG"],
        "XGAgainst": df["HomeXG"],
    })

    history = pd.concat(
        [home, away],
        ignore_index=True,
    )

    history["Date"] = pd.to_datetime(
        history["Date"]
    )

    return history.sort_values(
        ["Team", "Date"]
    ).reset_index(drop=True)


def main():
    print()
    print("FOOTBALL COPILOT")
    print("XG FEATURE LEAKAGE VALIDATION")
    print("=============================")
    print()

    df = pd.read_csv(FILE)

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    history = build_team_history(df)

    # Use fixtures where both teams have substantial
    # prior history so 5 and 10-match windows can be
    # independently reconstructed.

    candidates = df[
        df["Date"] >= "2023-01-01"
    ].copy()

    sample = candidates.sample(
        n=min(20, len(candidates)),
        random_state=42,
    )

    passes = 0
    checks = 0

    for _, match in sample.iterrows():

        date = match["Date"]
        home_team = match["HomeTeam"]
        away_team = match["AwayTeam"]

        print()
        print(
            f"{date.date()} | "
            f"{home_team} vs {away_team}"
        )

        home_prior = history[
            (history["Team"] == home_team)
            &
            (history["Date"] < date)
        ].sort_values("Date")

        away_prior = history[
            (history["Team"] == away_team)
            &
            (history["Date"] < date)
        ].sort_values("Date")

        tests = [
            (
                match["HomeXGForAvg5"],
                home_prior.tail(5)["XGFor"].mean(),
                "Home XG For Avg 5",
            ),
            (
                match["HomeXGAgainstAvg5"],
                home_prior.tail(5)["XGAgainst"].mean(),
                "Home XG Against Avg 5",
            ),
            (
                match["AwayXGForAvg5"],
                away_prior.tail(5)["XGFor"].mean(),
                "Away XG For Avg 5",
            ),
            (
                match["AwayXGAgainstAvg5"],
                away_prior.tail(5)["XGAgainst"].mean(),
                "Away XG Against Avg 5",
            ),
            (
                match["HomeXGForAvg10"],
                home_prior.tail(10)["XGFor"].mean(),
                "Home XG For Avg 10",
            ),
            (
                match["AwayXGForAvg10"],
                away_prior.tail(10)["XGFor"].mean(),
                "Away XG For Avg 10",
            ),
        ]

        for actual, expected, label in tests:
            checks += 1

            if check_close(
                actual,
                expected,
                label,
            ):
                passes += 1

    print()
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(f"Checks: {checks}")
    print(f"Passed: {passes}")
    print(f"Failed: {checks - passes}")

    if passes == checks:
        print()
        print(
            "PASS: rolling xG features use "
            "prior matches only."
        )
        print(
            "No sampled evidence of "
            "current-match leakage."
        )
    else:
        print()
        print(
            "FAIL: feature values do not match "
            "independently reconstructed "
            "pre-match history."
        )

    print()


if __name__ == "__main__":
    main()