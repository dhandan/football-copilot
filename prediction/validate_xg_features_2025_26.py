from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# FILE
# ==================================================

INPUT_FILE = Path(
    "data/processed/"
    "premier_league_2025_26_xg_features.csv"
)


# ==================================================
# LOAD
# ==================================================

print()
print("FOOTBALL COPILOT")
print("VALIDATE 2025/26 XG FEATURES")
print("============================")
print()

df = pd.read_csv(
    INPUT_FILE
)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce",
)

df = (
    df
    .sort_values(
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
        ]
    )
    .reset_index(
        drop=True
    )
)


if len(df) != 380:
    raise ValueError(
        f"Expected 380 matches, found {len(df)}."
    )


# ==================================================
# RECONSTRUCT TEAM-PERSPECTIVE MATCH HISTORY
# ==================================================

home_history = pd.DataFrame(
    {
        "Date":
            df["Date"],

        "Team":
            df["HomeTeam"],

        "Opponent":
            df["AwayTeam"],

        "Venue":
            "Home",

        "XGFor":
            df["BridgedHomeXG"],

        "XGAgainst":
            df["BridgedAwayXG"],
    }
)


away_history = pd.DataFrame(
    {
        "Date":
            df["Date"],

        "Team":
            df["AwayTeam"],

        "Opponent":
            df["HomeTeam"],

        "Venue":
            "Away",

        "XGFor":
            df["BridgedAwayXG"],

        "XGAgainst":
            df["BridgedHomeXG"],
    }
)


history = pd.concat(
    [
        home_history,
        away_history,
    ],
    ignore_index=True,
)


history = (
    history
    .sort_values(
        [
            "Team",
            "Date",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ==================================================
# INDEPENDENT EXPECTED VALUE
# ==================================================

def prior_mean(
    team,
    current_date,
    metric,
    window,
):
    prior = history[
        (
            history["Team"]
            ==
            team
        )
        &
        (
            history["Date"]
            <
            current_date
        )
    ].sort_values(
        "Date"
    )

    if prior.empty:
        return np.nan

    return (
        prior[
            metric
        ]
        .tail(
            window
        )
        .mean()
    )


# ==================================================
# SELECT SAMPLE
#
# Exclude first several weeks so sampled fixtures
# have prior history. Fixed random_state makes the
# validation reproducible.
# ==================================================

eligible = df[
    df["Date"]
    >=
    pd.Timestamp(
        "2025-10-01"
    )
].copy()


sample = eligible.sample(
    n=20,
    random_state=42,
).sort_values(
    "Date"
)


# ==================================================
# CHECK DEFINITIONS
# ==================================================

checks = [
    (
        "Home",
        "XGFor",
        5,
        "HomeXGForAvg5",
    ),
    (
        "Home",
        "XGAgainst",
        5,
        "HomeXGAgainstAvg5",
    ),
    (
        "Away",
        "XGFor",
        5,
        "AwayXGForAvg5",
    ),
    (
        "Away",
        "XGAgainst",
        5,
        "AwayXGAgainstAvg5",
    ),
    (
        "Home",
        "XGFor",
        10,
        "HomeXGForAvg10",
    ),
    (
        "Home",
        "XGAgainst",
        10,
        "HomeXGAgainstAvg10",
    ),
    (
        "Away",
        "XGFor",
        10,
        "AwayXGForAvg10",
    ),
    (
        "Away",
        "XGAgainst",
        10,
        "AwayXGAgainstAvg10",
    ),
]


# ==================================================
# VALIDATE
# ==================================================

results = []


for _, match in sample.iterrows():

    for (
        side,
        metric,
        window,
        feature,
    ) in checks:

        if side == "Home":
            team = match["HomeTeam"]
        else:
            team = match["AwayTeam"]

        expected = prior_mean(
            team=team,
            current_date=match["Date"],
            metric=metric,
            window=window,
        )

        stored = match[
            feature
        ]

        if (
            pd.isna(expected)
            and
            pd.isna(stored)
        ):
            passed = True

        elif (
            pd.isna(expected)
            or
            pd.isna(stored)
        ):
            passed = False

        else:
            passed = bool(
                np.isclose(
                    expected,
                    stored,
                    rtol=1e-10,
                    atol=1e-10,
                )
            )

        results.append(
            {
                "Date":
                    match["Date"],

                "HomeTeam":
                    match["HomeTeam"],

                "AwayTeam":
                    match["AwayTeam"],

                "Team":
                    team,

                "Feature":
                    feature,

                "Expected":
                    expected,

                "Stored":
                    stored,

                "Passed":
                    passed,
            }
        )


results = pd.DataFrame(
    results
)


# ==================================================
# SUMMARY
# ==================================================

total_checks = len(
    results
)

passed_checks = int(
    results[
        "Passed"
    ].sum()
)

failed_checks = (
    total_checks
    -
    passed_checks
)


print(
    "Fixtures sampled:",
    len(sample)
)

print(
    "Checks per fixture:",
    len(checks)
)

print()

print("VALIDATION SUMMARY")
print("==================")

print(
    "Checks:",
    total_checks
)

print(
    "Passed:",
    passed_checks
)

print(
    "Failed:",
    failed_checks
)


# ==================================================
# FAILURES
# ==================================================

failures = results[
    ~results[
        "Passed"
    ]
]


if not failures.empty:

    print()
    print("FAILED CHECKS")
    print("=============")

    print(
        failures[
            [
                "Date",
                "HomeTeam",
                "AwayTeam",
                "Team",
                "Feature",
                "Expected",
                "Stored",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.8f}"
            ),
        )
    )


# ==================================================
# FINAL VERDICT
# ==================================================

print()
print("LEAKAGE VERDICT")
print("===============")

if failed_checks == 0:

    print(
        "PASS: all sampled rolling xG "
        "features use strictly prior "
        "2025/26 matches only."
    )

    print(
        "No sampled evidence of "
        "current-match leakage."
    )

else:

    print(
        "FAIL: rolling feature "
        "construction requires "
        "investigation."
    )


print()
print("VALIDATION COMPLETE")
print("===================")