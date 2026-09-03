from pathlib import Path

import pandas as pd


# ==================================================
# FILES
# ==================================================

MODEL2_FILE = Path(
    "data/processed/"
    "prediction_features_v2.csv"
)

MODEL5_XG_FILE = Path(
    "data/processed/"
    "premier_league_2025_26_xg_features_with_history.csv"
)


# ==================================================
# LOAD MODEL 2
# ==================================================

print()
print("FOOTBALL COPILOT")
print("MODEL 5 2025/26 COHORT VALIDATION")
print("=================================")
print()

model2 = pd.read_csv(
    MODEL2_FILE
)

model2["Date"] = pd.to_datetime(
    model2["Date"],
    format="%Y-%m-%d",
    errors="raise",
)

model2 = model2[
    model2["Season"] == "2025/26"
].copy()


model2_cohort = (
    model2[
        [
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",
        ]
    ]
    .drop_duplicates()
)


# ==================================================
# LOAD MODEL 5 XG FEATURES
# ==================================================

model5 = pd.read_csv(
    MODEL5_XG_FILE
)

model5["Date"] = pd.to_datetime(
    model5["Date"],
    format="%Y-%m-%d",
    errors="raise",
)

model5 = model5[
    model5["Season"] == "2025/26"
].copy()


model5_cohort = (
    model5[
        [
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",
        ]
    ]
    .drop_duplicates()
)


# ==================================================
# BASIC COUNTS
# ==================================================

print("SOURCE COUNTS")
print("=============")

print(
    "Model 2 fixtures:",
    len(model2_cohort),
)

print(
    "Model 5 xG fixtures:",
    len(model5_cohort),
)


# ==================================================
# RECONCILE
# ==================================================

comparison = model5_cohort.merge(
    model2_cohort,
    on=[
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
    ],
    how="outer",
    indicator=True,
)


counts = (
    comparison["_merge"]
    .value_counts()
)


both = int(
    counts.get(
        "both",
        0,
    )
)

model5_only = int(
    counts.get(
        "left_only",
        0,
    )
)

model2_only = int(
    counts.get(
        "right_only",
        0,
    )
)


print()
print("COHORT RECONCILIATION")
print("=====================")

print(
    "Both:",
    both,
)

print(
    "Model 5 xG only:",
    model5_only,
)

print(
    "Model 2 only:",
    model2_only,
)


# ==================================================
# MODEL 5 FIXTURES EXCLUDED BY MODEL 2
# ==================================================

excluded = comparison[
    comparison["_merge"]
    ==
    "left_only"
].copy()


if not excluded.empty:

    excluded = (
        excluded
        .sort_values(
            [
                "Date",
                "HomeTeam",
                "AwayTeam",
            ]
        )
    )

    print()
    print("FIXTURES EXCLUDED BY MODEL 2")
    print("============================")

    print(
        excluded[
            [
                "Date",
                "HomeTeam",
                "AwayTeam",
            ]
        ]
        .to_string(
            index=False
        )
    )


# ==================================================
# EXCLUSION COUNTS BY TEAM
# ==================================================

if not excluded.empty:

    home_exclusions = (
        excluded[
            "HomeTeam"
        ]
        .value_counts()
    )

    away_exclusions = (
        excluded[
            "AwayTeam"
        ]
        .value_counts()
    )

    teams = sorted(
        set(
            home_exclusions.index
        )
        |
        set(
            away_exclusions.index
        )
    )


    exclusion_rows = []

    for team in teams:

        home_count = int(
            home_exclusions.get(
                team,
                0,
            )
        )

        away_count = int(
            away_exclusions.get(
                team,
                0,
            )
        )

        exclusion_rows.append(
            {
                "Team":
                    team,

                "HomeExclusions":
                    home_count,

                "AwayExclusions":
                    away_count,

                "TotalExclusions":
                    home_count
                    +
                    away_count,
            }
        )


    exclusion_summary = pd.DataFrame(
        exclusion_rows
    ).sort_values(
        "TotalExclusions",
        ascending=False,
    )


    print()
    print("EXCLUSIONS BY TEAM")
    print("==================")

    print(
        exclusion_summary
        .to_string(
            index=False
        )
    )


# ==================================================
# CHECK XG COMPLETENESS ON LOCKED COHORT
# ==================================================

locked = model5.merge(
    model2_cohort,
    on=[
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
    ],
    how="inner",
    validate="one_to_one",
)


MODEL5_XG_FEATURES = [
    "HomeXGForAvg5",
    "HomeXGAgainstAvg5",
    "AwayXGForAvg5",
    "AwayXGAgainstAvg5",
    "HomeXGForAvg10",
    "HomeXGAgainstAvg10",
    "AwayXGForAvg10",
    "AwayXGAgainstAvg10",
    "HomeXGDifferenceAvg5",
    "AwayXGDifferenceAvg5",
    "HomeXGForTrend",
    "AwayXGForTrend",
    "HomeXGAgainstTrend",
    "AwayXGAgainstTrend",
]


print()
print("LOCKED COHORT XG COMPLETENESS")
print("=============================")

missing_total = 0

for feature in MODEL5_XG_FEATURES:

    if feature not in locked.columns:
        raise ValueError(
            f"Missing Model 5 feature: {feature}"
        )

    missing = int(
        locked[
            feature
        ].isna().sum()
    )

    missing_total += missing

    print(
        f"{feature:<30} "
        f"missing={missing:>3}"
    )


# ==================================================
# FINAL VERDICT
# ==================================================

print()
print("LOCKED COHORT")
print("=============")

print(
    "2025/26 evaluation fixtures:",
    len(locked),
)

print(
    "Total missing xG feature values:",
    missing_total,
)


print()
print("COHORT VERDICT")
print("==============")

if (
    len(locked) > 0
    and
    model2_only == 0
    and
    missing_total == 0
):

    print(
        "PASS: common Model 2 / Model 5 "
        "2025/26 evaluation cohort is "
        "ready to lock."
    )

else:

    print(
        "REVIEW: cohort requires "
        "investigation before evaluation."
    )


print()
print("VALIDATION COMPLETE")
print("===================")