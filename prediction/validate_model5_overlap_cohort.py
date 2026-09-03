from pathlib import Path

import pandas as pd


# ==================================================
# FILES
# ==================================================

CANONICAL_FILE = Path(
    "data/processed/"
    "premier_league_matches_xg_enriched.csv"
)

MODEL2_FILE = Path(
    "data/processed/"
    "prediction_features_v2.csv"
)


# ==================================================
# LOCKED MODEL 5 TEST SEASONS
# ==================================================

TEST_SEASONS = [
    "2023/24",
    "2024/25",
]


# ==================================================
# LOAD DATA
# ==================================================

print()
print("FOOTBALL COPILOT")
print("MODEL 5 OVERLAP COHORT VALIDATION")
print("=================================")
print()

canonical = pd.read_csv(
    CANONICAL_FILE
)

model2 = pd.read_csv(
    MODEL2_FILE
)

canonical["Date"] = pd.to_datetime(
    canonical["Date"],
    errors="coerce",
)

model2["Date"] = pd.to_datetime(
    model2["Date"],
    errors="coerce",
)


# ==================================================
# FILTER TO MODEL 5 TEST WINDOW
# ==================================================

canonical = canonical[
    canonical["Season"].isin(
        TEST_SEASONS
    )
].copy()

model2 = model2[
    model2["Season"].isin(
        TEST_SEASONS
    )
].copy()


# ==================================================
# FIXTURE KEYS
# ==================================================

KEYS = [
    "Season",
    "Date",
    "HomeTeam",
    "AwayTeam",
]


canonical_keys = (
    canonical[
        KEYS
    ]
    .drop_duplicates()
)

model2_keys = (
    model2[
        KEYS
    ]
    .drop_duplicates()
)


# ==================================================
# COMPARE COHORTS
# ==================================================

comparison = canonical_keys.merge(
    model2_keys,
    on=KEYS,
    how="outer",
    indicator=True,
)


print("COHORT COUNTS")
print("=============")

print()
print(
    comparison["_merge"]
    .value_counts()
    .to_string()
)


# ==================================================
# BY SEASON
# ==================================================

print()
print("BY SEASON")
print("=========")

for season in TEST_SEASONS:

    season_comparison = comparison[
        comparison["Season"]
        ==
        season
    ]

    total = len(
        canonical_keys[
            canonical_keys["Season"]
            ==
            season
        ]
    )

    matched = int(
        (
            season_comparison["_merge"]
            ==
            "both"
        ).sum()
    )

    missing = int(
        (
            season_comparison["_merge"]
            ==
            "left_only"
        ).sum()
    )

    unexpected = int(
        (
            season_comparison["_merge"]
            ==
            "right_only"
        ).sum()
    )

    print(
        f"{season}: "
        f"canonical={total}, "
        f"in_model2={matched}, "
        f"missing={missing}, "
        f"unexpected={unexpected}"
    )


# ==================================================
# MISSING FROM MODEL 2
# ==================================================

missing = comparison[
    comparison["_merge"]
    ==
    "left_only"
].copy()

missing = missing.sort_values(
    [
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
    ]
)


print()
print("MISSING FROM prediction_features_v2.csv")
print("=======================================")

if missing.empty:

    print("None.")

else:

    print(
        missing[
            KEYS
        ].to_string(
            index=False
        )
    )


# ==================================================
# MISSING BY TEAM
# ==================================================

if not missing.empty:

    home_missing = (
        missing[
            "HomeTeam"
        ]
        .value_counts()
        .rename(
            "HomeMissing"
        )
    )

    away_missing = (
        missing[
            "AwayTeam"
        ]
        .value_counts()
        .rename(
            "AwayMissing"
        )
    )

    team_summary = (
        pd.concat(
            [
                home_missing,
                away_missing,
            ],
            axis=1,
        )
        .fillna(0)
    )

    team_summary[
        "TotalMissingAppearances"
    ] = (
        team_summary[
            "HomeMissing"
        ]
        +
        team_summary[
            "AwayMissing"
        ]
    )

    team_summary = (
        team_summary
        .sort_values(
            "TotalMissingAppearances",
            ascending=False,
        )
    )

    print()
    print("MISSING FIXTURES BY TEAM")
    print("========================")

    print(
        team_summary.to_string()
    )


# ==================================================
# CHECK POSITION WITHIN SEASON
# ==================================================

if not missing.empty:

    canonical_sorted = (
        canonical
        .sort_values(
            [
                "Season",
                "Date",
                "HomeTeam",
                "AwayTeam",
            ]
        )
        .copy()
    )

    canonical_sorted[
        "SeasonMatchNumber"
    ] = (
        canonical_sorted
        .groupby(
            "Season"
        )
        .cumcount()
        +
        1
    )

    missing_position = missing.merge(
        canonical_sorted[
            KEYS
            +
            [
                "SeasonMatchNumber"
            ]
        ],
        on=KEYS,
        how="left",
    )

    print()
    print("MISSING FIXTURE POSITION WITHIN SEASON")
    print("======================================")

    print(
        missing_position[
            KEYS
            +
            [
                "SeasonMatchNumber"
            ]
        ].to_string(
            index=False
        )
    )


# ==================================================
# VALIDATION SUMMARY
# ==================================================

both_count = int(
    (
        comparison["_merge"]
        ==
        "both"
    ).sum()
)

missing_count = int(
    (
        comparison["_merge"]
        ==
        "left_only"
    ).sum()
)

unexpected_count = int(
    (
        comparison["_merge"]
        ==
        "right_only"
    ).sum()
)


print()
print("VALIDATION SUMMARY")
print("==================")

print(
    f"Canonical fixtures: "
    f"{len(canonical_keys)}"
)

print(
    f"Model 2 fixtures: "
    f"{len(model2_keys)}"
)

print(
    f"Matched fixtures: "
    f"{both_count}"
)

print(
    f"Missing from Model 2: "
    f"{missing_count}"
)

print(
    f"Unexpected Model 2 fixtures: "
    f"{unexpected_count}"
)

print()
print("VALIDATION COMPLETE")
print("===================")