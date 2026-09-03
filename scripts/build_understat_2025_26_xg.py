from pathlib import Path

import pandas as pd
import understatapi


# ==================================================
# FILES
# ==================================================

FOOTBALL_COPILOT_FILE = Path(
    "data/processed/"
    "premier_league_matches.csv"
)

OUTPUT_FILE = Path(
    "data/external/understat_xg/"
    "premier_league_2025_26.csv"
)


# ==================================================
# FROZEN PROVIDER BRIDGE
# Derived from 2024/25 only
# ==================================================

HOME_INTERCEPT = 0.124993
HOME_SLOPE = 0.806923

AWAY_INTERCEPT = 0.123716
AWAY_SLOPE = 0.808671


# ==================================================
# TEAM NAME MAPPING
# ==================================================

TEAM_MAP = {

    "Arsenal":
        "Arsenal",

    "Aston Villa":
        "Aston Villa",

    "Bournemouth":
        "Bournemouth",

    "Brentford":
        "Brentford",

    "Brighton":
        "Brighton",

    "Burnley":
        "Burnley",

    "Chelsea":
        "Chelsea",

    "Crystal Palace":
        "Crystal Palace",

    "Everton":
        "Everton",

    "Fulham":
        "Fulham",

    "Leeds":
        "Leeds",

    "Liverpool":
        "Liverpool",

    "Manchester City":
        "Man City",

    "Manchester United":
        "Man United",

    "Newcastle United":
        "Newcastle",

    "Nottingham Forest":
        "Nott'm Forest",

    "Sunderland":
        "Sunderland",

    "Tottenham":
        "Tottenham",

    "West Ham":
        "West Ham",

    "Wolverhampton Wanderers":
        "Wolves",
}


# ==================================================
# LOAD UNDERSTAT 2025/26
# ==================================================

print()
print("FOOTBALL COPILOT")
print("BUILD UNDERSTAT 2025/26 XG")
print("==========================")
print()

with understatapi.UnderstatClient() as understat:

    league = understat.league(
        league="EPL"
    )

    matches = league.get_match_data(
        season="2025"
    )


print(
    "Understat matches returned:",
    len(matches),
)


rows = []

for match in matches:

    if not match.get(
        "isResult",
        False,
    ):
        continue

    rows.append(
        {
            "Season":
                "2025/26",

            "Date":
                pd.to_datetime(
                    match[
                        "datetime"
                    ]
                ).normalize(),

            "UnderstatHomeTeam":
                match[
                    "h"
                ][
                    "title"
                ],

            "UnderstatAwayTeam":
                match[
                    "a"
                ][
                    "title"
                ],

            "HomeGoals":
                int(
                    match[
                        "goals"
                    ][
                        "h"
                    ]
                ),

            "AwayGoals":
                int(
                    match[
                        "goals"
                    ][
                        "a"
                    ]
                ),

            "UnderstatHomeXG":
                float(
                    match[
                        "xG"
                    ][
                        "h"
                    ]
                ),

            "UnderstatAwayXG":
                float(
                    match[
                        "xG"
                    ][
                        "a"
                    ]
                ),
        }
    )


understat_df = pd.DataFrame(
    rows
)


# ==================================================
# NORMALISE TEAM NAMES
# ==================================================

understat_df[
    "HomeTeam"
] = (
    understat_df[
        "UnderstatHomeTeam"
    ]
    .map(
        TEAM_MAP
    )
)

understat_df[
    "AwayTeam"
] = (
    understat_df[
        "UnderstatAwayTeam"
    ]
    .map(
        TEAM_MAP
    )
)


unmapped_home = (
    understat_df[
        understat_df[
            "HomeTeam"
        ].isna()
    ][
        "UnderstatHomeTeam"
    ]
    .unique()
)

unmapped_away = (
    understat_df[
        understat_df[
            "AwayTeam"
        ].isna()
    ][
        "UnderstatAwayTeam"
    ]
    .unique()
)


if len(
    unmapped_home
) > 0:

    raise ValueError(
        "Unmapped home teams: "
        f"{unmapped_home}"
    )


if len(
    unmapped_away
) > 0:

    raise ValueError(
        "Unmapped away teams: "
        f"{unmapped_away}"
    )


# ==================================================
# APPLY FROZEN BRIDGE
# ==================================================

understat_df[
    "BridgedHomeXG"
] = (
    HOME_INTERCEPT
    +
    HOME_SLOPE
    *
    understat_df[
        "UnderstatHomeXG"
    ]
)


understat_df[
    "BridgedAwayXG"
] = (
    AWAY_INTERCEPT
    +
    AWAY_SLOPE
    *
    understat_df[
        "UnderstatAwayXG"
    ]
)


# ==================================================
# LOAD FOOTBALL COPILOT 2025/26
# ==================================================

football = pd.read_csv(
    FOOTBALL_COPILOT_FILE
)

football[
    "Date"
] = pd.to_datetime(
    football[
        "Date"
    ],
    errors="coerce",
).dt.normalize()


football = football[
    football[
        "Season"
    ]
    ==
    "2025/26"
].copy()


football = football[
    [
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
    ]
].drop_duplicates()


print()
print(
    "Football Copilot fixtures:",
    len(
        football
    ),
)


# ==================================================
# RECONCILE FIXTURES
# ==================================================

comparison = football.merge(
    understat_df[
        [
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",
        ]
    ],
    on=[
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
    ],
    how="outer",
    indicator=True,
)


print()
print("JOIN RESULT")
print("===========")

print(
    comparison[
        "_merge"
    ]
    .value_counts()
    .to_string()
)


left_only = comparison[
    comparison[
        "_merge"
    ]
    ==
    "left_only"
]

right_only = comparison[
    comparison[
        "_merge"
    ]
    ==
    "right_only"
]


if not left_only.empty:

    print()
    print("MISSING FROM UNDERSTAT")
    print("======================")

    print(
        left_only[
            [
                "Season",
                "Date",
                "HomeTeam",
                "AwayTeam",
            ]
        ].to_string(
            index=False
        )
    )


if not right_only.empty:

    print()
    print("UNEXPECTED UNDERSTAT FIXTURES")
    print("=============================")

    print(
        right_only[
            [
                "Season",
                "Date",
                "HomeTeam",
                "AwayTeam",
            ]
        ].to_string(
            index=False
        )
    )


matched = int(
    (
        comparison[
            "_merge"
        ]
        ==
        "both"
    ).sum()
)


# ==================================================
# SANITY CHECK DISTRIBUTIONS
# ==================================================

print()
print("XG DISTRIBUTION")
print("===============")

print()

print(
    "Mean raw Home xG:",
    f"{understat_df['UnderstatHomeXG'].mean():.4f}"
)

print(
    "Mean bridged Home xG:",
    f"{understat_df['BridgedHomeXG'].mean():.4f}"
)

print()

print(
    "Mean raw Away xG:",
    f"{understat_df['UnderstatAwayXG'].mean():.4f}"
)

print(
    "Mean bridged Away xG:",
    f"{understat_df['BridgedAwayXG'].mean():.4f}"
)

print()

print(
    "Mean raw combined xG:",
    f"{pd.concat([understat_df['UnderstatHomeXG'], understat_df['UnderstatAwayXG']]).mean():.4f}"
)

print(
    "Mean bridged combined xG:",
    f"{pd.concat([understat_df['BridgedHomeXG'], understat_df['BridgedAwayXG']]).mean():.4f}"
)


# ==================================================
# FINAL OUTPUT
# ==================================================

output = understat_df[
    [
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "HomeGoals",
        "AwayGoals",
        "UnderstatHomeXG",
        "UnderstatAwayXG",
        "BridgedHomeXG",
        "BridgedAwayXG",
    ]
].copy()


output = output.sort_values(
    [
        "Date",
        "HomeTeam",
        "AwayTeam",
    ]
).reset_index(
    drop=True
)


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


output.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ==================================================
# SUMMARY
# ==================================================

print()
print("VALIDATION SUMMARY")
print("==================")

print(
    "Understat completed fixtures:",
    len(
        output
    ),
)

print(
    "Football Copilot fixtures:",
    len(
        football
    ),
)

print(
    "Matched fixtures:",
    matched,
)

print(
    "Missing from Understat:",
    len(
        left_only
    ),
)

print(
    "Unexpected Understat fixtures:",
    len(
        right_only
    ),
)

print()
print("BRIDGE COEFFICIENTS")
print("===================")

print(
    "Home:",
    f"{HOME_INTERCEPT:.6f} + "
    f"{HOME_SLOPE:.6f} × Understat"
)

print(
    "Away:",
    f"{AWAY_INTERCEPT:.6f} + "
    f"{AWAY_SLOPE:.6f} × Understat"
)

print()

if (
    len(
        output
    )
    ==
    380
    and
    len(
        football
    )
    ==
    380
    and
    matched
    ==
    380
    and
    len(
        left_only
    )
    ==
    0
    and
    len(
        right_only
    )
    ==
    0
):

    print(
        "PASS: 2025/26 Understat xG "
        "reconciles perfectly with "
        "Football Copilot."
    )

else:

    print(
        "FAIL: fixture reconciliation "
        "requires investigation."
    )


print()
print(
    "Saved:",
    OUTPUT_FILE
)

print()
print("BUILD COMPLETE")
print("==============")