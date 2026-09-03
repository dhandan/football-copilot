from pathlib import Path

import numpy as np
import pandas as pd
import understatapi


# ==================================================
# FILES
# ==================================================

FBREF_FILE = Path(
    "data/processed/"
    "premier_league_matches_xg_enriched.csv"
)


# ==================================================
# LOAD FBREF-DERIVED XG
# ==================================================

print()
print("FOOTBALL COPILOT")
print("FBREF VS UNDERSTAT XG COMPARISON")
print("================================")
print()

fbref = pd.read_csv(
    FBREF_FILE
)

fbref["Date"] = pd.to_datetime(
    fbref["Date"],
    errors="coerce",
)

fbref = fbref[
    fbref["Season"]
    ==
    "2024/25"
].copy()


# ==================================================
# IDENTIFY MATCH XG COLUMNS
# ==================================================

required_fbref_columns = [
    "Season",
    "Date",
    "HomeTeam",
    "AwayTeam",
]

for column in required_fbref_columns:
    if column not in fbref.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )


candidate_home_xg = [
    "HomeXG",
    "home_xg",
    "Home_xG",
]

candidate_away_xg = [
    "AwayXG",
    "away_xg",
    "Away_xG",
]


home_xg_column = next(
    (
        column
        for column in candidate_home_xg
        if column in fbref.columns
    ),
    None,
)

away_xg_column = next(
    (
        column
        for column in candidate_away_xg
        if column in fbref.columns
    ),
    None,
)


if home_xg_column is None:
    raise ValueError(
        "Could not find FBref home xG column. "
        f"Available columns: {list(fbref.columns)}"
    )

if away_xg_column is None:
    raise ValueError(
        "Could not find FBref away xG column. "
        f"Available columns: {list(fbref.columns)}"
    )


print(
    "FBref Home xG column:",
    home_xg_column,
)

print(
    "FBref Away xG column:",
    away_xg_column,
)


fbref = fbref[
    [
        "Date",
        "HomeTeam",
        "AwayTeam",
        home_xg_column,
        away_xg_column,
    ]
].rename(
    columns={
        home_xg_column:
            "FBrefHomeXG",

        away_xg_column:
            "FBrefAwayXG",
    }
)


# ==================================================
# LOAD UNDERSTAT 2024/25
# ==================================================

with understatapi.UnderstatClient() as understat:

    league = understat.league(
        league="EPL"
    )

    matches = league.get_match_data(
        season="2024"
    )


print()
print(
    "Understat matches returned:",
    len(matches),
)


understat_rows = []

for match in matches:

    if not match.get(
        "isResult",
        False,
    ):
        continue

    understat_rows.append(
        {
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
    understat_rows
)


# ==================================================
# TEAM NAME NORMALISATION
# ==================================================

TEAM_MAP = {

    "Manchester City":
        "Man City",

    "Manchester United":
        "Man United",

    "Newcastle United":
        "Newcastle",

    "Nottingham Forest":
        "Nott'm Forest",

    "Tottenham":
        "Tottenham",

    "Wolverhampton Wanderers":
        "Wolves",

    "West Ham":
        "West Ham",

    "Leicester":
        "Leicester",

    "Ipswich":
        "Ipswich",

    "Southampton":
        "Southampton",

    "Brighton":
        "Brighton",

    "Bournemouth":
        "Bournemouth",

    "Arsenal":
        "Arsenal",

    "Aston Villa":
        "Aston Villa",

    "Brentford":
        "Brentford",

    "Chelsea":
        "Chelsea",

    "Crystal Palace":
        "Crystal Palace",

    "Everton":
        "Everton",

    "Fulham":
        "Fulham",

    "Liverpool":
        "Liverpool",
}


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
        "Unmapped Understat home teams: "
        f"{unmapped_home}"
    )


if len(
    unmapped_away
) > 0:

    raise ValueError(
        "Unmapped Understat away teams: "
        f"{unmapped_away}"
    )


# ==================================================
# NORMALISE DATES
# ==================================================

fbref[
    "Date"
] = (
    fbref[
        "Date"
    ]
    .dt
    .normalize()
)


# ==================================================
# JOIN
# ==================================================

comparison = fbref.merge(
    understat_df[
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "UnderstatHomeXG",
            "UnderstatAwayXG",
        ]
    ],
    on=[
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


matched = comparison[
    comparison[
        "_merge"
    ]
    ==
    "both"
].copy()


# ==================================================
# DIFFERENCES
# ==================================================

matched[
    "HomeXGDifference"
] = (
    matched[
        "UnderstatHomeXG"
    ]
    -
    matched[
        "FBrefHomeXG"
    ]
)

matched[
    "AwayXGDifference"
] = (
    matched[
        "UnderstatAwayXG"
    ]
    -
    matched[
        "FBrefAwayXG"
    ]
)


matched[
    "HomeXGAbsoluteDifference"
] = (
    matched[
        "HomeXGDifference"
    ]
    .abs()
)

matched[
    "AwayXGAbsoluteDifference"
] = (
    matched[
        "AwayXGDifference"
    ]
    .abs()
)


home_correlation = (
    matched[
        "FBrefHomeXG"
    ]
    .corr(
        matched[
            "UnderstatHomeXG"
        ]
    )
)

away_correlation = (
    matched[
        "FBrefAwayXG"
    ]
    .corr(
        matched[
            "UnderstatAwayXG"
        ]
    )
)


print()
print("XG COMPARISON")
print("=============")

print(
    "Matched fixtures:",
    len(
        matched
    ),
)

print()

print(
    "Home xG correlation:",
    f"{home_correlation:.4f}",
)

print(
    "Away xG correlation:",
    f"{away_correlation:.4f}",
)

print()

print(
    "Mean FBref Home xG:",
    f"{matched['FBrefHomeXG'].mean():.4f}",
)

print(
    "Mean Understat Home xG:",
    f"{matched['UnderstatHomeXG'].mean():.4f}",
)

print()

print(
    "Mean FBref Away xG:",
    f"{matched['FBrefAwayXG'].mean():.4f}",
)

print(
    "Mean Understat Away xG:",
    f"{matched['UnderstatAwayXG'].mean():.4f}",
)

print()

print(
    "Mean absolute Home xG difference:",
    f"{matched['HomeXGAbsoluteDifference'].mean():.4f}",
)

print(
    "Mean absolute Away xG difference:",
    f"{matched['AwayXGAbsoluteDifference'].mean():.4f}",
)

print()

print(
    "Mean signed Home xG difference:",
    f"{matched['HomeXGDifference'].mean():+.4f}",
)

print(
    "Mean signed Away xG difference:",
    f"{matched['AwayXGDifference'].mean():+.4f}",
)


# ==================================================
# COMBINED HOME + AWAY DISTRIBUTION
# ==================================================

fbref_all = np.concatenate(
    [
        matched[
            "FBrefHomeXG"
        ].to_numpy(),

        matched[
            "FBrefAwayXG"
        ].to_numpy(),
    ]
)

understat_all = np.concatenate(
    [
        matched[
            "UnderstatHomeXG"
        ].to_numpy(),

        matched[
            "UnderstatAwayXG"
        ].to_numpy(),
    ]
)


overall_correlation = np.corrcoef(
    fbref_all,
    understat_all,
)[
    0,
    1,
]


overall_mae = np.mean(
    np.abs(
        understat_all
        -
        fbref_all
    )
)


print()
print("COMBINED XG")
print("===========")

print(
    "Correlation:",
    f"{overall_correlation:.4f}",
)

print(
    "Mean absolute difference:",
    f"{overall_mae:.4f}",
)

print(
    "FBref mean:",
    f"{fbref_all.mean():.4f}",
)

print(
    "Understat mean:",
    f"{understat_all.mean():.4f}",
)


# ==================================================
# LARGEST DIFFERENCES
# ==================================================

matched[
    "TotalAbsoluteDifference"
] = (
    matched[
        "HomeXGAbsoluteDifference"
    ]
    +
    matched[
        "AwayXGAbsoluteDifference"
    ]
)


largest = (
    matched
    .sort_values(
        "TotalAbsoluteDifference",
        ascending=False,
    )
    .head(
        20
    )
)


print()
print("20 LARGEST FIXTURE DIFFERENCES")
print("==============================")

print(
    largest[
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FBrefHomeXG",
            "UnderstatHomeXG",
            "FBrefAwayXG",
            "UnderstatAwayXG",
            "TotalAbsoluteDifference",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)


# ==================================================
# VERDICT SUPPORT
# ==================================================

print()
print("SOURCE COMPATIBILITY CHECK")
print("==========================")

if (
    len(
        matched
    )
    ==
    380
    and
    overall_correlation
    >=
    0.90
    and
    overall_mae
    <=
    0.30
):

    print(
        "PASS: Understat is sufficiently "
        "aligned with FBref for further "
        "Model 5 validation."
    )

elif (
    len(
        matched
    )
    ==
    380
    and
    overall_correlation
    >=
    0.80
):

    print(
        "REVIEW: providers are directionally "
        "aligned but differences are material."
    )

else:

    print(
        "FAIL: provider differences are too "
        "large to mix without further work."
    )


print()
print("COMPARISON COMPLETE")
print("===================")