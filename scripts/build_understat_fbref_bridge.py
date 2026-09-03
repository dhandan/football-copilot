from pathlib import Path

import numpy as np
import pandas as pd
import understatapi

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==================================================
# FILES
# ==================================================

FBREF_FILE = Path(
    "data/processed/"
    "premier_league_matches_xg_enriched.csv"
)

OUTPUT_MATCHES = Path(
    "reports/model5/"
    "understat_fbref_bridge_2024_25.csv"
)

OUTPUT_SUMMARY = Path(
    "reports/model5/"
    "understat_fbref_bridge_summary.csv"
)


# ==================================================
# LOAD FBREF 2024/25
# ==================================================

print()
print("FOOTBALL COPILOT")
print("UNDERSTAT -> FBREF XG BRIDGE")
print("============================")
print()

fbref = pd.read_csv(
    FBREF_FILE
)

fbref["Date"] = pd.to_datetime(
    fbref["Date"],
    errors="coerce",
).dt.normalize()

fbref = fbref[
    fbref["Season"]
    ==
    "2024/25"
].copy()


required_columns = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "HomeXG",
    "AwayXG",
]

missing_columns = [
    column
    for column in required_columns
    if column not in fbref.columns
]

if missing_columns:
    raise ValueError(
        "Missing FBref columns: "
        +
        ", ".join(
            missing_columns
        )
    )


fbref = fbref[
    required_columns
].rename(
    columns={
        "HomeXG":
            "FBrefHomeXG",

        "AwayXG":
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


print(
    "Understat matches:",
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
    rows
)


# ==================================================
# TEAM MAPPING
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

    "Chelsea":
        "Chelsea",

    "Crystal Palace":
        "Crystal Palace",

    "Everton":
        "Everton",

    "Fulham":
        "Fulham",

    "Ipswich":
        "Ipswich",

    "Leicester":
        "Leicester",

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

    "Southampton":
        "Southampton",

    "Tottenham":
        "Tottenham",

    "West Ham":
        "West Ham",

    "Wolverhampton Wanderers":
        "Wolves",
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


unmapped_home = understat_df[
    understat_df[
        "HomeTeam"
    ].isna()
][
    "UnderstatHomeTeam"
].unique()


unmapped_away = understat_df[
    understat_df[
        "AwayTeam"
    ].isna()
][
    "UnderstatAwayTeam"
].unique()


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
    how="inner",
    validate="one_to_one",
)


print(
    "Matched fixtures:",
    len(
        comparison
    ),
)


if len(
    comparison
) != 380:

    raise ValueError(
        "Expected 380 matched fixtures."
    )


# ==================================================
# FIT HOME BRIDGE
# ==================================================

home_model = LinearRegression()

home_model.fit(
    comparison[
        [
            "UnderstatHomeXG"
        ]
    ],
    comparison[
        "FBrefHomeXG"
    ],
)


home_intercept = float(
    home_model.intercept_
)

home_slope = float(
    home_model.coef_[
        0
    ]
)


comparison[
    "MappedHomeXG"
] = (
    home_model.predict(
        comparison[
            [
                "UnderstatHomeXG"
            ]
        ]
    )
)


# ==================================================
# FIT AWAY BRIDGE
# ==================================================

away_model = LinearRegression()

away_model.fit(
    comparison[
        [
            "UnderstatAwayXG"
        ]
    ],
    comparison[
        "FBrefAwayXG"
    ],
)


away_intercept = float(
    away_model.intercept_
)

away_slope = float(
    away_model.coef_[
        0
    ]
)


comparison[
    "MappedAwayXG"
] = (
    away_model.predict(
        comparison[
            [
                "UnderstatAwayXG"
            ]
        ]
    )
)


# ==================================================
# RAW METRICS
# ==================================================

raw_home_mae = mean_absolute_error(
    comparison[
        "FBrefHomeXG"
    ],
    comparison[
        "UnderstatHomeXG"
    ],
)

raw_away_mae = mean_absolute_error(
    comparison[
        "FBrefAwayXG"
    ],
    comparison[
        "UnderstatAwayXG"
    ],
)


raw_home_rmse = np.sqrt(
    mean_squared_error(
        comparison[
            "FBrefHomeXG"
        ],
        comparison[
            "UnderstatHomeXG"
        ],
    )
)

raw_away_rmse = np.sqrt(
    mean_squared_error(
        comparison[
            "FBrefAwayXG"
        ],
        comparison[
            "UnderstatAwayXG"
        ],
    )
)


# ==================================================
# MAPPED METRICS
# ==================================================

mapped_home_mae = mean_absolute_error(
    comparison[
        "FBrefHomeXG"
    ],
    comparison[
        "MappedHomeXG"
    ],
)

mapped_away_mae = mean_absolute_error(
    comparison[
        "FBrefAwayXG"
    ],
    comparison[
        "MappedAwayXG"
    ],
)


mapped_home_rmse = np.sqrt(
    mean_squared_error(
        comparison[
            "FBrefHomeXG"
        ],
        comparison[
            "MappedHomeXG"
        ],
    )
)

mapped_away_rmse = np.sqrt(
    mean_squared_error(
        comparison[
            "FBrefAwayXG"
        ],
        comparison[
            "MappedAwayXG"
        ],
    )
)


home_r2 = r2_score(
    comparison[
        "FBrefHomeXG"
    ],
    comparison[
        "MappedHomeXG"
    ],
)

away_r2 = r2_score(
    comparison[
        "FBrefAwayXG"
    ],
    comparison[
        "MappedAwayXG"
    ],
)


# ==================================================
# COMBINED METRICS
# ==================================================

fbref_all = np.concatenate(
    [
        comparison[
            "FBrefHomeXG"
        ].to_numpy(),

        comparison[
            "FBrefAwayXG"
        ].to_numpy(),
    ]
)


raw_understat_all = np.concatenate(
    [
        comparison[
            "UnderstatHomeXG"
        ].to_numpy(),

        comparison[
            "UnderstatAwayXG"
        ].to_numpy(),
    ]
)


mapped_understat_all = np.concatenate(
    [
        comparison[
            "MappedHomeXG"
        ].to_numpy(),

        comparison[
            "MappedAwayXG"
        ].to_numpy(),
    ]
)


raw_combined_mae = mean_absolute_error(
    fbref_all,
    raw_understat_all,
)

mapped_combined_mae = mean_absolute_error(
    fbref_all,
    mapped_understat_all,
)


raw_combined_rmse = np.sqrt(
    mean_squared_error(
        fbref_all,
        raw_understat_all,
    )
)

mapped_combined_rmse = np.sqrt(
    mean_squared_error(
        fbref_all,
        mapped_understat_all,
    )
)


# ==================================================
# RESIDUALS
# ==================================================

comparison[
    "HomeResidual"
] = (
    comparison[
        "FBrefHomeXG"
    ]
    -
    comparison[
        "MappedHomeXG"
    ]
)

comparison[
    "AwayResidual"
] = (
    comparison[
        "FBrefAwayXG"
    ]
    -
    comparison[
        "MappedAwayXG"
    ]
)


home_mean_residual = (
    comparison[
        "HomeResidual"
    ].mean()
)

away_mean_residual = (
    comparison[
        "AwayResidual"
    ].mean()
)


# ==================================================
# PRINT COEFFICIENTS
# ==================================================

print()
print("BRIDGE EQUATIONS")
print("================")

print()

print(
    "Home:"
)

print(
    "FBrefHomeXG = "
    f"{home_intercept:.6f} "
    f"+ {home_slope:.6f} "
    "* UnderstatHomeXG"
)

print()

print(
    "Away:"
)

print(
    "FBrefAwayXG = "
    f"{away_intercept:.6f} "
    f"+ {away_slope:.6f} "
    "* UnderstatAwayXG"
)


# ==================================================
# PRINT QUALITY
# ==================================================

print()
print("RAW VS MAPPED ERROR")
print("===================")

print()

print(
    "Home raw MAE:",
    f"{raw_home_mae:.4f}"
)

print(
    "Home mapped MAE:",
    f"{mapped_home_mae:.4f}"
)

print(
    "Home raw RMSE:",
    f"{raw_home_rmse:.4f}"
)

print(
    "Home mapped RMSE:",
    f"{mapped_home_rmse:.4f}"
)

print(
    "Home R2:",
    f"{home_r2:.4f}"
)

print()

print(
    "Away raw MAE:",
    f"{raw_away_mae:.4f}"
)

print(
    "Away mapped MAE:",
    f"{mapped_away_mae:.4f}"
)

print(
    "Away raw RMSE:",
    f"{raw_away_rmse:.4f}"
)

print(
    "Away mapped RMSE:",
    f"{mapped_away_rmse:.4f}"
)

print(
    "Away R2:",
    f"{away_r2:.4f}"
)


print()
print("COMBINED")
print("========")

print(
    "Raw combined MAE:",
    f"{raw_combined_mae:.4f}"
)

print(
    "Mapped combined MAE:",
    f"{mapped_combined_mae:.4f}"
)

print(
    "Raw combined RMSE:",
    f"{raw_combined_rmse:.4f}"
)

print(
    "Mapped combined RMSE:",
    f"{mapped_combined_rmse:.4f}"
)

print(
    "FBref mean:",
    f"{fbref_all.mean():.4f}"
)

print(
    "Raw Understat mean:",
    f"{raw_understat_all.mean():.4f}"
)

print(
    "Mapped Understat mean:",
    f"{mapped_understat_all.mean():.4f}"
)


print()
print("MEAN RESIDUAL")
print("=============")

print(
    "Home:",
    f"{home_mean_residual:+.6f}"
)

print(
    "Away:",
    f"{away_mean_residual:+.6f}"
)


# ==================================================
# LARGEST RESIDUALS
# ==================================================

comparison[
    "TotalAbsoluteResidual"
] = (
    comparison[
        "HomeResidual"
    ].abs()
    +
    comparison[
        "AwayResidual"
    ].abs()
)


largest = (
    comparison
    .sort_values(
        "TotalAbsoluteResidual",
        ascending=False,
    )
    .head(
        20
    )
)


print()
print("20 LARGEST POST-MAPPING RESIDUALS")
print("=================================")

print(
    largest[
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FBrefHomeXG",
            "UnderstatHomeXG",
            "MappedHomeXG",
            "FBrefAwayXG",
            "UnderstatAwayXG",
            "MappedAwayXG",
            "TotalAbsoluteResidual",
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
# SUMMARY
# ==================================================

summary = pd.DataFrame(
    [
        {
            "Season":
                "2024/25",

            "Matches":
                len(
                    comparison
                ),

            "HomeIntercept":
                home_intercept,

            "HomeSlope":
                home_slope,

            "AwayIntercept":
                away_intercept,

            "AwaySlope":
                away_slope,

            "RawCombinedMAE":
                raw_combined_mae,

            "MappedCombinedMAE":
                mapped_combined_mae,

            "RawCombinedRMSE":
                raw_combined_rmse,

            "MappedCombinedRMSE":
                mapped_combined_rmse,

            "HomeR2":
                home_r2,

            "AwayR2":
                away_r2,

            "FBrefMean":
                fbref_all.mean(),

            "RawUnderstatMean":
                raw_understat_all.mean(),

            "MappedUnderstatMean":
                mapped_understat_all.mean(),
        }
    ]
)


# ==================================================
# SAVE
# ==================================================

OUTPUT_MATCHES.parent.mkdir(
    parents=True,
    exist_ok=True,
)


comparison.to_csv(
    OUTPUT_MATCHES,
    index=False,
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)


print()
print("FILES SAVED")
print("===========")

print(
    OUTPUT_MATCHES
)

print(
    OUTPUT_SUMMARY
)


print()
print("BRIDGE DIAGNOSTIC COMPLETE")
print("==========================")