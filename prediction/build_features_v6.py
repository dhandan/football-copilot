from pathlib import Path
import re

import numpy as np
import pandas as pd


# ==================================================
# FILES
# ==================================================

MATCH_FILE = "data/processed/matches_clean.csv"

M2_FEATURE_FILE = (
    "data/processed/prediction_features_v2.csv"
)

FBREF_FILE = (
    "data/external/fbref_xg/final_matches.csv"
)

UNDERSTAT_FILE = (
    "data/external/understat_xg/"
    "premier_league_2025_26.csv"
)

OUTPUT_FILE = (
    "data/processed/prediction_features_v6.csv"
)

RAW_FILES = {
    "2021/22": "data/raw/E0_2122.csv",
    "2022/23": "data/raw/E0_2223.csv",
    "2023/24": "data/raw/E0_2324.csv",
    "2024/25": "data/raw/E0_2425.csv",
    "2025/26": "data/raw/E0_2526.csv",
}


# ==================================================
# EXACT MODEL 2 FEATURES
# ==================================================

MODEL2_FEATURES = [
    "HomeRecentGoalsFor",
    "HomeRecentGoalsAgainst",
    "HomeRecentPPG",
    "AwayRecentGoalsFor",
    "AwayRecentGoalsAgainst",
    "AwayRecentPPG",
    "Home10GoalsFor",
    "Home10GoalsAgainst",
    "Home10PPG",
    "Away10GoalsFor",
    "Away10GoalsAgainst",
    "Away10PPG",
    "HomeVenuePPG",
    "HomeVenueGoalsFor",
    "AwayVenuePPG",
    "AwayVenueGoalsFor",
    "HomeSeasonPPG",
    "HomeSeasonGoalDifferencePG",
    "AwaySeasonPPG",
    "AwaySeasonGoalDifferencePG",
    "RecentPPGDifference",
    "TenMatchPPGDifference",
    "SeasonPPGDifference",
    "AttackVsDefenceHome",
    "AttackVsDefenceAway",
]


# ==================================================
# HELPERS
# ==================================================

def fail(message):
    raise RuntimeError(message)


def normalise_fbref_name(value):
    """
    Convert FBref team/opponent naming to the
    Football Copilot naming convention.
    """

    mapping = {
        "Brighton And Hove Albion": "Brighton",
        "Brighton": "Brighton",

        "Ipswich Town": "Ipswich",
        "Ipswich": "Ipswich",

        "Leeds United": "Leeds",
        "Leeds": "Leeds",

        "Leicester City": "Leicester",
        "Leicester": "Leicester",

        "Luton Town": "Luton",
        "Luton": "Luton",

        "Manchester City": "Man City",
        "Manchester Utd": "Man United",
        "Manchester United": "Man United",

        "Newcastle United": "Newcastle",
        "Newcastle Utd": "Newcastle",

        "Norwich City": "Norwich",
        "Norwich": "Norwich",

        "Nottingham Forest": "Nott'm Forest",
        "Nott'ham Forest": "Nott'm Forest",

        "Sheffield United": "Sheffield United",
        "Sheffield Utd": "Sheffield United",

        "Tottenham Hotspur": "Tottenham",
        "Tottenham": "Tottenham",

        "West Ham United": "West Ham",
        "West Ham": "West Ham",

        "Wolverhampton Wanderers": "Wolves",
        "Wolves": "Wolves",
    }

    return mapping.get(
        value,
        value,
    )


def parse_matchweek(value):

    if pd.isna(value):
        return np.nan

    result = re.search(
        r"(\d+)",
        str(value),
    )

    if result is None:
        return np.nan

    return int(
        result.group(1)
    )


def no_vig_probabilities(values):
    """
    Convert decimal odds into implied probabilities
    and remove the bookmaker margin.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    if (
        np.any(~np.isfinite(values))
        or
        np.any(values <= 0)
    ):
        return np.full(
            len(values),
            np.nan,
        )

    implied = 1.0 / values

    total = implied.sum()

    if total <= 0:
        return np.full(
            len(values),
            np.nan,
        )

    return implied / total


def safe_divide(
    numerator,
    denominator,
):

    if (
        pd.isna(numerator)
        or
        pd.isna(denominator)
        or
        denominator == 0
    ):
        return np.nan

    return numerator / denominator


def weighted_average(
    values,
    decay=0.88,
):
    """
    Exponential weighting.

    Input is ordered oldest -> newest, therefore
    the most recent observation receives the
    greatest weight.
    """

    values = pd.Series(
        values,
        dtype=float,
    ).dropna()

    if len(values) == 0:
        return np.nan

    n = len(values)

    weights = np.array(
        [
            decay ** (
                n - 1 - index
            )
            for index in range(n)
        ],
        dtype=float,
    )

    return np.average(
        values.to_numpy(),
        weights=weights,
    )


# ==================================================
# LOAD CORE DATA
# ==================================================

print()
print("FOOTBALL COPILOT")
print("BUILDING MODEL 6 FEATURES")
print("=========================")


matches = pd.read_csv(
    MATCH_FILE
)

m2 = pd.read_csv(
    M2_FEATURE_FILE
)

fbref = pd.read_csv(
    FBREF_FILE
)

understat = pd.read_csv(
    UNDERSTAT_FILE
)


matches["Date"] = pd.to_datetime(
    matches["Date"],
    errors="coerce",
).dt.normalize()


m2["Date"] = pd.to_datetime(
    m2["Date"],
    errors="coerce",
).dt.normalize()


fbref["date"] = pd.to_datetime(
    fbref["date"],
    errors="coerce",
).dt.normalize()


understat["Date"] = pd.to_datetime(
    understat["Date"],
    errors="coerce",
).dt.normalize()


if matches["Date"].isna().any():
    fail(
        "Invalid dates in matches_clean.csv"
    )


if m2["Date"].isna().any():
    fail(
        "Invalid dates in Model 2 features"
    )


if fbref["date"].isna().any():
    fail(
        "Invalid dates in FBref data"
    )


if understat["Date"].isna().any():
    fail(
        "Invalid dates in Understat data"
    )


if len(matches) != 1900:
    fail(
        f"Expected 1900 matches, found "
        f"{len(matches)}"
    )


# ==================================================
# VALIDATE MODEL 2 FEATURES
# ==================================================

missing_m2 = [
    column
    for column in MODEL2_FEATURES
    if column not in m2.columns
]


if missing_m2:

    fail(
        "Missing Model 2 features: "
        +
        ", ".join(missing_m2)
    )


print()
print(
    f"Model 2 feature rows: {len(m2)}"
)


# ==================================================
# PREPARE FBREF
# ==================================================

print()
print("PREPARING FBREF")
print("===============")


fbref["Team"] = (
    fbref["team"]
    .apply(normalise_fbref_name)
)


fbref["Opponent"] = (
    fbref["opponent"]
    .apply(normalise_fbref_name)
)


fbref["Matchweek"] = (
    fbref["round"]
    .apply(parse_matchweek)
)


# ==================================================
# BUILD FBREF XG HISTORY
#
# IMPORTANT:
#
# The numeric FBref season field is NOT used.
#
# The local FBref file contains:
#     2020/21 -> 2024/25
#
# We match the required 2021/22 -> 2024/25
# fixtures using:
#
#     Date
#     Team
#     Opponent
#     Venue
#
# ==================================================

print()
print("BUILDING FBREF XG HISTORY")
print("=========================")


target_fbref_matches = matches[
    matches["Season"].isin(
        [
            "2021/22",
            "2022/23",
            "2023/24",
            "2024/25",
        ]
    )
].copy()


if len(target_fbref_matches) != 1520:

    fail(
        "Expected 1520 target fixtures for "
        "2021/22 to 2024/25"
    )


fbref_rows = []

unmatched_fbref = []


for _, match in target_fbref_matches.iterrows():

    date = match["Date"]

    home = match["HomeTeam"]

    away = match["AwayTeam"]


    home_candidates = fbref[
        (fbref["date"] == date)
        &
        (fbref["Team"] == home)
        &
        (fbref["Opponent"] == away)
        &
        (fbref["venue"] == "Home")
    ]


    away_candidates = fbref[
        (fbref["date"] == date)
        &
        (fbref["Team"] == away)
        &
        (fbref["Opponent"] == home)
        &
        (fbref["venue"] == "Away")
    ]


    if (
        len(home_candidates) != 1
        or
        len(away_candidates) != 1
    ):

        unmatched_fbref.append(
            {
                "Season":
                    match["Season"],

                "Date":
                    date,

                "HomeTeam":
                    home,

                "AwayTeam":
                    away,

                "HomeMatches":
                    len(home_candidates),

                "AwayMatches":
                    len(away_candidates),
            }
        )

        continue


    home_row = (
        home_candidates.iloc[0]
    )

    away_row = (
        away_candidates.iloc[0]
    )


    if (
        home_row["Matchweek"]
        !=
        away_row["Matchweek"]
    ):

        fail(
            "FBref matchweek disagreement for "
            f"{home} vs {away} on "
            f"{date.date()}"
        )


    fbref_rows.append(
        {
            "Season":
                match["Season"],

            "Date":
                date,

            "HomeTeam":
                home,

            "AwayTeam":
                away,

            "HomeGoals":
                match["FTHG"],

            "AwayGoals":
                match["FTAG"],

            "HomeXG":
                float(
                    home_row["xg"]
                ),

            "AwayXG":
                float(
                    away_row["xg"]
                ),

            "Matchweek":
                home_row["Matchweek"],

            "XGSource":
                "FBref",
        }
    )


fbref_xg = pd.DataFrame(
    fbref_rows
)


print(
    "FBref matched fixtures:",
    len(fbref_xg),
    "/ 1520",
)


if unmatched_fbref:

    unmatched_df = pd.DataFrame(
        unmatched_fbref
    )

    print()
    print("UNMATCHED FBREF FIXTURES")
    print("========================")

    print(
        unmatched_df.head(
            30
        ).to_string(
            index=False
        )
    )

    print()

    print(
        "Unmatched by season:"
    )

    print(
        unmatched_df[
            "Season"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


if len(fbref_xg) != 1520:

    fail(
        "FBref must match all 1520 fixtures "
        "from 2021/22 to 2024/25"
    )


# ==================================================
# BUILD 2025/26 UNDERSTAT BRIDGED XG
#
# IMPORTANT:
#
# Do NOT use raw Understat xG here.
#
# The existing project has already frozen a
# provider bridge using 2024/25 data.
#
# BridgedHomeXG and BridgedAwayXG therefore put
# 2025/26 Understat onto the historical FBref scale.
# ==================================================

print()
print("BUILDING 2025/26 BRIDGED XG")
print("===========================")


required_understat = [
    "Season",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "BridgedHomeXG",
    "BridgedAwayXG",
]


missing_understat = [
    column
    for column in required_understat
    if column not in understat.columns
]


if missing_understat:

    fail(
        "Understat file missing columns: "
        +
        ", ".join(missing_understat)
    )


understat_2025 = understat[
    required_understat
].copy()


understat_2025 = understat_2025[
    understat_2025["Season"]
    ==
    "2025/26"
].copy()


if len(understat_2025) != 380:

    fail(
        "Expected 380 Understat 2025/26 "
        f"fixtures, found {len(understat_2025)}"
    )


target_2025 = matches[
    matches["Season"]
    ==
    "2025/26"
][
    [
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
    ]
].copy()


if len(target_2025) != 380:

    fail(
        "Expected 380 Football Copilot "
        "2025/26 fixtures"
    )


understat_joined = (
    target_2025.merge(
        understat_2025,
        on=[
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",
        ],
        how="left",
        validate="one_to_one",
    )
)


missing_2025_xg = (
    understat_joined[
        [
            "BridgedHomeXG",
            "BridgedAwayXG",
        ]
    ]
    .isna()
    .any(
        axis=1
    )
)


if missing_2025_xg.any():

    print()

    print(
        understat_joined.loc[
            missing_2025_xg,
            [
                "Date",
                "HomeTeam",
                "AwayTeam",
            ],
        ].to_string(
            index=False
        )
    )

    fail(
        "2025/26 bridged Understat xG "
        "does not cover all fixtures"
    )


# ==================================================
# 2025/26 MATCHWEEK
#
# The local FBref history ends at 2024/25 and the
# reconciled Understat 2025/26 file does not contain
# an authoritative Premier League round / matchweek.
#
# Do NOT infer matchweek by sorting fixtures and
# dividing the row number by 10. Postponements and
# rescheduled matches can make that assignment wrong.
#
# Matchweek is context metadata only and is not used
# as a Model 6 predictor. Leave it missing for 2025/26
# until an authoritative, reproducible source is added.
# ==================================================

understat_joined["Matchweek"] = np.nan


understat_xg = pd.DataFrame(
    {
        "Season":
            understat_joined[
                "Season"
            ],

        "Date":
            understat_joined[
                "Date"
            ],

        "HomeTeam":
            understat_joined[
                "HomeTeam"
            ],

        "AwayTeam":
            understat_joined[
                "AwayTeam"
            ],

        "HomeGoals":
            understat_joined[
                "FTHG"
            ],

        "AwayGoals":
            understat_joined[
                "FTAG"
            ],

        "HomeXG":
            understat_joined[
                "BridgedHomeXG"
            ],

        "AwayXG":
            understat_joined[
                "BridgedAwayXG"
            ],

        "Matchweek":
            understat_joined[
                "Matchweek"
            ],

        "XGSource":
            "Understat_Bridged",
    }
)


print(
    "Understat bridged fixtures:",
    len(understat_xg),
    "/ 380",
)


# ==================================================
# CANONICAL FIVE-SEASON XG HISTORY
# ==================================================

print()
print("BUILDING CANONICAL XG HISTORY")
print("=============================")


xg_matches = pd.concat(
    [
        fbref_xg,
        understat_xg,
    ],
    ignore_index=True,
)


xg_matches = (
    xg_matches
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


print(
    "Total canonical xG fixtures:",
    len(xg_matches),
    "/ 1900",
)


if len(xg_matches) != 1900:

    fail(
        "Canonical xG history must contain "
        "exactly 1900 fixtures"
    )


fixture_keys = [
    "Season",
    "Date",
    "HomeTeam",
    "AwayTeam",
]


if xg_matches[
    fixture_keys
].duplicated().any():

    fail(
        "Duplicate canonical xG fixtures"
    )


if (
    xg_matches[
        [
            "HomeXG",
            "AwayXG",
        ]
    ]
    .isna()
    .any()
    .any()
):

    fail(
        "Missing xG values in canonical history"
    )


# ==================================================
# VERIFY EXACT 1900-FIXTURE RECONCILIATION
# ==================================================

fixture_check = (
    matches[
        fixture_keys
    ]
    .merge(
        xg_matches[
            fixture_keys
            +
            [
                "HomeXG",
                "AwayXG",
            ]
        ],
        on=fixture_keys,
        how="left",
        validate="one_to_one",
    )
)


xg_covered = int(
    fixture_check[
        "HomeXG"
    ]
    .notna()
    .sum()
)


print(
    "Target fixture xG coverage:",
    xg_covered,
    "/ 1900",
)


if xg_covered != 1900:

    fail(
        "Canonical xG history does not "
        "cover all target fixtures"
    )


print()
print("XG SOURCE COUNTS")
print("================")

print(
    xg_matches[
        "XGSource"
    ]
    .value_counts()
    .to_string()
)


# ==================================================
# JOIN XG TO MATCHES
# ==================================================

matches_lookup = (
    matches.merge(
        xg_matches[
            fixture_keys
            +
            [
                "HomeXG",
                "AwayXG",
                "Matchweek",
                "XGSource",
            ]
        ],
        on=fixture_keys,
        how="left",
        validate="one_to_one",
    )
)


# ==================================================
# BUILD TEAM-PERSPECTIVE HISTORY
#
# Shots and SOT come from matches_clean.csv,
# which has consistent five-season coverage.
#
# xG comes from the canonical reconciled xG layer.
# ==================================================

print()
print("BUILDING TEAM HISTORY")
print("=====================")


team_rows = []


for _, match in matches_lookup.iterrows():

    # HOME PERSPECTIVE

    team_rows.append(
        {
            "Season":
                match["Season"],

            "Date":
                match["Date"],

            "Team":
                match["HomeTeam"],

            "Opponent":
                match["AwayTeam"],

            "Venue":
                "Home",

            "GF":
                match["FTHG"],

            "GA":
                match["FTAG"],

            "XG":
                match["HomeXG"],

            "XGA":
                match["AwayXG"],

            "ShotsFor":
                match["HS"],

            "ShotsAgainst":
                match["AS"],

            "SOTFor":
                match["HST"],

            "SOTAgainst":
                match["AST"],
        }
    )


    # AWAY PERSPECTIVE

    team_rows.append(
        {
            "Season":
                match["Season"],

            "Date":
                match["Date"],

            "Team":
                match["AwayTeam"],

            "Opponent":
                match["HomeTeam"],

            "Venue":
                "Away",

            "GF":
                match["FTAG"],

            "GA":
                match["FTHG"],

            "XG":
                match["AwayXG"],

            "XGA":
                match["HomeXG"],

            "ShotsFor":
                match["AS"],

            "ShotsAgainst":
                match["HS"],

            "SOTFor":
                match["AST"],

            "SOTAgainst":
                match["HST"],
        }
    )


team_history = pd.DataFrame(
    team_rows
)


team_history = (
    team_history
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


print(
    "Team-match rows:",
    len(team_history),
    "/ 3800",
)


if len(team_history) != 3800:

    fail(
        "Expected exactly 3800 "
        "team-match history rows"
    )


# ==================================================
# HISTORY FUNCTIONS
# ==================================================

def get_history(
    team,
    current_date,
    n=None,
    venue=None,
):
    """
    Critical leakage control:

    Only matches strictly BEFORE current_date
    are available to a fixture.
    """

    history = team_history[
        (
            team_history["Team"]
            ==
            team
        )
        &
        (
            team_history["Date"]
            <
            current_date
        )
    ].copy()


    if venue is not None:

        history = history[
            history["Venue"]
            ==
            venue
        ]


    history = history.sort_values(
        "Date"
    )


    if n is not None:

        history = history.tail(
            n
        )


    return history


def calculate_underlying_stats(
    history,
):

    games = len(history)


    if games == 0:

        return {
            "games": 0,

            "xg_for": np.nan,
            "xg_against": np.nan,
            "xgd": np.nan,

            "shots_for": np.nan,
            "shots_against": np.nan,

            "sot_for": np.nan,
            "sot_against": np.nan,

            "xg_per_shot": np.nan,
            "sot_rate": np.nan,

            "finishing_vs_xg": np.nan,
            "defence_vs_xga": np.nan,

            "ew_xg_for": np.nan,
            "ew_xg_against": np.nan,
            "ew_xgd": np.nan,

            "ew_shots_for": np.nan,
            "ew_shots_against": np.nan,

            "ew_sot_for": np.nan,
            "ew_sot_against": np.nan,
        }


    total_shots = (
        history[
            "ShotsFor"
        ].sum()
    )


    return {
        "games":
            games,

        "xg_for":
            history[
                "XG"
            ].mean(),

        "xg_against":
            history[
                "XGA"
            ].mean(),

        "xgd":
            (
                history["XG"]
                -
                history["XGA"]
            ).mean(),

        "shots_for":
            history[
                "ShotsFor"
            ].mean(),

        "shots_against":
            history[
                "ShotsAgainst"
            ].mean(),

        "sot_for":
            history[
                "SOTFor"
            ].mean(),

        "sot_against":
            history[
                "SOTAgainst"
            ].mean(),

        "xg_per_shot":
            safe_divide(
                history[
                    "XG"
                ].sum(),
                total_shots,
            ),

        "sot_rate":
            safe_divide(
                history[
                    "SOTFor"
                ].sum(),
                total_shots,
            ),

        "finishing_vs_xg":
            (
                history["GF"]
                -
                history["XG"]
            ).mean(),

        "defence_vs_xga":
            (
                history["GA"]
                -
                history["XGA"]
            ).mean(),

        "ew_xg_for":
            weighted_average(
                history[
                    "XG"
                ]
            ),

        "ew_xg_against":
            weighted_average(
                history[
                    "XGA"
                ]
            ),

        "ew_xgd":
            weighted_average(
                history["XG"]
                -
                history["XGA"]
            ),

        "ew_shots_for":
            weighted_average(
                history[
                    "ShotsFor"
                ]
            ),

        "ew_shots_against":
            weighted_average(
                history[
                    "ShotsAgainst"
                ]
            ),

        "ew_sot_for":
            weighted_average(
                history[
                    "SOTFor"
                ]
            ),

        "ew_sot_against":
            weighted_average(
                history[
                    "SOTAgainst"
                ]
            ),
    }


# ==================================================
# BUILD LEAKAGE-SAFE PERFORMANCE FEATURES
# ==================================================

print()
print("BUILDING LEAKAGE-SAFE PERFORMANCE FEATURES")
print("===========================================")


underlying_rows = []


for index, match in matches_lookup.iterrows():

    if index % 200 == 0:

        print(
            f"Processing fixture "
            f"{index + 1}/"
            f"{len(matches_lookup)}"
        )


    date = match["Date"]

    home = match["HomeTeam"]

    away = match["AwayTeam"]


    home_history = get_history(
        home,
        date,
        n=10,
    )


    away_history = get_history(
        away,
        date,
        n=10,
    )


    home_stats = (
        calculate_underlying_stats(
            home_history
        )
    )


    away_stats = (
        calculate_underlying_stats(
            away_history
        )
    )


    home_venue_history = get_history(
        home,
        date,
        n=5,
        venue="Home",
    )


    away_venue_history = get_history(
        away,
        date,
        n=5,
        venue="Away",
    )


    home_venue_stats = (
        calculate_underlying_stats(
            home_venue_history
        )
    )


    away_venue_stats = (
        calculate_underlying_stats(
            away_venue_history
        )
    )


    if len(home_history) > 0:

        home_rest = (
            date
            -
            home_history[
                "Date"
            ].max()
        ).days

    else:

        home_rest = np.nan


    if len(away_history) > 0:

        away_rest = (
            date
            -
            away_history[
                "Date"
            ].max()
        ).days

    else:

        away_rest = np.nan


    if (
        not pd.isna(home_rest)
        and
        not pd.isna(away_rest)
    ):

        rest_difference = (
            home_rest
            -
            away_rest
        )

    else:

        rest_difference = np.nan


    underlying_rows.append(
        {
            "Season":
                match["Season"],

            "Date":
                date,

            "HomeTeam":
                home,

            "AwayTeam":
                away,

            "Matchweek":
                match["Matchweek"],


            # ------------------------------
            # HOME UNDERLYING PERFORMANCE
            # ------------------------------

            "HomeXGFor10":
                home_stats[
                    "xg_for"
                ],

            "HomeXGAgainst10":
                home_stats[
                    "xg_against"
                ],

            "HomeXGD10":
                home_stats[
                    "xgd"
                ],

            "HomeShotsFor10":
                home_stats[
                    "shots_for"
                ],

            "HomeShotsAgainst10":
                home_stats[
                    "shots_against"
                ],

            "HomeSOTFor10":
                home_stats[
                    "sot_for"
                ],

            "HomeSOTAgainst10":
                home_stats[
                    "sot_against"
                ],

            "HomeXGPerShot10":
                home_stats[
                    "xg_per_shot"
                ],

            "HomeSOTRate10":
                home_stats[
                    "sot_rate"
                ],

            "HomeFinishingVsXG10":
                home_stats[
                    "finishing_vs_xg"
                ],

            "HomeDefenceVsXGA10":
                home_stats[
                    "defence_vs_xga"
                ],

            "HomeEWXGFor":
                home_stats[
                    "ew_xg_for"
                ],

            "HomeEWXGAgainst":
                home_stats[
                    "ew_xg_against"
                ],

            "HomeEWXGD":
                home_stats[
                    "ew_xgd"
                ],

            "HomeEWShotsFor":
                home_stats[
                    "ew_shots_for"
                ],

            "HomeEWShotsAgainst":
                home_stats[
                    "ew_shots_against"
                ],

            "HomeEWSOTFor":
                home_stats[
                    "ew_sot_for"
                ],

            "HomeEWSOTAgainst":
                home_stats[
                    "ew_sot_against"
                ],


            # ------------------------------
            # AWAY UNDERLYING PERFORMANCE
            # ------------------------------

            "AwayXGFor10":
                away_stats[
                    "xg_for"
                ],

            "AwayXGAgainst10":
                away_stats[
                    "xg_against"
                ],

            "AwayXGD10":
                away_stats[
                    "xgd"
                ],

            "AwayShotsFor10":
                away_stats[
                    "shots_for"
                ],

            "AwayShotsAgainst10":
                away_stats[
                    "shots_against"
                ],

            "AwaySOTFor10":
                away_stats[
                    "sot_for"
                ],

            "AwaySOTAgainst10":
                away_stats[
                    "sot_against"
                ],

            "AwayXGPerShot10":
                away_stats[
                    "xg_per_shot"
                ],

            "AwaySOTRate10":
                away_stats[
                    "sot_rate"
                ],

            "AwayFinishingVsXG10":
                away_stats[
                    "finishing_vs_xg"
                ],

            "AwayDefenceVsXGA10":
                away_stats[
                    "defence_vs_xga"
                ],

            "AwayEWXGFor":
                away_stats[
                    "ew_xg_for"
                ],

            "AwayEWXGAgainst":
                away_stats[
                    "ew_xg_against"
                ],

            "AwayEWXGD":
                away_stats[
                    "ew_xgd"
                ],

            "AwayEWShotsFor":
                away_stats[
                    "ew_shots_for"
                ],

            "AwayEWShotsAgainst":
                away_stats[
                    "ew_shots_against"
                ],

            "AwayEWSOTFor":
                away_stats[
                    "ew_sot_for"
                ],

            "AwayEWSOTAgainst":
                away_stats[
                    "ew_sot_against"
                ],


            # ------------------------------
            # VENUE PERFORMANCE
            # ------------------------------

            "HomeVenueXGFor5":
                home_venue_stats[
                    "xg_for"
                ],

            "HomeVenueXGAgainst5":
                home_venue_stats[
                    "xg_against"
                ],

            "AwayVenueXGFor5":
                away_venue_stats[
                    "xg_for"
                ],

            "AwayVenueXGAgainst5":
                away_venue_stats[
                    "xg_against"
                ],


            # ------------------------------
            # RELATIVE / MATCHUP FEATURES
            # ------------------------------

            "XGDDifference":
                (
                    home_stats[
                        "xgd"
                    ]
                    -
                    away_stats[
                        "xgd"
                    ]
                ),

            "EWXGDDifference":
                (
                    home_stats[
                        "ew_xgd"
                    ]
                    -
                    away_stats[
                        "ew_xgd"
                    ]
                ),

            "ShotDifference":
                (
                    home_stats[
                        "shots_for"
                    ]
                    -
                    away_stats[
                        "shots_for"
                    ]
                ),

            "SOTDifference":
                (
                    home_stats[
                        "sot_for"
                    ]
                    -
                    away_stats[
                        "sot_for"
                    ]
                ),

            "XGAttackVsDefenceHome":
                (
                    home_stats[
                        "xg_for"
                    ]
                    -
                    away_stats[
                        "xg_against"
                    ]
                ),

            "XGAttackVsDefenceAway":
                (
                    away_stats[
                        "xg_for"
                    ]
                    -
                    home_stats[
                        "xg_against"
                    ]
                ),

            "SOTAttackVsDefenceHome":
                (
                    home_stats[
                        "sot_for"
                    ]
                    -
                    away_stats[
                        "sot_against"
                    ]
                ),

            "SOTAttackVsDefenceAway":
                (
                    away_stats[
                        "sot_for"
                    ]
                    -
                    home_stats[
                        "sot_against"
                    ]
                ),


            # ------------------------------
            # CONTEXT
            # ------------------------------

            "HomeRestDays":
                home_rest,

            "AwayRestDays":
                away_rest,

            "RestDaysDifference":
                rest_difference,

            "HomeUnderlyingHistoryGames":
                home_stats[
                    "games"
                ],

            "AwayUnderlyingHistoryGames":
                away_stats[
                    "games"
                ],
        }
    )


underlying = pd.DataFrame(
    underlying_rows
)


# ==================================================
# MARKET DATA
# ==================================================

print()
print("BUILDING OPENING MARKET FEATURES")
print("================================")


market_frames = []


for season, filename in RAW_FILES.items():

    data = pd.read_csv(
        filename
    ).copy()


    data["Date"] = pd.to_datetime(
        data["Date"],
        dayfirst=True,
        errors="coerce",
    ).dt.normalize()


    required = [
        "Date",
        "HomeTeam",
        "AwayTeam",

        "AvgH",
        "AvgD",
        "AvgA",

        "AvgCH",
        "AvgCD",
        "AvgCA",

        "Avg>2.5",
        "Avg<2.5",

        "AvgC>2.5",
        "AvgC<2.5",

        "AHh",
        "AvgAHH",
        "AvgAHA",

        "AHCh",
        "AvgCAHH",
        "AvgCAHA",
    ]


    missing = [
        column
        for column in required
        if column not in data.columns
    ]


    if missing:

        fail(
            f"{filename} missing columns: "
            +
            ", ".join(missing)
        )


    data = data[
        required
    ].copy()


    data["Season"] = season


    market_frames.append(
        data
    )


market = pd.concat(
    market_frames,
    ignore_index=True,
)


print(
    "Market fixtures:",
    len(market),
    "/ 1900",
)


if len(market) != 1900:

    fail(
        "Expected exactly 1900 "
        "market fixtures"
    )


market_rows = []


for _, row in market.iterrows():

    open_1x2 = (
        no_vig_probabilities(
            [
                row["AvgH"],
                row["AvgD"],
                row["AvgA"],
            ]
        )
    )


    close_1x2 = (
        no_vig_probabilities(
            [
                row["AvgCH"],
                row["AvgCD"],
                row["AvgCA"],
            ]
        )
    )


    open_ou = (
        no_vig_probabilities(
            [
                row["Avg>2.5"],
                row["Avg<2.5"],
            ]
        )
    )


    close_ou = (
        no_vig_probabilities(
            [
                row["AvgC>2.5"],
                row["AvgC<2.5"],
            ]
        )
    )


    open_ah = (
        no_vig_probabilities(
            [
                row["AvgAHH"],
                row["AvgAHA"],
            ]
        )
    )


    close_ah = (
        no_vig_probabilities(
            [
                row["AvgCAHH"],
                row["AvgCAHA"],
            ]
        )
    )


    market_rows.append(
        {
            "Season":
                row["Season"],

            "Date":
                row["Date"],

            "HomeTeam":
                row["HomeTeam"],

            "AwayTeam":
                row["AwayTeam"],


            # ------------------------------
            # OPENING MARKET
            # MODEL 6 CANDIDATE INPUTS
            # ------------------------------

            "MarketOpenHomeProb":
                open_1x2[0],

            "MarketOpenDrawProb":
                open_1x2[1],

            "MarketOpenAwayProb":
                open_1x2[2],

            "MarketOpenHomeAwayDiff":
                (
                    open_1x2[0]
                    -
                    open_1x2[2]
                ),

            "MarketOpenOver25Prob":
                open_ou[0],

            "MarketOpenUnder25Prob":
                open_ou[1],

            "MarketOpenAHLine":
                row["AHh"],

            "MarketOpenAHHomeProb":
                open_ah[0],

            "MarketOpenAHAwayProb":
                open_ah[1],


            # ------------------------------
            # CLOSING MARKET
            # BENCHMARK ONLY
            # ------------------------------

            "BenchmarkCloseHomeProb":
                close_1x2[0],

            "BenchmarkCloseDrawProb":
                close_1x2[1],

            "BenchmarkCloseAwayProb":
                close_1x2[2],

            "BenchmarkCloseOver25Prob":
                close_ou[0],

            "BenchmarkCloseUnder25Prob":
                close_ou[1],

            "BenchmarkCloseAHLine":
                row["AHCh"],

            "BenchmarkCloseAHHomeProb":
                close_ah[0],

            "BenchmarkCloseAHAwayProb":
                close_ah[1],
        }
    )


market_features = pd.DataFrame(
    market_rows
)


# ==================================================
# ASSEMBLE MODEL 6 DATASET
#
# Start with the exact Model 2 fixture population.
#
# This means the first Model 6 OOT comparison is
# apples-to-apples against the controlled Model 2
# recreation.
# ==================================================

print()
print("ASSEMBLING MODEL 6 DATASET")
print("===========================")


base_columns = (
    fixture_keys
    +
    [
        "HomeGoals",
        "AwayGoals",
    ]
    +
    MODEL2_FEATURES
)


model6 = m2[
    base_columns
].copy()


model6 = model6.merge(
    underlying,
    on=fixture_keys,
    how="left",
    validate="one_to_one",
)


model6 = model6.merge(
    market_features,
    on=fixture_keys,
    how="left",
    validate="one_to_one",
)


model6["ActualResult"] = np.select(
    [
        model6["HomeGoals"]
        >
        model6["AwayGoals"],

        model6["HomeGoals"]
        ==
        model6["AwayGoals"],
    ],
    [
        "H",
        "D",
    ],
    default="A",
)


# ==================================================
# FINAL VALIDATION
# ==================================================

print()
print("MODEL 6 DATA VALIDATION")
print("=======================")


print(
    "Rows:",
    len(model6),
)


print(
    "Columns:",
    len(model6.columns),
)


if len(model6) != len(m2):

    fail(
        "Model 6 fixture population differs "
        "from Model 2"
    )


if model6[
    fixture_keys
].duplicated().any():

    fail(
        "Duplicate Model 6 fixtures"
    )


# ==================================================
# CORE XG COVERAGE
# ==================================================

core_xg_features = [
    "HomeXGFor10",
    "HomeXGAgainst10",
    "HomeXGD10",

    "AwayXGFor10",
    "AwayXGAgainst10",
    "AwayXGD10",

    "HomeEWXGD",
    "AwayEWXGD",

    "XGDDifference",
]


print()
print("CORE XG FEATURE MISSING VALUES")
print("==============================")

print(
    model6[
        core_xg_features
    ]
    .isna()
    .sum()
    .to_string()
)


# ==================================================
# SHOT COVERAGE
# ==================================================

shot_features = [
    "HomeShotsFor10",
    "HomeShotsAgainst10",
    "HomeSOTFor10",
    "HomeSOTAgainst10",

    "AwayShotsFor10",
    "AwayShotsAgainst10",
    "AwaySOTFor10",
    "AwaySOTAgainst10",
]


print()
print("SHOT / SOT FEATURE MISSING VALUES")
print("=================================")

print(
    model6[
        shot_features
    ]
    .isna()
    .sum()
    .to_string()
)


# ==================================================
# MARKET COVERAGE
# ==================================================

market_predictors = [
    "MarketOpenHomeProb",
    "MarketOpenDrawProb",
    "MarketOpenAwayProb",
    "MarketOpenHomeAwayDiff",

    "MarketOpenOver25Prob",
    "MarketOpenUnder25Prob",

    "MarketOpenAHLine",
    "MarketOpenAHHomeProb",
    "MarketOpenAHAwayProb",
]


print()
print("OPENING MARKET MISSING VALUES")
print("=============================")

print(
    model6[
        market_predictors
    ]
    .isna()
    .sum()
    .to_string()
)


# ==================================================
# MATCHWEEK COVERAGE
# ==================================================

print()
print("MATCHWEEK MISSING VALUES")
print("========================")

print(
    model6[
        "Matchweek"
    ]
    .isna()
    .sum()
)


# ==================================================
# ROWS BY SEASON
# ==================================================

print()
print("ROWS BY SEASON")
print("==============")

print(
    model6[
        "Season"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ==================================================
# HISTORICAL GW6 METADATA CHECK
#
# Matchweek is authoritative only where it came from
# FBref. The Model 6 dataset deliberately starts from
# the exact Model 2 fixture population, so the number
# of GW6 rows here is NOT expected to be 10 per season.
# Fixtures excluded by the Model 2 history requirement
# must remain excluded for an apples-to-apples test.
#
# 2025/26 is intentionally not assigned a synthetic
# matchweek in this builder. Exact 2025/26 GW6 fixtures
# must be supplied later from an authoritative source
# before the agreed three-season deployment simulation.
# ==================================================

exact_gw6 = model6[
    (
        model6["Season"].isin(
            [
                "2023/24",
                "2024/25",
            ]
        )
    )
    &
    (
        model6["Matchweek"]
        ==
        6
    )
].copy()


print()
print("HISTORICAL GW6 METADATA CHECK")
print("=============================")

exact_gw6_counts = (
    exact_gw6.groupby(
        "Season"
    )
    .size()
)

print(
    exact_gw6_counts
    .reindex(
        [
            "2023/24",
            "2024/25",
        ],
        fill_value=0,
    )
    .to_string()
)

print()
print(
    "2025/26 exact Matchweek metadata: "
    "not available in the current reconciled xG sources"
)

print(
    "No synthetic 2025/26 Matchweek values were created."
)


# ==================================================
# SAVE
# ==================================================

model6 = (
    model6
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


Path(
    "data/processed"
).mkdir(
    parents=True,
    exist_ok=True,
)


model6.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print(
    "Saved:",
    OUTPUT_FILE,
)


print()
print(
    "MODEL 6 FEATURE BUILD COMPLETE"
)

print(
    "=============================="
)