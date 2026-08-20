from pathlib import Path

import pandas as pd


# ==================================================
# PATHS
# ==================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

CHAMPIONSHIP_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "championship_team_seasons.csv"
)

PREMIER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "matches_clean.csv"
)

TRANSLATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "promotion_translation_factors.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "promoted_team_priors_2026_27.csv"
)


# ==================================================
# CONFIG
# ==================================================

SOURCE_SEASON = "2025/26"
TARGET_SEASON = "2026/27"

PROMOTED_TEAMS = [
    "Coventry",
    "Hull",
    "Ipswich",
]

TRANSLATED_WEIGHT = 0.75
BASELINE_WEIGHT = 0.25


# ==================================================
# LOAD DATA
# ==================================================

championship = pd.read_csv(
    CHAMPIONSHIP_FILE
)

premier = pd.read_csv(
    PREMIER_FILE
)

translation = pd.read_csv(
    TRANSLATION_FILE
)


# ==================================================
# GET MEDIAN TRANSLATION FACTORS
# ==================================================

def get_factor(metric):

    row = translation[
        translation["Metric"]
        ==
        metric
    ]

    if row.empty:

        raise ValueError(
            f"No translation factor found for {metric}"
        )

    return float(
        row.iloc[0]["MedianRatio"]
    )


ppg_factor = get_factor(
    "PPG"
)

gf_factor = get_factor(
    "GoalsForPG"
)

ga_factor = get_factor(
    "GoalsAgainstPG"
)


# ==================================================
# HISTORICAL PREMIER LEAGUE BASELINE
#
# Uses only seasons before 2026/27.
# ==================================================

historical_pl = premier[
    premier["Season"]
    !=
    TARGET_SEASON
].copy()


total_matches = len(
    historical_pl
)

if total_matches == 0:

    raise ValueError(
        "No historical Premier League matches found."
    )


total_team_matches = (
    total_matches
    *
    2
)


draws = (
    historical_pl["FTR"]
    ==
    "D"
).sum()


total_points = (
    total_matches
    *
    3
    -
    draws
)


total_goals = (
    historical_pl["FTHG"].sum()
    +
    historical_pl["FTAG"].sum()
)


baseline_ppg = (
    total_points
    /
    total_team_matches
)

baseline_gf = (
    total_goals
    /
    total_team_matches
)

baseline_ga = baseline_gf


# ==================================================
# DISPLAY METHODOLOGY
# ==================================================

print()
print("FOOTBALL COPILOT")
print("2026/27 PROMOTED TEAM PRIORS")
print("============================")

print()
print("TRANSLATION FACTORS")
print("===================")

print(
    f"PPG:              {ppg_factor:.3f}"
)

print(
    f"Goals For PG:     {gf_factor:.3f}"
)

print(
    f"Goals Against PG: {ga_factor:.3f}"
)


print()
print("HISTORICAL PL BASELINE")
print("======================")

print(
    f"PPG:              {baseline_ppg:.3f}"
)

print(
    f"Goals For PG:     {baseline_gf:.3f}"
)

print(
    f"Goals Against PG: {baseline_ga:.3f}"
)


print()
print("SHRINKAGE")
print("=========")

print(
    f"Translated Championship: "
    f"{TRANSLATED_WEIGHT:.0%}"
)

print(
    f"Premier League baseline: "
    f"{BASELINE_WEIGHT:.0%}"
)


# ==================================================
# GENERATE PRIORS
# ==================================================

rows = []


for team in PROMOTED_TEAMS:

    source = championship[
        (
            championship["Season"]
            ==
            SOURCE_SEASON
        )
        &
        (
            championship["Team"]
            ==
            team
        )
    ]


    if source.empty:

        raise ValueError(
            f"No {SOURCE_SEASON} Championship "
            f"data found for {team}"
        )


    source = source.iloc[0]


    # ----------------------------------------------
    # TRANSLATE CHAMPIONSHIP PERFORMANCE
    # ----------------------------------------------

    translated_ppg = (
        source["PPG"]
        *
        ppg_factor
    )

    translated_gf = (
        source["GoalsForPG"]
        *
        gf_factor
    )

    translated_ga = (
        source["GoalsAgainstPG"]
        *
        ga_factor
    )


    # ----------------------------------------------
    # SHRINK TOWARDS PL BASELINE
    # ----------------------------------------------

    prior_ppg = (
        TRANSLATED_WEIGHT
        *
        translated_ppg
        +
        BASELINE_WEIGHT
        *
        baseline_ppg
    )

    prior_gf = (
        TRANSLATED_WEIGHT
        *
        translated_gf
        +
        BASELINE_WEIGHT
        *
        baseline_gf
    )

    prior_ga = (
        TRANSLATED_WEIGHT
        *
        translated_ga
        +
        BASELINE_WEIGHT
        *
        baseline_ga
    )


    rows.append(
        {

            "TargetSeason":
                TARGET_SEASON,

            "Team":
                team,

            "SourceSeason":
                SOURCE_SEASON,

            "SourceDivision":
                "Championship",

            "SourcePPG":
                source["PPG"],

            "SourceGoalsForPG":
                source["GoalsForPG"],

            "SourceGoalsAgainstPG":
                source["GoalsAgainstPG"],

            "PPGTranslationFactor":
                ppg_factor,

            "GoalsForTranslationFactor":
                gf_factor,

            "GoalsAgainstTranslationFactor":
                ga_factor,

            "TranslatedWeight":
                TRANSLATED_WEIGHT,

            "BaselineWeight":
                BASELINE_WEIGHT,

            "TranslatedPPG":
                translated_ppg,

            "TranslatedGoalsForPG":
                translated_gf,

            "TranslatedGoalsAgainstPG":
                translated_ga,

            "PriorPPG":
                prior_ppg,

            "PriorGoalsForPG":
                prior_gf,

            "PriorGoalsAgainstPG":
                prior_ga,

            "PriorType":
                "PromotedTeamColdStart",
        }
    )


# ==================================================
# SAVE
# ==================================================

output = pd.DataFrame(
    rows
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
# PRINT RESULTS
# ==================================================

print()
print("FINAL PROMOTED TEAM PRIORS")
print("==========================")

display_columns = [
    "Team",
    "SourcePPG",
    "SourceGoalsForPG",
    "SourceGoalsAgainstPG",
    "TranslatedPPG",
    "TranslatedGoalsForPG",
    "TranslatedGoalsAgainstPG",
    "PriorPPG",
    "PriorGoalsForPG",
    "PriorGoalsAgainstPG",
]


print(
    output[
        display_columns
    ].to_string(
        index=False,
        formatters={
            "SourcePPG":
                "{:.3f}".format,

            "SourceGoalsForPG":
                "{:.3f}".format,

            "SourceGoalsAgainstPG":
                "{:.3f}".format,

            "TranslatedPPG":
                "{:.3f}".format,

            "TranslatedGoalsForPG":
                "{:.3f}".format,

            "TranslatedGoalsAgainstPG":
                "{:.3f}".format,

            "PriorPPG":
                "{:.3f}".format,

            "PriorGoalsForPG":
                "{:.3f}".format,

            "PriorGoalsAgainstPG":
                "{:.3f}".format,
        },
    )
)


print()
print(
    f"Saved: {OUTPUT_FILE}"
)