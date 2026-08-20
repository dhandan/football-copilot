from pathlib import Path
import numpy as np
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

TRANSLATION_SAMPLE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "promotion_translation_sample.csv"
)

OUTPUT_DETAIL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "promoted_team_prior_backtest_detail.csv"
)

OUTPUT_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "promoted_team_prior_backtest_summary.csv"
)


# ==================================================
# CONFIG
# ==================================================

EARLY_MATCHES = 10

WEIGHTS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]


# ==================================================
# LOAD DATA
# ==================================================

championship = pd.read_csv(
    CHAMPIONSHIP_FILE
)

premier = pd.read_csv(
    PREMIER_FILE
)

sample = pd.read_csv(
    TRANSLATION_SAMPLE_FILE
)

premier["Date"] = pd.to_datetime(
    premier["Date"],
    errors="coerce",
)


# ==================================================
# PREMIER LEAGUE BASELINE
# ==================================================

def league_baseline_for_season(
    season
):

    season_df = premier[
        premier["Season"]
        ==
        season
    ].copy()

    if season_df.empty:
        return None

    total_matches = len(
        season_df
    )

    total_team_matches = (
        total_matches
        *
        2
    )

    total_points = (
        total_matches
        *
        3
    )

    draws = (
        season_df["FTR"]
        ==
        "D"
    ).sum()

    total_points -= draws

    total_goals = (
        season_df["FTHG"].sum()
        +
        season_df["FTAG"].sum()
    )

    return {

        "PPG":
            total_points
            /
            total_team_matches,

        "GoalsForPG":
            total_goals
            /
            total_team_matches,

        "GoalsAgainstPG":
            total_goals
            /
            total_team_matches,
    }


# ==================================================
# EARLY PL PERFORMANCE
# ==================================================

def early_pl_stats(
    season,
    team,
    n=10,
):

    team_matches = premier[
        (
            premier["Season"]
            ==
            season
        )
        &
        (
            (
                premier["HomeTeam"]
                ==
                team
            )
            |
            (
                premier["AwayTeam"]
                ==
                team
            )
        )
    ].copy()

    team_matches = (
        team_matches
        .sort_values(
            "Date"
        )
        .head(
            n
        )
    )

    if team_matches.empty:
        return None

    goals_for = 0
    goals_against = 0
    points = 0

    for _, match in (
        team_matches
        .iterrows()
    ):

        if (
            match["HomeTeam"]
            ==
            team
        ):

            gf = match["FTHG"]
            ga = match["FTAG"]

        else:

            gf = match["FTAG"]
            ga = match["FTHG"]

        goals_for += gf
        goals_against += ga

        if gf > ga:
            points += 3

        elif gf == ga:
            points += 1

    played = len(
        team_matches
    )

    return {

        "Played":
            played,

        "PPG":
            points / played,

        "GoalsForPG":
            goals_for / played,

        "GoalsAgainstPG":
            goals_against / played,
    }


# ==================================================
# LEAVE-ONE-OUT TRANSLATION FACTORS
# ==================================================

def translation_factors_without_team(
    excluded_team,
    excluded_pl_season,
):

    subset = sample[
        ~(
            (
                sample[
                    "PremierLeagueTeamName"
                ]
                ==
                excluded_team
            )
            &
            (
                sample[
                    "PremierLeagueSeason"
                ]
                ==
                excluded_pl_season
            )
        )
    ].copy()

    return {

        "PPG":
            subset[
                "PPGRatio"
            ].median(),

        "GoalsForPG":
            subset[
                "GoalsForRatio"
            ].median(),

        "GoalsAgainstPG":
            subset[
                "GoalsAgainstRatio"
            ].median(),
    }


# ==================================================
# BUILD BACKTEST
# ==================================================

rows = []


for _, promoted in (
    sample.iterrows()
):

    championship_season = promoted[
        "ChampionshipSeason"
    ]

    premier_season = promoted[
        "PremierLeagueSeason"
    ]

    championship_team = promoted[
        "Team"
    ]

    pl_team = promoted[
        "PremierLeagueTeamName"
    ]


    factors = (
        translation_factors_without_team(
            pl_team,
            premier_season,
        )
    )


    baseline = (
        league_baseline_for_season(
            premier_season
        )
    )


    actual = (
        early_pl_stats(
            premier_season,
            pl_team,
            n=EARLY_MATCHES,
        )
    )


    if (
        baseline is None
        or
        actual is None
    ):
        continue


    translated_ppg = (
        promoted[
            "ChampionshipPPG"
        ]
        *
        factors[
            "PPG"
        ]
    )

    translated_gf = (
        promoted[
            "ChampionshipGoalsForPG"
        ]
        *
        factors[
            "GoalsForPG"
        ]
    )

    translated_ga = (
        promoted[
            "ChampionshipGoalsAgainstPG"
        ]
        *
        factors[
            "GoalsAgainstPG"
        ]
    )


    for weight in WEIGHTS:

        prior_ppg = (
            weight
            *
            translated_ppg
            +
            (
                1
                -
                weight
            )
            *
            baseline[
                "PPG"
            ]
        )

        prior_gf = (
            weight
            *
            translated_gf
            +
            (
                1
                -
                weight
            )
            *
            baseline[
                "GoalsForPG"
            ]
        )

        prior_ga = (
            weight
            *
            translated_ga
            +
            (
                1
                -
                weight
            )
            *
            baseline[
                "GoalsAgainstPG"
            ]
        )


        ppg_abs_error = abs(
            prior_ppg
            -
            actual[
                "PPG"
            ]
        )

        gf_abs_error = abs(
            prior_gf
            -
            actual[
                "GoalsForPG"
            ]
        )

        ga_abs_error = abs(
            prior_ga
            -
            actual[
                "GoalsAgainstPG"
            ]
        )


        combined_mae = np.mean(
            [
                ppg_abs_error,
                gf_abs_error,
                ga_abs_error,
            ]
        )


        rows.append(
            {

                "ChampionshipSeason":
                    championship_season,

                "PremierLeagueSeason":
                    premier_season,

                "Team":
                    championship_team,

                "PremierLeagueTeamName":
                    pl_team,

                "WeightTranslated":
                    weight,

                "WeightBaseline":
                    1
                    -
                    weight,

                "TranslatedPPG":
                    translated_ppg,

                "TranslatedGoalsForPG":
                    translated_gf,

                "TranslatedGoalsAgainstPG":
                    translated_ga,

                "BaselinePPG":
                    baseline[
                        "PPG"
                    ],

                "BaselineGoalsForPG":
                    baseline[
                        "GoalsForPG"
                    ],

                "BaselineGoalsAgainstPG":
                    baseline[
                        "GoalsAgainstPG"
                    ],

                "PriorPPG":
                    prior_ppg,

                "PriorGoalsForPG":
                    prior_gf,

                "PriorGoalsAgainstPG":
                    prior_ga,

                "ActualEarlyPPG":
                    actual[
                        "PPG"
                    ],

                "ActualEarlyGoalsForPG":
                    actual[
                        "GoalsForPG"
                    ],

                "ActualEarlyGoalsAgainstPG":
                    actual[
                        "GoalsAgainstPG"
                    ],

                "PPGAbsoluteError":
                    ppg_abs_error,

                "GoalsForAbsoluteError":
                    gf_abs_error,

                "GoalsAgainstAbsoluteError":
                    ga_abs_error,

                "CombinedMAE":
                    combined_mae,
            }
        )


detail = pd.DataFrame(
    rows
)


# ==================================================
# SUMMARY BY WEIGHT
# ==================================================

summary = (
    detail
    .groupby(
        "WeightTranslated",
        as_index=False,
    )
    .agg(
        Teams=(
            "Team",
            "count",
        ),

        PPG_MAE=(
            "PPGAbsoluteError",
            "mean",
        ),

        GoalsFor_MAE=(
            "GoalsForAbsoluteError",
            "mean",
        ),

        GoalsAgainst_MAE=(
            "GoalsAgainstAbsoluteError",
            "mean",
        ),

        Combined_MAE=(
            "CombinedMAE",
            "mean",
        ),
    )
)


summary[
    "WeightBaseline"
] = (
    1
    -
    summary[
        "WeightTranslated"
    ]
)


summary = summary[
    [
        "WeightTranslated",
        "WeightBaseline",
        "Teams",
        "PPG_MAE",
        "GoalsFor_MAE",
        "GoalsAgainst_MAE",
        "Combined_MAE",
    ]
]


summary = summary.sort_values(
    "Combined_MAE"
)


# ==================================================
# SAVE
# ==================================================

detail.to_csv(
    OUTPUT_DETAIL,
    index=False,
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)


# ==================================================
# PRINT
# ==================================================

print()
print("FOOTBALL COPILOT")
print("PROMOTED TEAM PRIOR BACKTEST")
print("============================")

print()
print(
    f"Early Premier League window: "
    f"{EARLY_MATCHES} matches"
)

print()
print("SUMMARY BY SHRINKAGE WEIGHT")
print("===========================")

print(
    summary.to_string(
        index=False,
        formatters={
            "WeightTranslated":
                "{:.2f}".format,

            "WeightBaseline":
                "{:.2f}".format,

            "PPG_MAE":
                "{:.3f}".format,

            "GoalsFor_MAE":
                "{:.3f}".format,

            "GoalsAgainst_MAE":
                "{:.3f}".format,

            "Combined_MAE":
                "{:.3f}".format,
        },
    )
)


best = summary.iloc[0]


print()
print("BEST SHRINKAGE WEIGHT")
print("=====================")

print(
    f"Translated Championship weight: "
    f"{best['WeightTranslated']:.0%}"
)

print(
    f"Premier League baseline weight: "
    f"{best['WeightBaseline']:.0%}"
)

print(
    f"Combined MAE: "
    f"{best['Combined_MAE']:.3f}"
)


print()
print("FILES SAVED")
print("===========")

print(
    OUTPUT_DETAIL
)

print(
    OUTPUT_SUMMARY
)