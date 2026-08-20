from pathlib import Path

import pandas as pd


# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


PRIORS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "promoted_team_priors_2026_27.csv"
)


# ==================================================
# PROMOTED TEAM NAME ALIASES
#
# Live fixture API names → prior file names
# ==================================================

PROMOTED_TEAM_ALIASES = {

    "Coventry City":
        "Coventry",

    "Coventry":
        "Coventry",

    "Hull City":
        "Hull",

    "Hull":
        "Hull",

    "Ipswich Town":
        "Ipswich",

    "Ipswich":
        "Ipswich",
}


# ==================================================
# LOAD PRIORS
# ==================================================

def load_promoted_team_priors():

    if not PRIORS_FILE.exists():

        raise FileNotFoundError(
            "Promoted-team priors file does not exist: "
            f"{PRIORS_FILE}"
        )


    priors = pd.read_csv(
        PRIORS_FILE
    )


    required_columns = {

        "Team",
        "PriorPPG",
        "PriorGoalsForPG",
        "PriorGoalsAgainstPG",
    }


    missing_columns = (
        required_columns
        -
        set(
            priors.columns
        )
    )


    if missing_columns:

        raise ValueError(
            "Promoted-team priors file is missing "
            f"columns: {sorted(missing_columns)}"
        )


    return priors


PRIORS = (
    load_promoted_team_priors()
)


# ==================================================
# TEAM HELPERS
# ==================================================

def canonical_promoted_team_name(
    team
):

    return PROMOTED_TEAM_ALIASES.get(
        team
    )


def is_promoted_cold_start_team(
    team
):

    canonical_name = (
        canonical_promoted_team_name(
            team
        )
    )


    if canonical_name is None:

        return False


    return canonical_name in set(
        PRIORS["Team"]
    )


# ==================================================
# GET RAW PRIOR
# ==================================================

def get_promoted_team_prior(
    team
):

    canonical_name = (
        canonical_promoted_team_name(
            team
        )
    )


    if canonical_name is None:

        raise ValueError(
            f"{team} is not configured as a "
            "promoted-team cold-start club."
        )


    row = PRIORS[
        PRIORS["Team"]
        ==
        canonical_name
    ]


    if row.empty:

        raise ValueError(
            f"No promoted-team prior found for "
            f"{team}"
        )


    row = row.iloc[0]


    return {

        "team":
            team,

        "canonical_team":
            canonical_name,

        "ppg":
            float(
                row[
                    "PriorPPG"
                ]
            ),

        "goals_for_pg":
            float(
                row[
                    "PriorGoalsForPG"
                ]
            ),

        "goals_against_pg":
            float(
                row[
                    "PriorGoalsAgainstPG"
                ]
            ),

        "goal_difference_pg":
            (
                float(
                    row[
                        "PriorGoalsForPG"
                    ]
                )
                -
                float(
                    row[
                        "PriorGoalsAgainstPG"
                    ]
                )
            ),

        "games":
            0,

        "source":
            "PromotedTeamColdStart",

        "source_season":
            row.get(
                "SourceSeason",
                "2025/26",
            ),

        "target_season":
            row.get(
                "TargetSeason",
                "2026/27",
            ),
    }


# ==================================================
# MODEL-2 COMPATIBLE STATS
# ==================================================

def get_promoted_model_stats(
    team,
    context="generic",
):

    """
    Return a Model-2-compatible stats dictionary.

    During the initial cold-start period we use the
    validated promoted-team prior for all historical
    windows because the club has no Premier League
    observations in the production dataset.

    context is recorded for transparency but does not
    currently change the prior values.

    Valid examples:
        recent_5
        recent_10
        home_venue
        away_venue
        season
    """

    prior = (
        get_promoted_team_prior(
            team
        )
    )


    return {

        "goals_for_pg":
            prior[
                "goals_for_pg"
            ],

        "goals_against_pg":
            prior[
                "goals_against_pg"
            ],

        "ppg":
            prior[
                "ppg"
            ],

        "goal_difference_pg":
            prior[
                "goal_difference_pg"
            ],

        "games":
            0,

        "data_source":
            "PromotedTeamColdStart",

        "context":
            context,
    }


# ==================================================
# DISPLAY / TEST
# ==================================================

if __name__ == "__main__":

    print()
    print("FOOTBALL COPILOT")
    print("PROMOTED TEAM ADAPTER")
    print("=====================")


    for team in [
        "Coventry City",
        "Hull City",
    ]:

        print()
        print(team)
        print("-" * len(team))

        prior = (
            get_promoted_team_prior(
                team
            )
        )

        print(
            f"PPG:              "
            f"{prior['ppg']:.3f}"
        )

        print(
            f"Goals For PG:     "
            f"{prior['goals_for_pg']:.3f}"
        )

        print(
            f"Goals Against PG: "
            f"{prior['goals_against_pg']:.3f}"
        )

        print(
            f"Goal Difference:  "
            f"{prior['goal_difference_pg']:.3f}"
        )

        print(
            f"Source:           "
            f"{prior['source']}"
        )