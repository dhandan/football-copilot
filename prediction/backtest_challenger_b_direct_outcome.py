from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import poisson

from sklearn.linear_model import (
    LogisticRegression,
    PoissonRegressor,
)
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Files
# --------------------------------------------------

MODEL2_FILE = (
    "data/processed/"
    "prediction_features_v2.csv"
)

H3_FILE = (
    "reports/post_gw5/"
    "h3_underlying_form_predictions.csv"
)

DETAIL_FILE = (
    "reports/post_gw5/"
    "challenger_b_predictions.csv"
)

SUMMARY_FILE = (
    "reports/post_gw5/"
    "challenger_b_summary.csv"
)

SEASON_FILE = (
    "reports/post_gw5/"
    "challenger_b_by_season.csv"
)


# --------------------------------------------------
# Frozen Model 2 features
# --------------------------------------------------

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


# --------------------------------------------------
# H3 underlying-performance features
#
# These are reconstructed leakage-safely from
# matches_clean.csv below.
# --------------------------------------------------

H3_FEATURES = [

    "HomeRecentShotsFor",
    "HomeRecentShotsAgainst",
    "HomeRecentSOTFor",
    "HomeRecentSOTAgainst",
    "HomeRecentSOTRate",

    "AwayRecentShotsFor",
    "AwayRecentShotsAgainst",
    "AwayRecentSOTFor",
    "AwayRecentSOTAgainst",
    "AwayRecentSOTRate",

    "RecentShotsDifference",
    "RecentSOTDifference",

    "HomeAttackSOTVsAwayDefence",
    "AwayAttackSOTVsHomeDefence",
]


CHALLENGER_FEATURES = (
    MODEL2_FEATURES
    +
    H3_FEATURES
)


LABEL_MAP = {
    "H": 0,
    "D": 1,
    "A": 2,
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def actual_result(
    home_goals,
    away_goals,
):

    if home_goals > away_goals:
        return "H"

    if home_goals == away_goals:
        return "D"

    return "A"


def get_previous_matches(
    matches,
    team,
    current_date,
    n=5,
):

    previous = matches[
        (
            (
                matches["HomeTeam"]
                ==
                team
            )
            |
            (
                matches["AwayTeam"]
                ==
                team
            )
        )
        &
        (
            matches["Date"]
            <
            current_date
        )
    ].copy()

    previous = previous.sort_values(
        "Date"
    )

    return previous.tail(
        n
    )


def calculate_shot_stats(
    previous,
    team,
):

    games = len(
        previous
    )

    if games == 0:

        return {
            "shots_for": 0.0,
            "shots_against": 0.0,
            "sot_for": 0.0,
            "sot_against": 0.0,
            "sot_rate": 0.0,
        }

    shots_for = 0.0
    shots_against = 0.0

    sot_for = 0.0
    sot_against = 0.0

    for _, match in previous.iterrows():

        if (
            match["HomeTeam"]
            ==
            team
        ):

            shots_for += (
                match["HS"]
            )

            shots_against += (
                match["AS"]
            )

            sot_for += (
                match["HST"]
            )

            sot_against += (
                match["AST"]
            )

        else:

            shots_for += (
                match["AS"]
            )

            shots_against += (
                match["HS"]
            )

            sot_for += (
                match["AST"]
            )

            sot_against += (
                match["HST"]
            )

    sot_rate = (
        sot_for / shots_for
        if shots_for > 0
        else 0.0
    )

    return {
        "shots_for":
            shots_for / games,

        "shots_against":
            shots_against / games,

        "sot_for":
            sot_for / games,

        "sot_against":
            sot_against / games,

        "sot_rate":
            sot_rate,
    }


def add_h3_features(
    base,
    matches,
):

    rows = []

    for _, fixture in base.iterrows():

        date = fixture[
            "Date"
        ]

        home_team = fixture[
            "HomeTeam"
        ]

        away_team = fixture[
            "AwayTeam"
        ]

        home_previous = (
            get_previous_matches(
                matches,
                home_team,
                date,
                n=5,
            )
        )

        away_previous = (
            get_previous_matches(
                matches,
                away_team,
                date,
                n=5,
            )
        )

        home = calculate_shot_stats(
            home_previous,
            home_team,
        )

        away = calculate_shot_stats(
            away_previous,
            away_team,
        )

        rows.append(
            {
                "HomeRecentShotsFor":
                    home["shots_for"],

                "HomeRecentShotsAgainst":
                    home["shots_against"],

                "HomeRecentSOTFor":
                    home["sot_for"],

                "HomeRecentSOTAgainst":
                    home["sot_against"],

                "HomeRecentSOTRate":
                    home["sot_rate"],

                "AwayRecentShotsFor":
                    away["shots_for"],

                "AwayRecentShotsAgainst":
                    away["shots_against"],

                "AwayRecentSOTFor":
                    away["sot_for"],

                "AwayRecentSOTAgainst":
                    away["sot_against"],

                "AwayRecentSOTRate":
                    away["sot_rate"],

                "RecentShotsDifference":
                    (
                        home["shots_for"]
                        -
                        away["shots_for"]
                    ),

                "RecentSOTDifference":
                    (
                        home["sot_for"]
                        -
                        away["sot_for"]
                    ),

                "HomeAttackSOTVsAwayDefence":
                    (
                        home["sot_for"]
                        -
                        away["sot_against"]
                    ),

                "AwayAttackSOTVsHomeDefence":
                    (
                        away["sot_for"]
                        -
                        home["sot_against"]
                    ),
            }
        )

    feature_frame = pd.DataFrame(
        rows,
        index=base.index,
    )

    return pd.concat(
        [
            base,
            feature_frame,
        ],
        axis=1,
    )


def poisson_probabilities(
    home_xg,
    away_xg,
    max_goals=8,
):

    home = 0.0
    draw = 0.0
    away = 0.0

    modal_probability = -1.0
    modal_score = None

    for home_goals in range(
        max_goals + 1
    ):

        for away_goals in range(
            max_goals + 1
        ):

            probability = (
                poisson.pmf(
                    home_goals,
                    home_xg,
                )
                *
                poisson.pmf(
                    away_goals,
                    away_xg,
                )
            )

            if home_goals > away_goals:
                home += probability

            elif home_goals == away_goals:
                draw += probability

            else:
                away += probability

            if probability > modal_probability:

                modal_probability = (
                    probability
                )

                modal_score = (
                    f"{home_goals}-"
                    f"{away_goals}"
                )

    total = (
        home
        +
        draw
        +
        away
    )

    return {
        "H":
            home / total,

        "D":
            draw / total,

        "A":
            away / total,

        "modal_score":
            modal_score,
    }


def metrics(
    data,
    prefix,
):

    actual = data[
        "ActualResult"
    ]

    predicted = data[
        f"{prefix}PredictedResult"
    ]

    probabilities = data[
        [
            f"{prefix}HomeProbability",
            f"{prefix}DrawProbability",
            f"{prefix}AwayProbability",
        ]
    ].to_numpy()

    numeric_actual = actual.map(
        LABEL_MAP
    )

    accuracy = (
        actual == predicted
    ).mean()

    loss = log_loss(
        numeric_actual,
        probabilities,
        labels=[
            0,
            1,
            2,
        ],
    )

    actual_matrix = np.zeros(
        (
            len(data),
            3,
        )
    )

    for index, result in enumerate(
        actual
    ):

        actual_matrix[
            index,
            LABEL_MAP[result],
        ] = 1.0

    brier = np.mean(
        np.sum(
            (
                probabilities
                -
                actual_matrix
            )
            ** 2,
            axis=1,
        )
    )

    predicted_draws = int(
        (
            predicted == "D"
        ).sum()
    )

    actual_draws = int(
        (
            actual == "D"
        ).sum()
    )

    return {
        "Matches":
            len(data),

        "AccuracyPct":
            accuracy * 100,

        "LogLoss":
            loss,

        "Brier":
            brier,

        "PredictedDraws":
            predicted_draws,

        "ActualDraws":
            actual_draws,
    }


# --------------------------------------------------
# Load
# --------------------------------------------------

print()
print("FOOTBALL COPILOT")
print("CHALLENGER B: DIRECT OUTCOME MODEL")
print("==================================")

base = pd.read_csv(
    MODEL2_FILE
)

matches = pd.read_csv(
    "data/processed/"
    "matches_clean.csv"
)

base["Date"] = pd.to_datetime(
    base["Date"],
    errors="coerce",
)

matches["Date"] = pd.to_datetime(
    matches["Date"],
    errors="coerce",
)

base = base.sort_values(
    "Date"
).reset_index(
    drop=True
)

matches = matches.sort_values(
    "Date"
).reset_index(
    drop=True
)


print()
print(
    "Building leakage-safe "
    "shots/SOT features..."
)

df = add_h3_features(
    base,
    matches,
)


df["ActualResult"] = df.apply(
    lambda row:
        actual_result(
            row["HomeGoals"],
            row["AwayGoals"],
        ),
    axis=1,
)


SEASONS = sorted(
    df["Season"].unique()
)

print()
print(
    "Seasons:",
    SEASONS,
)


# --------------------------------------------------
# Walk-forward
# --------------------------------------------------

all_rows = []


for test_index in range(
    2,
    len(SEASONS),
):

    test_season = (
        SEASONS[
            test_index
        ]
    )

    training_seasons = (
        SEASONS[
            :test_index
        ]
    )

    train = df[
        df["Season"].isin(
            training_seasons
        )
    ].copy()

    test = df[
        df["Season"]
        ==
        test_season
    ].copy()

    print()
    print(
        "Training on:",
        ", ".join(
            training_seasons
        ),
    )

    print(
        "Testing on:",
        test_season,
    )

    # ----------------------------------------------
    # Exact Model 2 comparator
    # ----------------------------------------------

    home_model = PoissonRegressor(
        alpha=0.1,
        max_iter=1000,
    )

    away_model = PoissonRegressor(
        alpha=0.1,
        max_iter=1000,
    )

    home_model.fit(
        train[
            MODEL2_FEATURES
        ],
        train["HomeGoals"],
    )

    away_model.fit(
        train[
            MODEL2_FEATURES
        ],
        train["AwayGoals"],
    )


    # ----------------------------------------------
    # Challenger B
    # ----------------------------------------------

    challenger = Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=0.25,
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    challenger.fit(
        train[
            CHALLENGER_FEATURES
        ],
        train[
            "ActualResult"
        ],
    )


    # ----------------------------------------------
    # Predict
    # ----------------------------------------------

    test = test.reset_index(
        drop=True
    )

    m2_home_xg = (
        home_model.predict(
            test[
                MODEL2_FEATURES
            ]
        )
    )

    m2_away_xg = (
        away_model.predict(
            test[
                MODEL2_FEATURES
            ]
        )
    )

    challenger_probabilities = (
        challenger.predict_proba(
            test[
                CHALLENGER_FEATURES
            ]
        )
    )

    class_order = list(
        challenger.named_steps[
            "classifier"
        ].classes_
    )


    for index in range(
        len(test)
    ):

        actual = test.loc[
            index,
            "ActualResult",
        ]

        m2 = poisson_probabilities(
            m2_home_xg[index],
            m2_away_xg[index],
        )

        cb_probability_map = {
            class_name:
                challenger_probabilities[
                    index,
                    class_index,
                ]

            for (
                class_index,
                class_name
            ) in enumerate(
                class_order
            )
        }

        cb_home = (
            cb_probability_map["H"]
        )

        cb_draw = (
            cb_probability_map["D"]
        )

        cb_away = (
            cb_probability_map["A"]
        )

        m2_prediction = max(
            ["H", "D", "A"],
            key=lambda result:
                m2[result],
        )

        cb_prediction = max(
            ["H", "D", "A"],
            key=lambda result:
                cb_probability_map[
                    result
                ],
        )

        all_rows.append(
            {
                "Season":
                    test_season,

                "Date":
                    test.loc[
                        index,
                        "Date",
                    ],

                "HomeTeam":
                    test.loc[
                        index,
                        "HomeTeam",
                    ],

                "AwayTeam":
                    test.loc[
                        index,
                        "AwayTeam",
                    ],

                "ActualResult":
                    actual,

                "M2HomeProbability":
                    m2["H"],

                "M2DrawProbability":
                    m2["D"],

                "M2AwayProbability":
                    m2["A"],

                "M2PredictedResult":
                    m2_prediction,

                "CBHomeProbability":
                    cb_home,

                "CBDrawProbability":
                    cb_draw,

                "CBAwayProbability":
                    cb_away,

                "CBPredictedResult":
                    cb_prediction,
            }
        )


results = pd.DataFrame(
    all_rows
)


# --------------------------------------------------
# Overall
# --------------------------------------------------

summary_rows = []


for name, prefix in [
    ("Model 2", "M2"),
    (
        "Challenger B",
        "CB",
    ),
]:

    summary_rows.append(
        {
            "Model":
                name,

            **metrics(
                results,
                prefix,
            ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


print()
print("FULL OOT COMPARISON")
print("===================")

print(
    summary.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


# --------------------------------------------------
# By season
# --------------------------------------------------

season_rows = []


for season in sorted(
    results["Season"].unique()
):

    season_data = results[
        results["Season"]
        ==
        season
    ].copy()

    for name, prefix in [
        ("Model 2", "M2"),
        (
            "Challenger B",
            "CB",
        ),
    ]:

        season_rows.append(
            {
                "Season":
                    season,

                "Model":
                    name,

                **metrics(
                    season_data,
                    prefix,
                ),
            }
        )


season_summary = pd.DataFrame(
    season_rows
)


print()
print("COMPARISON BY SEASON")
print("====================")

print(
    season_summary.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


# --------------------------------------------------
# Close-fixture diagnostic
#
# Use MODEL 2's probability margin so both models
# are evaluated on exactly the same difficult games.
# --------------------------------------------------

m2_probabilities = results[
    [
        "M2HomeProbability",
        "M2DrawProbability",
        "M2AwayProbability",
    ]
].to_numpy()

sorted_m2 = np.sort(
    m2_probabilities,
    axis=1,
)

results[
    "M2DecisionMargin"
] = (
    sorted_m2[:, 2]
    -
    sorted_m2[:, 1]
)


close_results = results[
    results[
        "M2DecisionMargin"
    ]
    <= 0.10
].copy()


print()
print("CLOSE FIXTURES: MODEL 2 MARGIN <=10PP")
print("=====================================")

close_rows = []


for name, prefix in [
    ("Model 2", "M2"),
    (
        "Challenger B",
        "CB",
    ),
]:

    close_rows.append(
        {
            "Model":
                name,

            **metrics(
                close_results,
                prefix,
            ),
        }
    )


close_summary = pd.DataFrame(
    close_rows
)

print(
    close_summary.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


# --------------------------------------------------
# Frozen promotion criteria
# --------------------------------------------------

m2 = summary[
    summary["Model"]
    ==
    "Model 2"
].iloc[0]

cb = summary[
    summary["Model"]
    ==
    "Challenger B"
].iloc[0]


log_loss_pass = (
    cb["LogLoss"]
    <
    m2["LogLoss"]
)

brier_pass = (
    cb["Brier"]
    <
    m2["Brier"]
)

accuracy_change = (
    cb["AccuracyPct"]
    -
    m2["AccuracyPct"]
)

accuracy_pass = (
    accuracy_change
    >= -1.0
)


season_pivot = (
    season_summary.pivot(
        index="Season",
        columns="Model",
        values=[
            "LogLoss",
            "Brier",
        ],
    )
)


season_improvements = 0


for season in (
    season_pivot.index
):

    better_log_loss = (
        season_pivot.loc[
            season,
            (
                "LogLoss",
                "Challenger B",
            ),
        ]
        <
        season_pivot.loc[
            season,
            (
                "LogLoss",
                "Model 2",
            ),
        ]
    )

    better_brier = (
        season_pivot.loc[
            season,
            (
                "Brier",
                "Challenger B",
            ),
        ]
        <
        season_pivot.loc[
            season,
            (
                "Brier",
                "Model 2",
            ),
        ]
    )

    if (
        better_log_loss
        or
        better_brier
    ):

        season_improvements += 1


stability_pass = (
    season_improvements >= 2
)


print()
print("PRE-DEFINED PROMOTION CRITERIA")
print("==============================")

print(
    "Log Loss improves:       ",
    "PASS"
    if log_loss_pass
    else "FAIL",
)

print(
    "Brier improves:          ",
    "PASS"
    if brier_pass
    else "FAIL",
)

print(
    "Accuracy >= -1.0pp:      ",
    "PASS"
    if accuracy_pass
    else "FAIL",
    f"({accuracy_change:+.4f}pp)",
)

print(
    "Improvement >=2 seasons: ",
    "PASS"
    if stability_pass
    else "FAIL",
    f"({season_improvements}/3)",
)


full_oot_pass = (
    log_loss_pass
    and
    brier_pass
    and
    accuracy_pass
    and
    stability_pass
)


print()

if full_oot_pass:

    print(
        "FULL OOT GATE: PASS"
    )

    print(
        "Challenger B qualifies for the "
        "historical GW6 deployment simulation."
    )

else:

    print(
        "FULL OOT GATE: FAIL"
    )

    print(
        "Challenger B does not qualify for "
        "promotion on the full OOT evidence."
    )


# --------------------------------------------------
# Save
# --------------------------------------------------

Path(
    "reports/post_gw5"
).mkdir(
    parents=True,
    exist_ok=True,
)


results.to_csv(
    DETAIL_FILE,
    index=False,
)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)

season_summary.to_csv(
    SEASON_FILE,
    index=False,
)


print()
print("FILES SAVED")
print("===========")

print(
    DETAIL_FILE
)

print(
    SUMMARY_FILE
)

print(
    SEASON_FILE
)


print()
print("CHALLENGER B BACKTEST COMPLETE")
print("==============================")