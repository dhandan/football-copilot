from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import poisson

from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import log_loss


# ==================================================
# CONFIGURATION
# ==================================================

FEATURE_FILE = (
    "data/processed/"
    "prediction_features_v3.csv"
)

MARKET_FILE = (
    "data/processed/"
    "market_probabilities.csv"
)

DETAIL_OUTPUT = (
    "data/processed/"
    "final_model_validation_results.csv"
)

SUMMARY_OUTPUT = (
    "data/processed/"
    "final_model_validation_summary.csv"
)

SEASON_OUTPUT = (
    "data/processed/"
    "final_model_validation_by_season.csv"
)

EDGE_OUTPUT = (
    "data/processed/"
    "final_model_market_edges.csv"
)

EDGE_SUMMARY_OUTPUT = (
    "data/processed/"
    "final_model_market_edge_summary.csv"
)


# ==================================================
# FROZEN MODEL 2 SPECIFICATION
# ==================================================

MODEL_NAME = "Model 2"

MODEL_TYPE = "Poisson regression"

MODEL_ALPHA = 0.1


MODEL_FEATURES = [

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


LABEL_MAP = {
    "H": 0,
    "D": 1,
    "A": 2,
}


REVERSE_LABEL_MAP = {
    0: "H",
    1: "D",
    2: "A",
}


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def actual_result(
    home_goals,
    away_goals,
):

    if home_goals > away_goals:
        return "H"

    if home_goals == away_goals:
        return "D"

    return "A"


def match_probabilities(
    home_xg,
    away_xg,
    max_goals=8,
):

    home_probability = 0.0
    draw_probability = 0.0
    away_probability = 0.0


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

                home_probability += (
                    probability
                )


            elif home_goals == away_goals:

                draw_probability += (
                    probability
                )


            else:

                away_probability += (
                    probability
                )


    total = (
        home_probability
        +
        draw_probability
        +
        away_probability
    )


    return np.array(
        [
            home_probability / total,
            draw_probability / total,
            away_probability / total,
        ]
    )


def predicted_result(
    probabilities,
):

    return REVERSE_LABEL_MAP[
        int(
            np.argmax(
                probabilities
            )
        )
    ]


def multiclass_brier(
    actual_results,
    probabilities,
):

    actual_matrix = np.zeros(
        (
            len(actual_results),
            3,
        )
    )


    for row_number, result in enumerate(
        actual_results
    ):

        actual_matrix[
            row_number,
            LABEL_MAP[result]
        ] = 1


    return np.mean(
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


def calculate_metrics(
    actual_results,
    probabilities,
):

    predictions = np.array(
        [
            predicted_result(
                probability
            )
            for probability
            in probabilities
        ]
    )


    actual_array = np.array(
        actual_results
    )


    accuracy = np.mean(
        predictions
        ==
        actual_array
    )


    numeric_actual = np.array(
        [
            LABEL_MAP[result]
            for result
            in actual_results
        ]
    )


    loss = log_loss(
        numeric_actual,
        probabilities,
        labels=[
            0,
            1,
            2,
        ],
    )


    brier = multiclass_brier(
        actual_results,
        probabilities,
    )


    return {
        "AccuracyPct":
            accuracy * 100,

        "LogLoss":
            loss,

        "Brier":
            brier,

        "Predictions":
            predictions,
    }


# ==================================================
# LOAD DATA
# ==================================================

print()
print("FOOTBALL COPILOT")
print("FINAL MODEL VALIDATION")
print("======================")
print()

print(
    f"Selected model: {MODEL_NAME}"
)

print(
    f"Model type: {MODEL_TYPE}"
)

print(
    f"Poisson alpha: {MODEL_ALPHA}"
)

print(
    f"Number of features: "
    f"{len(MODEL_FEATURES)}"
)


features = pd.read_csv(
    FEATURE_FILE
)


features["Date"] = pd.to_datetime(
    features["Date"],
    errors="coerce",
)


market = pd.read_csv(
    MARKET_FILE
)


market["Date"] = pd.to_datetime(
    market["Date"],
    errors="coerce",
)


# ==================================================
# JOIN MODEL + MARKET DATA
# ==================================================

df = features.merge(
    market[
        [
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",

            "HomeOdds",
            "DrawOdds",
            "AwayOdds",

            "MarketHomeProbability",
            "MarketDrawProbability",
            "MarketAwayProbability",

            "OddsSource",
        ]
    ],

    on=[
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
    ],

    how="inner",
)


df = df.sort_values(
    "Date"
).reset_index(
    drop=True
)


seasons = sorted(
    df[
        "Season"
    ].unique()
)


print()
print(
    "Available seasons:",
    seasons,
)


# ==================================================
# WALK-FORWARD VALIDATION
# ==================================================

print()
print("WALK-FORWARD VALIDATION")
print("=======================")


all_results = []


for test_index in range(
    2,
    len(seasons)
):

    test_season = (
        seasons[
            test_index
        ]
    )


    training_seasons = (
        seasons[
            :test_index
        ]
    )


    train = df[
        df[
            "Season"
        ].isin(
            training_seasons
        )
    ].copy()


    test = df[
        df[
            "Season"
        ]
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
    # Train frozen Model 2
    # ----------------------------------------------

    home_model = PoissonRegressor(
        alpha=MODEL_ALPHA,
        max_iter=1000,
    )


    away_model = PoissonRegressor(
        alpha=MODEL_ALPHA,
        max_iter=1000,
    )


    home_model.fit(
        train[
            MODEL_FEATURES
        ],
        train[
            "HomeGoals"
        ],
    )


    away_model.fit(
        train[
            MODEL_FEATURES
        ],
        train[
            "AwayGoals"
        ],
    )


    # ----------------------------------------------
    # Predict expected goals
    # ----------------------------------------------

    predicted_home_goals = (
        home_model.predict(
            test[
                MODEL_FEATURES
            ]
        )
    )


    predicted_away_goals = (
        away_model.predict(
            test[
                MODEL_FEATURES
            ]
        )
    )


    test = test.reset_index(
        drop=True
    )


    # ----------------------------------------------
    # Match probabilities
    # ----------------------------------------------

    for row_number in range(
        len(test)
    ):

        home_xg = (
            predicted_home_goals[
                row_number
            ]
        )

        away_xg = (
            predicted_away_goals[
                row_number
            ]
        )


        probabilities = (
            match_probabilities(
                home_xg,
                away_xg,
            )
        )


        actual = actual_result(
            test.loc[
                row_number,
                "HomeGoals"
            ],
            test.loc[
                row_number,
                "AwayGoals"
            ],
        )


        all_results.append(
            {
                "Season":
                    test_season,

                "Date":
                    test.loc[
                        row_number,
                        "Date"
                    ],

                "HomeTeam":
                    test.loc[
                        row_number,
                        "HomeTeam"
                    ],

                "AwayTeam":
                    test.loc[
                        row_number,
                        "AwayTeam"
                    ],

                "HomeGoals":
                    test.loc[
                        row_number,
                        "HomeGoals"
                    ],

                "AwayGoals":
                    test.loc[
                        row_number,
                        "AwayGoals"
                    ],

                "ActualResult":
                    actual,

                "ExpectedHomeGoals":
                    home_xg,

                "ExpectedAwayGoals":
                    away_xg,

                "ModelHomeProbability":
                    probabilities[0],

                "ModelDrawProbability":
                    probabilities[1],

                "ModelAwayProbability":
                    probabilities[2],

                "MarketHomeProbability":
                    test.loc[
                        row_number,
                        "MarketHomeProbability"
                    ],

                "MarketDrawProbability":
                    test.loc[
                        row_number,
                        "MarketDrawProbability"
                    ],

                "MarketAwayProbability":
                    test.loc[
                        row_number,
                        "MarketAwayProbability"
                    ],

                "HomeOdds":
                    test.loc[
                        row_number,
                        "HomeOdds"
                    ],

                "DrawOdds":
                    test.loc[
                        row_number,
                        "DrawOdds"
                    ],

                "AwayOdds":
                    test.loc[
                        row_number,
                        "AwayOdds"
                    ],

                "OddsSource":
                    test.loc[
                        row_number,
                        "OddsSource"
                    ],
            }
        )


results = pd.DataFrame(
    all_results
)


# ==================================================
# OVERALL METRICS
# ==================================================

actual_results = results[
    "ActualResult"
].tolist()


model_probabilities = results[
    [
        "ModelHomeProbability",
        "ModelDrawProbability",
        "ModelAwayProbability",
    ]
].to_numpy()


market_probabilities = results[
    [
        "MarketHomeProbability",
        "MarketDrawProbability",
        "MarketAwayProbability",
    ]
].to_numpy()


model_metrics = calculate_metrics(
    actual_results,
    model_probabilities,
)


market_metrics = calculate_metrics(
    actual_results,
    market_probabilities,
)


results[
    "ModelPredictedResult"
] = model_metrics[
    "Predictions"
]


results[
    "MarketPredictedResult"
] = market_metrics[
    "Predictions"
]


summary = pd.DataFrame(
    [
        {
            "Source":
                "Model 2",

            "Matches":
                len(results),

            "AccuracyPct":
                model_metrics[
                    "AccuracyPct"
                ],

            "LogLoss":
                model_metrics[
                    "LogLoss"
                ],

            "Brier":
                model_metrics[
                    "Brier"
                ],
        },

        {
            "Source":
                "Market",

            "Matches":
                len(results),

            "AccuracyPct":
                market_metrics[
                    "AccuracyPct"
                ],

            "LogLoss":
                market_metrics[
                    "LogLoss"
                ],

            "Brier":
                market_metrics[
                    "Brier"
                ],
        },
    ]
)


print()
print("FINAL OVERALL COMPARISON")
print("========================")


print(
    summary.to_string(
        index=False,
        formatters={
            "AccuracyPct":
                "{:.2f}".format,

            "LogLoss":
                "{:.4f}".format,

            "Brier":
                "{:.4f}".format,
        },
    )
)


# ==================================================
# PERFORMANCE BY SEASON
# ==================================================

season_rows = []


for season in sorted(
    results[
        "Season"
    ].unique()
):

    subset = results[
        results[
            "Season"
        ]
        ==
        season
    ].copy()


    actual = subset[
        "ActualResult"
    ].tolist()


    for (
        source,
        columns,
    ) in [

        (
            "Model 2",
            [
                "ModelHomeProbability",
                "ModelDrawProbability",
                "ModelAwayProbability",
            ],
        ),

        (
            "Market",
            [
                "MarketHomeProbability",
                "MarketDrawProbability",
                "MarketAwayProbability",
            ],
        ),
    ]:

        probabilities = subset[
            columns
        ].to_numpy()


        metrics = calculate_metrics(
            actual,
            probabilities,
        )


        season_rows.append(
            {
                "Season":
                    season,

                "Source":
                    source,

                "Matches":
                    len(subset),

                "AccuracyPct":
                    metrics[
                        "AccuracyPct"
                    ],

                "LogLoss":
                    metrics[
                        "LogLoss"
                    ],

                "Brier":
                    metrics[
                        "Brier"
                    ],
            }
        )


season_summary = pd.DataFrame(
    season_rows
)


print()
print("FINAL COMPARISON BY SEASON")
print("==========================")


print(
    season_summary.to_string(
        index=False,
        formatters={
            "AccuracyPct":
                "{:.2f}".format,

            "LogLoss":
                "{:.4f}".format,

            "Brier":
                "{:.4f}".format,
        },
    )
)


# ==================================================
# MODEL VS MARKET DIFFERENCES
# ==================================================

results[
    "HomeEdge"
] = (
    results[
        "ModelHomeProbability"
    ]
    -
    results[
        "MarketHomeProbability"
    ]
)


results[
    "DrawEdge"
] = (
    results[
        "ModelDrawProbability"
    ]
    -
    results[
        "MarketDrawProbability"
    ]
)


results[
    "AwayEdge"
] = (
    results[
        "ModelAwayProbability"
    ]
    -
    results[
        "MarketAwayProbability"
    ]
)


# ==================================================
# LARGEST POSITIVE MODEL EDGE
# ==================================================

def best_edge(row):

    edges = {
        "H":
            row[
                "HomeEdge"
            ],

        "D":
            row[
                "DrawEdge"
            ],

        "A":
            row[
                "AwayEdge"
            ],
    }


    selection = max(
        edges,
        key=edges.get,
    )


    return pd.Series(
        {
            "Selection":
                selection,

            "Edge":
                edges[
                    selection
                ],
        }
    )


edge_data = results.apply(
    best_edge,
    axis=1,
)


results = pd.concat(
    [
        results,
        edge_data,
    ],
    axis=1,
)


# ==================================================
# SELECT ODDS
# ==================================================

def get_selected_odds(row):

    if row[
        "Selection"
    ] == "H":

        return row[
            "HomeOdds"
        ]


    if row[
        "Selection"
    ] == "D":

        return row[
            "DrawOdds"
        ]


    return row[
        "AwayOdds"
    ]


results[
    "SelectedOdds"
] = results.apply(
    get_selected_odds,
    axis=1,
)


results[
    "Won"
] = (
    results[
        "Selection"
    ]
    ==
    results[
        "ActualResult"
    ]
)


results[
    "Profit"
] = np.where(
    results[
        "Won"
    ],

    results[
        "SelectedOdds"
    ]
    -
    1,

    -1.0,
)


# ==================================================
# EDGE THRESHOLD SUMMARY
# ==================================================

threshold_rows = []


for threshold in [
    0.00,
    0.02,
    0.05,
    0.075,
    0.10,
]:

    subset = results[
        results[
            "Edge"
        ]
        >= threshold
    ].copy()


    if len(subset) == 0:
        continue


    selections = len(
        subset
    )


    winners = int(
        subset[
            "Won"
        ].sum()
    )


    profit = (
        subset[
            "Profit"
        ].sum()
    )


    threshold_rows.append(
        {
            "MinimumEdgePct":
                threshold * 100,

            "Selections":
                selections,

            "Winners":
                winners,

            "StrikeRatePct":
                winners
                /
                selections
                *
                100,

            "AverageEdgePct":
                subset[
                    "Edge"
                ].mean()
                *
                100,

            "AverageOdds":
                subset[
                    "SelectedOdds"
                ].mean(),

            "Profit":
                profit,

            "ROIPct":
                profit
                /
                selections
                *
                100,
        }
    )


edge_summary = pd.DataFrame(
    threshold_rows
)


print()
print("FINAL MODEL VS MARKET EDGE ANALYSIS")
print("===================================")


print(
    edge_summary.to_string(
        index=False,
        formatters={
            "MinimumEdgePct":
                "{:.1f}".format,

            "StrikeRatePct":
                "{:.2f}".format,

            "AverageEdgePct":
                "{:.2f}".format,

            "AverageOdds":
                "{:.2f}".format,

            "Profit":
                "{:.2f}".format,

            "ROIPct":
                "{:.2f}".format,
        },
    )
)


# ==================================================
# EDGE PERFORMANCE BY SEASON
# ==================================================

print()
print("5%+ EDGE PERFORMANCE BY SEASON")
print("==============================")


five_percent_edges = results[
    results[
        "Edge"
    ]
    >= 0.05
].copy()


for season in sorted(
    five_percent_edges[
        "Season"
    ].unique()
):

    subset = five_percent_edges[
        five_percent_edges[
            "Season"
        ]
        ==
        season
    ]


    selections = len(
        subset
    )


    profit = subset[
        "Profit"
    ].sum()


    print(
        f"{season}: "
        f"{selections} selections, "
        f"profit {profit:.2f}, "
        f"ROI "
        f"{(
            profit
            /
            selections
            *
            100
        ):.2f}%"
    )


# ==================================================
# SAVE EVERYTHING
# ==================================================

Path(
    "data/processed"
).mkdir(
    parents=True,
    exist_ok=True,
)


results.to_csv(
    DETAIL_OUTPUT,
    index=False,
)


summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


season_summary.to_csv(
    SEASON_OUTPUT,
    index=False,
)


results[
    [
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "ActualResult",
        "Selection",
        "Edge",
        "SelectedOdds",
        "Won",
        "Profit",
    ]
].to_csv(
    EDGE_OUTPUT,
    index=False,
)


edge_summary.to_csv(
    EDGE_SUMMARY_OUTPUT,
    index=False,
)


print()
print("FINAL VALIDATION FILES SAVED")
print("============================")


print(
    DETAIL_OUTPUT
)

print(
    SUMMARY_OUTPUT
)

print(
    SEASON_OUTPUT
)

print(
    EDGE_OUTPUT
)

print(
    EDGE_SUMMARY_OUTPUT
)


print()
print("MODEL 2 IS NOW FROZEN")
print("=====================")

print(
    "No further model specification "
    "changes should be made before "
    "production integration."
)