import pickle

import numpy as np
import pandas as pd
from scipy.stats import poisson


MODEL_FILE = (
    "models/poisson_models.pkl"
)


# --------------------------------------------------
# Load model bundle
# --------------------------------------------------

with open(
    MODEL_FILE,
    "rb",
) as file:

    MODEL_BUNDLE = pickle.load(
        file
    )


HOME_MODEL = MODEL_BUNDLE[
    "home_model"
]

AWAY_MODEL = MODEL_BUNDLE[
    "away_model"
]

FEATURE_COLUMNS = MODEL_BUNDLE[
    "features"
]


# --------------------------------------------------
# Predict expected goals
# --------------------------------------------------

def predict_expected_goals(
    feature_values,
):

    feature_df = pd.DataFrame(
        [
            feature_values
        ]
    )

    feature_df = feature_df[
        FEATURE_COLUMNS
    ]

    home_xg = HOME_MODEL.predict(
        feature_df
    )[0]

    away_xg = AWAY_MODEL.predict(
        feature_df
    )[0]

    return (
        float(home_xg),
        float(away_xg),
    )


# --------------------------------------------------
# Calculate result probabilities
# --------------------------------------------------

def calculate_match_probabilities(
    home_xg,
    away_xg,
    max_goals=8,
):

    home_win_probability = 0.0
    draw_probability = 0.0
    away_win_probability = 0.0

    score_probabilities = []


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

            score_probabilities.append(
                {
                    "home_goals": (
                        home_goals
                    ),
                    "away_goals": (
                        away_goals
                    ),
                    "probability": (
                        probability
                    ),
                }
            )

            if home_goals > away_goals:

                home_win_probability += (
                    probability
                )

            elif home_goals == away_goals:

                draw_probability += (
                    probability
                )

            else:

                away_win_probability += (
                    probability
                )


    total = (
        home_win_probability
        +
        draw_probability
        +
        away_win_probability
    )


    home_win_probability /= total
    draw_probability /= total
    away_win_probability /= total


    score_probabilities = sorted(
        score_probabilities,
        key=lambda row: (
            row["probability"]
        ),
        reverse=True,
    )


    return {
        "home_win_probability": (
            home_win_probability
        ),
        "draw_probability": (
            draw_probability
        ),
        "away_win_probability": (
            away_win_probability
        ),
        "most_likely_scores": (
            score_probabilities[:5]
        ),
    }


# --------------------------------------------------
# Combined prediction
# --------------------------------------------------

def predict_match(
    feature_values,
):

    home_xg, away_xg = (
        predict_expected_goals(
            feature_values
        )
    )

    probabilities = (
        calculate_match_probabilities(
            home_xg,
            away_xg,
        )
    )


    return {
        "expected_home_goals": (
            round(home_xg, 2)
        ),

        "expected_away_goals": (
            round(away_xg, 2)
        ),

        "home_win_probability": (
            round(
                probabilities[
                    "home_win_probability"
                ]
                * 100,
                1,
            )
        ),

        "draw_probability": (
            round(
                probabilities[
                    "draw_probability"
                ]
                * 100,
                1,
            )
        ),

        "away_win_probability": (
            round(
                probabilities[
                    "away_win_probability"
                ]
                * 100,
                1,
            )
        ),

        "most_likely_scores": (
            probabilities[
                "most_likely_scores"
            ]
        ),
    }