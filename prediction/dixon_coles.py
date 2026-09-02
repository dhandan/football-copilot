from __future__ import annotations

import numpy as np
from scipy.stats import poisson


# --------------------------------------------------
# Dixon-Coles low-score adjustment
# --------------------------------------------------

def dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    home_xg: float,
    away_xg: float,
    rho: float,
) -> float:
    """
    Dixon-Coles correction factor for the four
    low-scoring outcomes:

    0-0
    0-1
    1-0
    1-1

    All other scorelines receive no adjustment.
    """

    if home_goals == 0 and away_goals == 0:
        return 1.0 - (
            home_xg
            * away_xg
            * rho
        )

    if home_goals == 0 and away_goals == 1:
        return 1.0 + (
            home_xg
            * rho
        )

    if home_goals == 1 and away_goals == 0:
        return 1.0 + (
            away_xg
            * rho
        )

    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho

    return 1.0


# --------------------------------------------------
# Score probability matrix
# --------------------------------------------------

def score_probability_matrix(
    home_xg: float,
    away_xg: float,
    rho: float,
    max_goals: int = 8,
) -> np.ndarray:
    """
    Build a Dixon-Coles adjusted score probability
    matrix.

    Rows represent home goals.
    Columns represent away goals.
    """

    matrix = np.zeros(
        (
            max_goals + 1,
            max_goals + 1,
        ),
        dtype=float,
    )

    for home_goals in range(
        max_goals + 1
    ):

        home_probability = poisson.pmf(
            home_goals,
            home_xg,
        )

        for away_goals in range(
            max_goals + 1
        ):

            away_probability = poisson.pmf(
                away_goals,
                away_xg,
            )

            independent_probability = (
                home_probability
                * away_probability
            )

            adjustment = dixon_coles_tau(
                home_goals=home_goals,
                away_goals=away_goals,
                home_xg=home_xg,
                away_xg=away_xg,
                rho=rho,
            )

            adjusted_probability = (
                independent_probability
                * adjustment
            )

            matrix[
                home_goals,
                away_goals,
            ] = max(
                adjusted_probability,
                0.0,
            )

    total_probability = matrix.sum()

    if total_probability <= 0:
        raise ValueError(
            "Dixon-Coles score matrix has "
            "non-positive total probability."
        )

    matrix /= total_probability

    return matrix


# --------------------------------------------------
# 1X2 probabilities
# --------------------------------------------------

def match_probabilities(
    home_xg: float,
    away_xg: float,
    rho: float,
    max_goals: int = 8,
) -> dict:
    """
    Convert a Dixon-Coles score matrix into
    Home / Draw / Away probabilities and return
    modal-score diagnostics.
    """

    matrix = score_probability_matrix(
        home_xg=home_xg,
        away_xg=away_xg,
        rho=rho,
        max_goals=max_goals,
    )

    home_probability = float(
        np.tril(
            matrix,
            k=-1,
        ).sum()
    )

    draw_probability = float(
        np.trace(
            matrix
        )
    )

    away_probability = float(
        np.triu(
            matrix,
            k=1,
        ).sum()
    )

    total = (
        home_probability
        + draw_probability
        + away_probability
    )

    home_probability /= total
    draw_probability /= total
    away_probability /= total

    modal_index = np.unravel_index(
        np.argmax(matrix),
        matrix.shape,
    )

    modal_home_goals = int(
        modal_index[0]
    )

    modal_away_goals = int(
        modal_index[1]
    )

    modal_score = (
        f"{modal_home_goals}-"
        f"{modal_away_goals}"
    )

    return {
        "home_probability":
            home_probability,

        "draw_probability":
            draw_probability,

        "away_probability":
            away_probability,

        "modal_home_goals":
            modal_home_goals,

        "modal_away_goals":
            modal_away_goals,

        "modal_score":
            modal_score,

        "modal_probability":
            float(
                matrix[
                    modal_home_goals,
                    modal_away_goals,
                ]
            ),

        "score_matrix":
            matrix,
    }


# --------------------------------------------------
# Independent Poisson reference
# --------------------------------------------------

def independent_poisson_probabilities(
    home_xg: float,
    away_xg: float,
    max_goals: int = 8,
) -> dict:
    """
    Reference calculation equivalent to the
    independent Poisson score-matrix approach.

    rho=0 makes the Dixon-Coles adjustment neutral.
    """

    return match_probabilities(
        home_xg=home_xg,
        away_xg=away_xg,
        rho=0.0,
        max_goals=max_goals,
    )