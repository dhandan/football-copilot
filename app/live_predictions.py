from pathlib import Path

import pandas as pd
import streamlit as st


# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "predictions"
)


# ==================================================
# FIND LATEST OFFICIAL SNAPSHOT
# ==================================================

def get_latest_prediction_file():

    prediction_files = [
        path
        for path in PREDICTION_DIR.glob(
            "*_predictions.csv"
        )
        if "superseded" not in path.name.lower()
    ]

    if not prediction_files:
        return None

    prediction_files = sorted(
        prediction_files,
        key=lambda path:
            path.stat().st_mtime,
        reverse=True,
    )

    return prediction_files[0]


# ==================================================
# LOAD PREDICTIONS
# ==================================================

@st.cache_data
def load_live_predictions(
    file_path_string
):

    file_path = Path(
        file_path_string
    )

    return pd.read_csv(
        file_path
    )


# ==================================================
# FORMAT TIMESTAMP
# ==================================================

def format_prediction_timestamp(
    timestamp
):

    try:

        parsed = pd.to_datetime(
            timestamp
        )

        return parsed.strftime(
            "%d %b %Y %H:%M"
        )

    except Exception:

        return str(
            timestamp
        )


# ==================================================
# RENDER ONE FIXTURE
# ==================================================

def render_fixture(
    fixture
):

    home_team = fixture[
        "HomeTeam"
    ]

    away_team = fixture[
        "AwayTeam"
    ]

    predicted_result = fixture[
        "PredictedResult"
    ]

    most_likely_score = fixture[
        "MostLikelyScore"
    ]

    home_probability = float(
        fixture[
            "HomeWinProbability"
        ]
    )

    draw_probability = float(
        fixture[
            "DrawProbability"
        ]
    )

    away_probability = float(
        fixture[
            "AwayWinProbability"
        ]
    )

    expected_home_goals = float(
        fixture[
            "ExpectedHomeGoals"
        ]
    )

    expected_away_goals = float(
        fixture[
            "ExpectedAwayGoals"
        ]
    )


    st.markdown(
        f"### {home_team} v {away_team}"
    )


    outcome_col, score_col = st.columns(
        2
    )


    outcome_col.metric(
        "Predicted outcome",
        predicted_result,
    )


    score_col.metric(
        "Most likely scoreline",
        most_likely_score,
    )


    probability_columns = st.columns(
        3
    )


    probability_columns[
        0
    ].metric(
        f"{home_team} win",
        f"{home_probability:.1f}%",
    )


    probability_columns[
        1
    ].metric(
        "Draw",
        f"{draw_probability:.1f}%",
    )


    probability_columns[
        2
    ].metric(
        f"{away_team} win",
        f"{away_probability:.1f}%",
    )


    probability_df = pd.DataFrame(
        {
            "Outcome": [
                f"{home_team} win",
                "Draw",
                f"{away_team} win",
            ],
            "Probability": [
                home_probability,
                draw_probability,
                away_probability,
            ],
        }
    ).set_index(
        "Outcome"
    )


    st.bar_chart(
        probability_df,
        use_container_width=True,
    )


    xg_columns = st.columns(
        2
    )


    xg_columns[
        0
    ].metric(
        f"{home_team} expected goals",
        f"{expected_home_goals:.2f}",
    )


    xg_columns[
        1
    ].metric(
        f"{away_team} expected goals",
        f"{expected_away_goals:.2f}",
    )


    cold_start_used = bool(
        fixture.get(
            "ColdStartUsed",
            False,
        )
    )


    if cold_start_used:

        cold_start_team = fixture.get(
            "ColdStartTeams",
            ""
        )

        st.info(
            "Promoted-team cold-start prior used "
            f"for {cold_start_team}."
        )


    with st.expander(
        "Prediction details"
    ):

        st.write(
            f"**Fixture ID:** "
            f"{fixture['FixtureId']}"
        )

        st.write(
            f"**Fixture date:** "
            f"{fixture['FixtureDate']}"
        )

        st.write(
            f"**Kick-off:** "
            f"{fixture['FixtureTime']}"
        )

        st.write(
            f"**Model:** "
            f"{fixture['ModelVersion']}"
        )

        st.write(
            f"**Feature season:** "
            f"{fixture['FeatureSeason']}"
        )

        if cold_start_used:

            st.write(
                "**Cold-start method:** "
                f"{fixture.get('ColdStartMethod', '')}"
            )


    st.divider()


# ==================================================
# MAIN LIVE PREDICTIONS RENDERER
# ==================================================

def render_live_predictions():

    prediction_file = (
        get_latest_prediction_file()
    )


    if prediction_file is None:

        st.info(
            "No official live prediction "
            "snapshot is available yet."
        )

        return


    predictions = (
        load_live_predictions(
            str(
                prediction_file
            )
        )
    )


    if predictions.empty:

        st.warning(
            "The latest prediction snapshot "
            "contains no fixtures."
        )

        return


    season = str(
        predictions.iloc[0][
            "Season"
        ]
    )

    gameweek = int(
        predictions.iloc[0][
            "Gameweek"
        ]
    )

    prediction_timestamp = (
        format_prediction_timestamp(
            predictions.iloc[0][
                "PredictionTimestamp"
            ]
        )
    )


    st.markdown(
        "## 🔴 Live 2026/27 Predictions"
    )


    st.caption(
        "Frozen pre-match predictions. "
        "The probabilities displayed here are "
        "read from the official gameweek snapshot "
        "and are not regenerated when the app loads."
    )


    summary_columns = st.columns(
        4
    )


    summary_columns[
        0
    ].metric(
        "Season",
        season,
    )


    summary_columns[
        1
    ].metric(
        "Gameweek",
        gameweek,
    )


    summary_columns[
        2
    ].metric(
        "Fixtures",
        len(
            predictions
        ),
    )


    cold_start_count = int(
        predictions[
            "ColdStartUsed"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
        .sum()
    )


    summary_columns[
        3
    ].metric(
        "Cold starts",
        cold_start_count,
    )


    st.caption(
        f"Official snapshot created: "
        f"{prediction_timestamp}"
    )


    st.markdown(
        "### Gameweek summary"
    )


    summary_table = (
        predictions[
            [
                "HomeTeam",
                "AwayTeam",
                "PredictedResult",
                "MostLikelyScore",
                "HomeWinProbability",
                "DrawProbability",
                "AwayWinProbability",
            ]
        ]
        .copy()
    )


    summary_table.columns = [
        "Home",
        "Away",
        "Predicted outcome",
        "Most likely score",
        "Home %",
        "Draw %",
        "Away %",
    ]


    st.dataframe(
        summary_table,
        use_container_width=True,
        hide_index=True,
    )


    st.markdown(
        "### Fixture detail"
    )


    for _, fixture in (
        predictions.iterrows()
    ):

        render_fixture(
            fixture
        )


    st.caption(
        "Football Copilot probabilities are "
        "statistical estimates and are not "
        "betting recommendations."
    )