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

EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "live"
    / "evaluations"
)


# ==================================================
# HISTORICAL MODEL BENCHMARKS
# ==================================================

HISTORICAL_ACCURACY = 52.69
HISTORICAL_LOG_LOSS = 0.9927
HISTORICAL_BRIER = 0.5927


# ==================================================
# FIND LATEST OFFICIAL PREDICTION SNAPSHOT
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
# FIND MATCHING EVALUATION FILE
# ==================================================

def get_evaluation_file(
    season,
    gameweek,
):

    season_slug = (
        str(
            season
        )
        .replace(
            "/",
            "_",
        )
    )

    evaluation_file = (
        EVALUATION_DIR
        /
        (
            f"{season_slug}_gw"
            f"{int(gameweek):02d}_evaluation.csv"
        )
    )

    if evaluation_file.exists():

        return evaluation_file

    return None


# ==================================================
# LOAD CSV
# ==================================================

@st.cache_data
def load_csv(
    file_path_string,
):

    return pd.read_csv(
        Path(
            file_path_string
        )
    )


# ==================================================
# FORMAT TIMESTAMP
# ==================================================

def format_prediction_timestamp(
    timestamp,
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
# BOOLEAN HELPER
# ==================================================

def to_bool(
    value,
):

    if isinstance(
        value,
        bool,
    ):

        return value

    return (
        str(
            value
        )
        .strip()
        .lower()
        in [
            "true",
            "1",
            "yes",
        ]
    )


# ==================================================
# RENDER PREDICTION-ONLY FIXTURE
# ==================================================

def render_prediction_fixture(
    fixture,
):

    home_team = fixture[
        "HomeTeam"
    ]

    away_team = fixture[
        "AwayTeam"
    ]


    st.markdown(
        f"### {home_team} v {away_team}"
    )


    outcome_col, score_col = (
        st.columns(
            2
        )
    )


    outcome_col.metric(
        "Predicted outcome",
        fixture[
            "PredictedResult"
        ],
    )


    score_col.metric(
        "Most likely scoreline",
        fixture[
            "MostLikelyScore"
        ],
    )


    probability_columns = (
        st.columns(
            3
        )
    )


    probability_columns[
        0
    ].metric(
        f"{home_team} win",
        (
            f"{float(fixture['HomeWinProbability']):.1f}%"
        ),
    )


    probability_columns[
        1
    ].metric(
        "Draw",
        (
            f"{float(fixture['DrawProbability']):.1f}%"
        ),
    )


    probability_columns[
        2
    ].metric(
        f"{away_team} win",
        (
            f"{float(fixture['AwayWinProbability']):.1f}%"
        ),
    )


    probability_df = pd.DataFrame(
        {
            "Outcome": [
                f"{home_team} win",
                "Draw",
                f"{away_team} win",
            ],
            "Probability": [
                float(
                    fixture[
                        "HomeWinProbability"
                    ]
                ),
                float(
                    fixture[
                        "DrawProbability"
                    ]
                ),
                float(
                    fixture[
                        "AwayWinProbability"
                    ]
                ),
            ],
        }
    ).set_index(
        "Outcome"
    )


    st.bar_chart(
        probability_df,
        use_container_width=True,
    )


    xg_columns = (
        st.columns(
            2
        )
    )


    xg_columns[
        0
    ].metric(
        f"{home_team} expected goals",
        (
            f"{float(fixture['ExpectedHomeGoals']):.2f}"
        ),
    )


    xg_columns[
        1
    ].metric(
        f"{away_team} expected goals",
        (
            f"{float(fixture['ExpectedAwayGoals']):.2f}"
        ),
    )


    if to_bool(
        fixture.get(
            "ColdStartUsed",
            False,
        )
    ):

        st.info(
            "Promoted-team cold-start prior used for "
            f"{fixture.get('ColdStartTeams', '')}."
        )


    st.divider()


# ==================================================
# RENDER COMPLETED FIXTURE
# ==================================================

def render_evaluated_fixture(
    fixture,
):

    home_team = fixture[
        "HomeTeam"
    ]

    away_team = fixture[
        "AwayTeam"
    ]


    correct = to_bool(
        fixture[
            "OutcomeCorrect"
        ]
    )


    symbol = (
        "✅"
        if correct
        else
        "❌"
    )


    st.markdown(
        f"### {home_team} v {away_team}"
    )


    status_columns = (
        st.columns(
            4
        )
    )


    status_columns[
        0
    ].metric(
        "Predicted outcome",
        fixture[
            "PredictedResult"
        ],
    )


    status_columns[
        1
    ].metric(
        "Actual outcome",
        fixture[
            "ActualResult"
        ],
    )


    status_columns[
        2
    ].metric(
        "Predicted score",
        fixture[
            "MostLikelyScore"
        ],
    )


    status_columns[
        3
    ].metric(
        "Actual score",
        (
            f"{fixture['ActualScore']} {symbol}"
        ),
    )


    probability_columns = (
        st.columns(
            3
        )
    )


    probability_columns[
        0
    ].metric(
        f"{home_team} win",
        (
            f"{float(fixture['HomeWinProbability']):.1f}%"
        ),
    )


    probability_columns[
        1
    ].metric(
        "Draw",
        (
            f"{float(fixture['DrawProbability']):.1f}%"
        ),
    )


    probability_columns[
        2
    ].metric(
        f"{away_team} win",
        (
            f"{float(fixture['AwayWinProbability']):.1f}%"
        ),
    )


    detail_columns = (
        st.columns(
            3
        )
    )


    detail_columns[
        0
    ].metric(
        "Actual outcome probability",
        (
            f"{float(fixture['ActualOutcomeProbability']) * 100:.1f}%"
        ),
    )


    detail_columns[
        1
    ].metric(
        "Fixture Log Loss",
        (
            f"{float(fixture['LogLoss']):.4f}"
        ),
    )


    detail_columns[
        2
    ].metric(
        "Fixture Brier",
        (
            f"{float(fixture['Brier']):.4f}"
        ),
    )


    with st.expander(
        "Goal prediction detail"
    ):

        goal_columns = (
            st.columns(
                4
            )
        )


        goal_columns[
            0
        ].metric(
            f"{home_team} expected goals",
            (
                f"{float(fixture['ExpectedHomeGoals']):.2f}"
            ),
        )


        goal_columns[
            1
        ].metric(
            f"{home_team} actual goals",
            int(
                fixture[
                    "ActualHomeGoals"
                ]
            ),
        )


        goal_columns[
            2
        ].metric(
            f"{away_team} expected goals",
            (
                f"{float(fixture['ExpectedAwayGoals']):.2f}"
            ),
        )


        goal_columns[
            3
        ].metric(
            f"{away_team} actual goals",
            int(
                fixture[
                    "ActualAwayGoals"
                ]
            ),
        )


    if to_bool(
        fixture.get(
            "ColdStartUsed",
            False,
        )
    ):

        st.info(
            "Promoted-team cold-start prior used for "
            f"{fixture.get('ColdStartTeams', '')}."
        )


    st.divider()


# ==================================================
# RENDER COMPLETED GAMEWEEK
# ==================================================

def render_completed_gameweek(
    evaluation,
    season,
    gameweek,
    prediction_timestamp,
):

    matches = len(
        evaluation
    )


    correct = int(
        evaluation[
            "OutcomeCorrect"
        ]
        .apply(
            to_bool
        )
        .sum()
    )


    accuracy = (
        correct
        /
        matches
        *
        100
    )


    exact_scores = int(
        evaluation[
            "ExactScoreCorrect"
        ]
        .apply(
            to_bool
        )
        .sum()
    )


    log_loss = float(
        evaluation[
            "LogLoss"
        ]
        .mean()
    )


    brier = float(
        evaluation[
            "Brier"
        ]
        .mean()
    )


    home_goal_mae = float(
        evaluation[
            "HomeGoalAbsoluteError"
        ]
        .mean()
    )


    away_goal_mae = float(
        evaluation[
            "AwayGoalAbsoluteError"
        ]
        .mean()
    )


    total_goal_mae = float(
        evaluation[
            "TotalGoalsAbsoluteError"
        ]
        .mean()
    )


    st.markdown(
        "## 🔴 2026/27 Live Model Experiment"
    )


    st.success(
        f"Gameweek {gameweek} completed and evaluated."
    )


    st.caption(
        "The predictions below were frozen before "
        "the matches were played and have not been "
        "retrospectively changed."
    )


    st.caption(
        f"Prediction snapshot: "
        f"{prediction_timestamp}"
    )


    # Verified retrospective comparison is available for GW8 only.
    # Do not display these historical figures as results for later gameweeks.
    if int(gameweek) == 8 and str(season) == "2026/27":
        st.markdown("### Forecast model comparison | GW8")
        st.caption(
            "Verified frozen-forecast scorecard, 10 completed matches. "
            "Lower Log Loss and Brier are better. Source: docs/gameweeks/GW08.md."
        )
        st.dataframe(
            pd.DataFrame([
                {"Method": "Market only", "Role": "External benchmark",
                 "Correct": "4/10", "Accuracy": "40%", "Log Loss": 1.1303, "Brier": 0.6918},
                {"Method": "40/60 blend", "Role": "Provisional leader",
                 "Correct": "3/10", "Accuracy": "30%", "Log Loss": 1.2021, "Brier": 0.7446},
                {"Method": "Adaptive confidence", "Role": "Experimental challenger",
                 "Correct": "3/10", "Accuracy": "30%", "Log Loss": 1.1981, "Brier": 0.7415},
                {"Method": "Model 2", "Role": "Official frozen benchmark",
                 "Correct": "3/10", "Accuracy": "30%", "Log Loss": 1.3439, "Brier": 0.8396},
            ]),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("### Comparable prospective performance | GW6–GW8")
        st.dataframe(
            pd.DataFrame([
                {"Method": "40/60 blend", "Correct": "14/30", "Accuracy": "46.7%",
                 "Log Loss": 1.0309, "Brier": 0.6225},
                {"Method": "Model 2", "Correct": "11/30", "Accuracy": "36.7%",
                 "Log Loss": 1.1560, "Brier": 0.7125},
            ]),
            use_container_width=True,
            hide_index=True,
        )
        st.info(
            "40/60 is the leading provisional candidate over 30 comparable matches, "
            "not a formally promoted production champion. Model 2 remains the official "
            "frozen benchmark. GW8 alone favoured market-only; this is not proof of "
            "persistent model superiority."
        )
        st.markdown("### Model 2 benchmark | GW8 detailed evaluation")

    # ==================================================
    # MAIN KPI CARDS
    # ==================================================

    metric_columns = (
        st.columns(
            4
        )
    )


    metric_columns[
        0
    ].metric(
        "Model 2 accuracy",
        f"{accuracy:.1f}%",
        (
            f"{accuracy - HISTORICAL_ACCURACY:+.2f}pp "
            "vs historical"
        ),
    )


    metric_columns[
        1
    ].metric(
        "Model 2 Log Loss",
        f"{log_loss:.4f}",
        (
            f"{log_loss - HISTORICAL_LOG_LOSS:+.4f}"
        ),
        delta_color="inverse",
    )


    metric_columns[
        2
    ].metric(
        "Model 2 Brier",
        f"{brier:.4f}",
        (
            f"{brier - HISTORICAL_BRIER:+.4f}"
        ),
        delta_color="inverse",
    )


    metric_columns[
        3
    ].metric(
        "Model 2 exact scores",
        (
            f"{exact_scores}/{matches}"
        ),
    )


    # ==================================================
    # HISTORICAL COMPARISON
    # ==================================================

    st.markdown(
        "### Historical comparison"
    )


    comparison_df = pd.DataFrame(
        {
            "Metric": [
                "1X2 Accuracy",
                "Log Loss",
                "Brier Score",
            ],

            "GW Result": [
                f"{accuracy:.1f}%",
                f"{log_loss:.4f}",
                f"{brier:.4f}",
            ],

            "Historical Model 2": [
                f"{HISTORICAL_ACCURACY:.2f}%",
                f"{HISTORICAL_LOG_LOSS:.4f}",
                f"{HISTORICAL_BRIER:.4f}",
            ],
        }
    )


    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )


    st.caption(
    f"GW{gameweek} contains {matches} matches, "
    "so Gameweek-level differences from historical "
    "performance should not be treated as statistically "
    "meaningful in isolation."
    )         


    # ==================================================
    # FIXTURE SUMMARY
    # ==================================================

    st.markdown(
        "### Prediction vs actual"
    )


    summary_table = (
        evaluation[
            [
                "HomeTeam",
                "AwayTeam",
                "PredictedResult",
                "MostLikelyScore",
                "ActualScore",
                "OutcomeCorrect",
                "ExactScoreCorrect",
            ]
        ]
        .copy()
    )


    summary_table[
        "Fixture"
    ] = (
        summary_table["HomeTeam"]
        + " v "
        + summary_table["AwayTeam"]
    )


    summary_table[
        "Outcome correct"
    ] = summary_table[
        "OutcomeCorrect"
    ].apply(
        lambda value:
            "✅"
            if to_bool(value)
            else "❌"
    )


    summary_table[
        "Exact score"
    ] = summary_table[
        "ExactScoreCorrect"
    ].apply(
        lambda value:
            "✅"
            if to_bool(value)
            else "❌"
    )


    summary_table = summary_table[
        [
            "Fixture",
            "PredictedResult",
            "MostLikelyScore",
            "ActualScore",
            "Outcome correct",
            "Exact score",
        ]
    ]


    summary_table.columns = [
        "Fixture",
        "Predicted outcome",
        "Predicted score",
        "Actual score",
        "Outcome correct?",
        "Exact score?",
    ]


    st.dataframe(
        summary_table,
        use_container_width=True,
        hide_index=True,
    )


    # ==================================================
    # GOAL ERROR
    # ==================================================

    st.markdown(
        "### Goal prediction error"
    )


    goal_columns = (
        st.columns(
            3
        )
    )


    goal_columns[
        0
    ].metric(
        "Home goals MAE",
        f"{home_goal_mae:.3f}",
    )


    goal_columns[
        1
    ].metric(
        "Away goals MAE",
        f"{away_goal_mae:.3f}",
    )


    goal_columns[
        2
    ].metric(
        "Total goals MAE",
        f"{total_goal_mae:.3f}",
    )


    # ==================================================
    # DIAGNOSTICS
    # ==================================================

    worst = (
        evaluation
        .sort_values(
            "ActualOutcomeProbability",
            ascending=True,
        )
        .iloc[
            0
        ]
    )


    best = (
        evaluation
        .sort_values(
            "ActualOutcomeProbability",
            ascending=False,
        )
        .iloc[
            0
        ]
    )


    st.markdown(
        "### Diagnostics"
    )


    diagnostic_columns = (
        st.columns(
            2
        )
    )


    with diagnostic_columns[
        0
    ]:

        st.warning(
            "**Biggest model surprise**\n\n"
            f"{worst['HomeTeam']} v "
            f"{worst['AwayTeam']}\n\n"
            f"Actual outcome: "
            f"{worst['ActualResult']}\n\n"
            f"Probability assigned: "
            f"{float(worst['ActualOutcomeProbability']) * 100:.1f}%"
        )


    with diagnostic_columns[
        1
    ]:

        st.success(
            "**Highest-confidence success**\n\n"
            f"{best['HomeTeam']} v "
            f"{best['AwayTeam']}\n\n"
            f"Actual outcome: "
            f"{best['ActualResult']}\n\n"
            f"Probability assigned: "
            f"{float(best['ActualOutcomeProbability']) * 100:.1f}%"
        )


    # ==================================================
    # DRAW / 1-1 DIAGNOSTIC
    # ==================================================

    predicted_draws = int(
        (
            evaluation[
                "PredictedResult"
            ]
            ==
            "Draw"
        )
        .sum()
    )


    actual_draws = int(
        (
            evaluation[
                "ActualResult"
            ]
            ==
            "Draw"
        )
        .sum()
    )


    one_one_predictions = int(
        (
            evaluation[
                "MostLikelyScore"
            ]
            .astype(
                str
            )
            ==
            "1-1"
        )
        .sum()
    )


    actual_one_one = int(
        (
            evaluation[
                "ActualScore"
            ]
            .astype(
                str
            )
            ==
            "1-1"
        )
        .sum()
    )


    st.markdown(
        "### Draw and scoreline diagnostic"
    )


    draw_columns = (
        st.columns(
            4
        )
    )


    draw_columns[
        0
    ].metric(
        "Predicted 1X2 draws",
        predicted_draws,
    )


    draw_columns[
        1
    ].metric(
        "Actual draws",
        actual_draws,
    )


    draw_columns[
        2
    ].metric(
        "1-1 modal predictions",
        one_one_predictions,
    )


    draw_columns[
        3
    ].metric(
        "Actual 1-1 results",
        actual_one_one,
    )


    # ==================================================
    # COLD START PERFORMANCE
    # ==================================================

    if (
        "ColdStartUsed"
        in evaluation.columns
    ):

        cold_start = (
            evaluation[
                evaluation[
                    "ColdStartUsed"
                ]
                .apply(
                    to_bool
                )
            ]
        )


        if not cold_start.empty:

            cold_matches = len(
                cold_start
            )


            cold_correct = int(
                cold_start[
                    "OutcomeCorrect"
                ]
                .apply(
                    to_bool
                )
                .sum()
            )


            cold_accuracy = (
                cold_correct
                /
                cold_matches
                *
                100
            )


            cold_log_loss = float(
                cold_start[
                    "LogLoss"
                ]
                .mean()
            )


            cold_brier = float(
                cold_start[
                    "Brier"
                ]
                .mean()
            )


            st.markdown(
                "### Promoted-team cold-start"
            )


            cold_columns = (
                st.columns(
                    4
                )
            )


            cold_columns[
                0
            ].metric(
                "Fixtures",
                cold_matches,
            )


            cold_columns[
                1
            ].metric(
                "Accuracy",
                f"{cold_accuracy:.1f}%",
            )


            cold_columns[
                2
            ].metric(
                "Log Loss",
                f"{cold_log_loss:.4f}",
            )


            cold_columns[
                3
            ].metric(
                "Brier",
                f"{cold_brier:.4f}",
            )


    # ==================================================
    # WHAT WE LEARNED
    # ==================================================

    st.markdown(
        f"### Model 2 diagnostics: completed GW{gameweek}"
    )

    st.markdown(
        f"""
These are ongoing diagnostic questions, not confirmed model defects.

1. **Expected-goals compression**  
   Monitor whether predicted scoring levels are too close to the league average.

2. **1-1 modal concentration**  
   In GW{gameweek}, {one_one_predictions} of {matches} Model 2 predictions had
   1-1 as their most likely scoreline, compared with {actual_one_one}
   actual 1-1 results. These figures cover this gameweek only.

3. **Promoted-team cold-start**  
   Continue monitoring Coventry, Hull and Ipswich to assess whether
   their initial strength estimates were overly conservative.

4. **Upset calibration**  
   Monitor lower-probability outcomes across completed gameweeks rather
   than drawing conclusions from a single result.
"""
    )

    st.info(
        "Model 2 remains the frozen official benchmark for recorded forecasts. "
        "The initial GW1–GW5 monitoring window has ended. "
        "The 40/60 market blend is a provisional challenger, not formally "
        "promoted. Retrospective changes to frozen predictions are prohibited."
    )


    # ==================================================
    # FIXTURE DETAIL
    # ==================================================

    with st.expander(
        "View all fixture-level evaluation details"
    ):

        for _, fixture in (
            evaluation.iterrows()
        ):

            render_evaluated_fixture(
                fixture
            )


# ==================================================
# RENDER ACTIVE GAMEWEEK
# ==================================================

def render_active_gameweek(
    predictions,
    season,
    gameweek,
    prediction_timestamp,
):

    st.markdown(
        "## 🔴 Live 2026/27 Predictions"
    )


    st.caption(
        "Frozen pre-match predictions. "
        "These probabilities are read from the "
        "official gameweek snapshot and are not "
        "regenerated when the app loads."
    )


    summary_columns = (
        st.columns(
            4
        )
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
        .apply(
            to_bool
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


    with st.expander(
        "View fixture prediction details"
    ):

        for _, fixture in (
            predictions.iterrows()
        ):

            render_prediction_fixture(
                fixture
            )


# ==================================================
# MAIN RENDERER
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


    predictions = load_csv(
        str(
            prediction_file
        )
    )


    if predictions.empty:

        st.warning(
            "The latest prediction snapshot "
            "contains no fixtures."
        )

        return


    season = str(
        predictions.iloc[
            0
        ][
            "Season"
        ]
    )


    gameweek = int(
        predictions.iloc[
            0
        ][
            "Gameweek"
        ]
    )


    prediction_timestamp = (
        format_prediction_timestamp(
            predictions.iloc[
                0
            ][
                "PredictionTimestamp"
            ]
        )
    )


    evaluation_file = (
        get_evaluation_file(
            season,
            gameweek,
        )
    )


    if evaluation_file is not None:

        evaluation = (
            load_csv(
                str(
                    evaluation_file
                )
            )
        )


        render_completed_gameweek(
            evaluation,
            season,
            gameweek,
            prediction_timestamp,
        )


    else:

        render_active_gameweek(
            predictions,
            season,
            gameweek,
            prediction_timestamp,
        )


    st.caption(
        "Football Copilot probabilities are "
        "statistical estimates and are not "
        "betting recommendations."
    )