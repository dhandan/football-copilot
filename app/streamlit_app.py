from pathlib import Path
import sys
import json
import re

import pandas as pd
import streamlit as st
from ollama import chat


# ==================================================
# PROJECT SETUP
# ==================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ==================================================
# IMPORT LIVE PREDICTIONS
# ==================================================

from app.live_predictions import (
    render_live_predictions,
)


# ==================================================
# IMPORT FOOTBALL COPILOT TOOLS
# ==================================================

from agent.football_tools import (
    team_record,
    recent_form,
    form_summary,
    league_table,
    home_away_record,
    head_to_head,
    team_comparison,
    fixture_prediction,
)


# ==================================================
# APPLICATION CONFIGURATION
# ==================================================

MODEL = "qwen3:4b"

PRODUCTION_MODEL_NAME = "Model 2"

PRODUCTION_MODEL_TYPE = (
    "Poisson regression"
)

VALIDATION_MATCHES = 1061

MODEL_ACCURACY = 52.69
MODEL_LOG_LOSS = 0.9927
MODEL_BRIER = 0.5927

MARKET_ACCURACY = 54.48
MARKET_LOG_LOSS = 0.9641
MARKET_BRIER = 0.5730


# ==================================================
# SYSTEM PROMPT
# ==================================================

SYSTEM_PROMPT = """
You are Football Copilot, a conversational Premier League
analytics and statistical prediction assistant.

CRITICAL DATA AUTHORITY RULE:

Football Copilot tools are the ONLY source of truth for
football statistics, seasons, results, standings, team
performance and statistical match predictions.

Your own training knowledge is NOT a valid source for those
facts.

The local Football Copilot database contains Premier League
data through the 2025/26 season.

2025/26 IS a valid historical season in this application.

Never decide that a season does not exist before querying a
Football Copilot tool.

If your prior knowledge conflicts with a tool result, the
TOOL RESULT is correct for this application.


TOOL ROUTING:

Use team_record for a team's performance in a specific season.

Use form_summary for recent performance summaries.

Use recent_form when the user asks to see individual recent
matches or results.

Use league_table for league standings.

Use home_away_record for home-versus-away performance.

Use head_to_head for meetings between two teams.

Use team_comparison to compare two teams during a season.

Use fixture_prediction for predictions, probabilities,
expected goals, likely outcomes and likely scorelines.


PREDICTION RULES:

The production model is frozen Model 2 using Poisson
regression.

Never invent or alter probabilities returned by the tool.

Predictions are statistical estimates, not guarantees.

For the 2026/27 season some newly promoted teams can use
the Football Copilot promoted-team cold-start framework.

The prediction tool determines whether that framework is
required.


MARKET RULES:

The application does not currently contain live bookmaker
prices for future fixtures.

Never invent bookmaker odds.

Historical market probabilities outperformed Model 2 overall.

Historical model-market disagreements did not produce a
persistent positive-return signal.


CONVERSATION RULES:

Use conversation history for follow-up questions.

Never invent football statistics, probabilities, injuries,
line-ups, player availability, tactics or bookmaker odds.

Explain returned tool results clearly and concisely.

Do not output JSON or a function-call description to the user
when a tool should be executed.
"""


# ==================================================
# TOOL REGISTRY
# ==================================================

TOOLS = [
    team_record,
    recent_form,
    form_summary,
    league_table,
    home_away_record,
    head_to_head,
    team_comparison,
    fixture_prediction,
]


TOOL_FUNCTIONS = {

    "team_record":
        team_record,

    "recent_form":
        recent_form,

    "form_summary":
        form_summary,

    "league_table":
        league_table,

    "home_away_record":
        home_away_record,

    "head_to_head":
        head_to_head,

    "team_comparison":
        team_comparison,

    "fixture_prediction":
        fixture_prediction,
}


# ==================================================
# LOAD VALID TEAM NAMES
# ==================================================

def load_team_names():

    teams = set()


    # --------------------------------------------------
    # Historical Premier League teams
    # --------------------------------------------------

    match_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "matches_clean.csv"
    )


    try:

        df = pd.read_csv(
            match_file,
            usecols=[
                "HomeTeam",
                "AwayTeam",
            ],
        )


        teams.update(
            df[
                "HomeTeam"
            ].dropna()
        )


        teams.update(
            df[
                "AwayTeam"
            ].dropna()
        )


    except Exception:

        pass


    # --------------------------------------------------
    # Live fixture teams
    # --------------------------------------------------

    fixture_directory = (
        PROJECT_ROOT
        / "data"
        / "live"
        / "fixtures"
    )


    try:

        fixture_files = sorted(
            fixture_directory.glob(
                "fixtures_*.csv"
            ),
            key=lambda path:
                path.stat().st_mtime,
            reverse=True,
        )


        if fixture_files:

            fixture_df = pd.read_csv(
                fixture_files[0]
            )


            if (
                "HomeTeam"
                in fixture_df.columns
            ):

                teams.update(
                    fixture_df[
                        "HomeTeam"
                    ].dropna()
                )


            if (
                "AwayTeam"
                in fixture_df.columns
            ):

                teams.update(
                    fixture_df[
                        "AwayTeam"
                    ].dropna()
                )


    except Exception:

        pass


    return sorted(
        teams,
        key=len,
        reverse=True,
    )


TEAM_NAMES = (
    load_team_names()
)


# ==================================================
# TEXT / ENTITY HELPERS
# ==================================================

def extract_season(
    text,
):

    match = re.search(
        r"\b(20\d{2}/\d{2})\b",
        text,
    )


    if match:

        return match.group(
            1
        )


    return None


def extract_number_of_games(
    text,
    default=10,
):

    patterns = [
        r"last\s+(\d+)",
        r"previous\s+(\d+)",
        r"recent\s+(\d+)",
    ]


    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )


        if match:

            return int(
                match.group(
                    1
                )
            )


    return default


def extract_team_mentions(
    text,
):

    lowered = (
        text.lower()
    )


    found = []


    for team in TEAM_NAMES:

        if (
            team.lower()
            in lowered
        ):

            found.append(
                team
            )


    result = []


    for team in found:

        if (
            team
            not in result
        ):

            result.append(
                team
            )


    return result


def last_fixture_prediction(
    conversation,
):

    for turn in reversed(
        conversation
    ):

        if (
            turn.get(
                "role"
            )
            !=
            "assistant"
        ):

            continue


        for result in turn.get(
            "ui_results",
            [],
        ):

            if (
                isinstance(
                    result,
                    dict,
                )
                and
                result.get(
                    "type"
                )
                ==
                "fixture_prediction"
            ):

                return result


    return None


# ==================================================
# DETERMINISTIC ROUTER
# ==================================================

def deterministic_route(
    prompt,
    conversation,
):

    text = (
        prompt.strip()
    )

    lowered = (
        text.lower()
    )

    season = (
        extract_season(
            text
        )
    )

    teams = (
        extract_team_mentions(
            text
        )
    )


    # ==================================================
    # FOLLOW-UP HOME TEAM SWITCH
    # ==================================================

    if (
        "at home"
        in lowered
        and
        (
            "what if"
            in lowered
            or
            "were at home"
            in lowered
        )
    ):

        previous_prediction = (
            last_fixture_prediction(
                conversation
            )
        )


        if (
            previous_prediction
            and
            len(teams) >= 1
        ):

            requested_home_team = (
                teams[
                    0
                ]
            )

            previous_home = (
                previous_prediction.get(
                    "home_team"
                )
            )

            previous_away = (
                previous_prediction.get(
                    "away_team"
                )
            )


            if (
                requested_home_team
                ==
                previous_home
            ):

                return {

                    "name":
                        "fixture_prediction",

                    "arguments":
                        {

                            "home_team":
                                previous_home,

                            "away_team":
                                previous_away,
                        },
                }


            if (
                requested_home_team
                ==
                previous_away
            ):

                return {

                    "name":
                        "fixture_prediction",

                    "arguments":
                        {

                            "home_team":
                                previous_away,

                            "away_team":
                                previous_home,
                        },
                }


    # ==================================================
    # FIXTURE PREDICTION
    # ==================================================

    prediction_words = [
        "predict",
        "prediction",
        "forecast",
        "chances",
        "chance of",
        "probability",
        "likely score",
        "likely outcome",
        "expected goals",
        "most likely score",
    ]


    asks_prediction = any(
        word
        in lowered
        for word
        in prediction_words
    )


    if asks_prediction:

        # ----------------------------------------------
        # Explicit Team A vs Team B
        # ----------------------------------------------

        versus_match = re.search(
            r"(.+?)\s+v(?:s\.?)?\s+(.+)",
            text,
            flags=re.IGNORECASE,
        )


        if versus_match:

            left_text = (
                versus_match
                .group(
                    1
                )
            )

            right_text = (
                versus_match
                .group(
                    2
                )
            )


            left_teams = (
                extract_team_mentions(
                    left_text
                )
            )

            right_teams = (
                extract_team_mentions(
                    right_text
                )
            )


            if (
                left_teams
                and
                right_teams
            ):

                return {

                    "name":
                        "fixture_prediction",

                    "arguments":
                        {

                            "home_team":
                                left_teams[
                                    -1
                                ],

                            "away_team":
                                right_teams[
                                    0
                                ],
                        },
                }


        # ----------------------------------------------
        # Team A chances vs Team B at home
        # ----------------------------------------------

        if (
            len(
                teams
            )
            >=
            2
            and
            "at home"
            in lowered
        ):

            return {

                "name":
                    "fixture_prediction",

                "arguments":
                    {

                        "home_team":
                            teams[
                                0
                            ],

                        "away_team":
                            teams[
                                1
                            ],
                    },
            }


        # ----------------------------------------------
        # Two explicit teams
        # ----------------------------------------------

        if (
            len(
                teams
            )
            >=
            2
        ):

            return {

                "name":
                    "fixture_prediction",

                "arguments":
                    {

                        "home_team":
                            teams[
                                0
                            ],

                        "away_team":
                            teams[
                                1
                            ],
                    },
            }


    # ==================================================
    # LEAGUE TABLE
    # ==================================================

    if (
        "league table"
        in lowered
        or
        "standings"
        in lowered
    ):

        if season:

            return {

                "name":
                    "league_table",

                "arguments":
                    {

                        "season":
                            season,
                    },
            }


    # ==================================================
    # HOME / AWAY RECORD
    # ==================================================

    if (
        len(
            teams
        )
        >=
        1
        and
        season
        and
        "home"
        in lowered
        and
        "away"
        in lowered
        and
        (
            "compare"
            in lowered
            or
            "performance"
            in lowered
            or
            "record"
            in lowered
        )
    ):

        return {

            "name":
                "home_away_record",

            "arguments":
                {

                    "team":
                        teams[
                            0
                        ],

                    "season":
                        season,
                },
        }


    # ==================================================
    # TEAM COMPARISON
    # ==================================================

    if (
        len(
            teams
        )
        >=
        2
        and
        season
        and
        "compare"
        in lowered
    ):

        return {

            "name":
                "team_comparison",

            "arguments":
                {

                    "team1":
                        teams[
                            0
                        ],

                    "team2":
                        teams[
                            1
                        ],

                    "season":
                        season,
                },
        }


    # ==================================================
    # HEAD TO HEAD
    # ==================================================

    h2h_phrases = [
        "head to head",
        "head-to-head",
        "matches against",
        "meetings between",
        "games against",
    ]


    if (
        len(
            teams
        )
        >=
        2
        and
        any(
            phrase
            in lowered
            for phrase
            in h2h_phrases
        )
    ):

        return {

            "name":
                "head_to_head",

            "arguments":
                {

                    "team1":
                        teams[
                            0
                        ],

                    "team2":
                        teams[
                            1
                        ],
                },
        }


    # ==================================================
    # RECENT INDIVIDUAL MATCHES
    # ==================================================

    individual_match_words = [
        "show me",
        "list",
        "individual",
    ]


    if (
        len(
            teams
        )
        >=
        1
        and
        (
            "last"
            in lowered
            or
            "recent"
            in lowered
        )
        and
        (
            "matches"
            in lowered
            or
            "results"
            in lowered
            or
            "games"
            in lowered
        )
        and
        any(
            phrase
            in lowered
            for phrase
            in individual_match_words
        )
    ):

        games = (
            extract_number_of_games(
                text,
                default=10,
            )
        )


        return {

            "name":
                "recent_form",

            "arguments":
                {

                    "team":
                        teams[
                            0
                        ],

                    "games":
                        games,
                },
        }


    # ==================================================
    # RECENT FORM SUMMARY
    # ==================================================

    if (
        len(
            teams
        )
        >=
        1
        and
        (
            "recent form"
            in lowered
            or
            "performed over"
            in lowered
            or
            "performed in their last"
            in lowered
            or
            "how are"
            in lowered
            or
            "how have"
            in lowered
        )
        and
        (
            "last"
            in lowered
            or
            "recent"
            in lowered
        )
    ):

        games = (
            extract_number_of_games(
                text,
                default=10,
            )
        )


        return {

            "name":
                "form_summary",

            "arguments":
                {

                    "team":
                        teams[
                            0
                        ],

                    "games":
                        games,
                },
        }


    # ==================================================
    # TEAM SEASON RECORD
    # ==================================================

    performance_words = [
        "perform",
        "performed",
        "performance",
        "record",
        "how did",
    ]


    if (
        len(
            teams
        )
        >=
        1
        and
        season
        and
        any(
            phrase
            in lowered
            for phrase
            in performance_words
        )
    ):

        return {

            "name":
                "team_record",

            "arguments":
                {

                    "team":
                        teams[
                            0
                        ],

                    "season":
                        season,
                },
        }


    return None


# ==================================================
# EXECUTE TOOL
# ==================================================

def execute_tool(
    function_name,
    arguments,
):

    if (
        function_name
        not in TOOL_FUNCTIONS
    ):

        return {

            "error":
                (
                    "Unknown tool requested: "
                    f"{function_name}"
                )
        }


    function = (
        TOOL_FUNCTIONS[
            function_name
        ]
    )


    try:

        return function(
            **arguments
        )


    except Exception as error:

        return {

            "error":
                str(
                    error
                )
        }


# ==================================================
# EXPLAIN TOOL RESULT
# ==================================================

def explain_tool_result(
    messages,
    function_name,
    arguments,
    result,
):

    tool_context = {

        "tool":
            function_name,

        "arguments":
            arguments,

        "result":
            result,
    }


    explanation_messages = (
        list(
            messages
        )
        +
        [
            {

                "role":
                    "system",

                "content":
                    (
                        "A Football Copilot tool has already "
                        "been executed for the user's latest "
                        "question. Use ONLY the structured "
                        "tool result below to answer the "
                        "question. Do not output JSON, a "
                        "function call or tool instructions. "
                        "Do not contradict the returned data. "
                        "If the result contains an error, "
                        "explain the error clearly.\n\n"
                        "TOOL RESULT:\n"
                        +
                        json.dumps(
                            tool_context,
                            default=str,
                        )
                    ),
            }
        ]
    )


    response = chat(
        model=MODEL,
        messages=explanation_messages,
    )


    answer = (
        response
        .message
        .content
    )


    messages.append(
        {

            "role":
                "assistant",

            "content":
                answer,
        }
    )


    return answer


# ==================================================
# TEXT TOOL CALL FALLBACK
# ==================================================

def parse_text_tool_call(
    content,
):

    if not content:

        return None


    cleaned = (
        content
        .strip()
        .replace(
            "```json",
            "",
        )
        .replace(
            "```",
            "",
        )
        .strip()
    )


    # --------------------------------------------------
    # Entire response as JSON
    # --------------------------------------------------

    try:

        parsed = json.loads(
            cleaned
        )


        if (
            isinstance(
                parsed,
                dict,
            )
            and
            parsed.get(
                "name"
            )
            in TOOL_FUNCTIONS
            and
            isinstance(
                parsed.get(
                    "arguments"
                ),
                dict,
            )
        ):

            return parsed


    except Exception:

        pass


    # --------------------------------------------------
    # Extract JSON object from response
    # --------------------------------------------------

    match = re.search(
        r'\{\s*"name"\s*:\s*"[^"]+"\s*,'
        r'\s*"arguments"\s*:\s*\{.*?\}\s*\}',
        cleaned,
        flags=re.DOTALL,
    )


    if not match:

        return None


    try:

        parsed = json.loads(
            match.group(
                0
            )
        )


        if (
            parsed.get(
                "name"
            )
            in TOOL_FUNCTIONS
            and
            isinstance(
                parsed.get(
                    "arguments"
                ),
                dict,
            )
        ):

            return parsed


    except Exception:

        return None


    return None


# ==================================================
# AGENT
# ==================================================

def run_agent(
    messages,
    prompt,
    conversation,
):

    ui_results = []
    debug_info = []


    # ==================================================
    # 1. DETERMINISTIC ROUTING
    # ==================================================

    deterministic_call = (
        deterministic_route(
            prompt,
            conversation,
        )
    )


    if deterministic_call:

        function_name = (
            deterministic_call[
                "name"
            ]
        )

        arguments = (
            deterministic_call[
                "arguments"
            ]
        )


        debug_info.append(
            {

                "routing":
                    "deterministic",

                "tool":
                    function_name,

                "arguments":
                    arguments,
            }
        )


        result = (
            execute_tool(
                function_name,
                arguments,
            )
        )


        ui_results.append(
            result
        )


        answer = (
            explain_tool_result(
                messages,
                function_name,
                arguments,
                result,
            )
        )


        return (
            answer,
            ui_results,
            debug_info,
        )


    # ==================================================
    # 2. NATIVE OLLAMA TOOL CALLING
    # ==================================================

    response = chat(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
    )


    if (
        response
        .message
        .tool_calls
    ):

        messages.append(
            response.message
        )


        for tool_call in (
            response
            .message
            .tool_calls
        ):

            function_name = (
                tool_call
                .function
                .name
            )

            arguments = (
                tool_call
                .function
                .arguments
            )


            debug_info.append(
                {

                    "routing":
                        "ollama_tool_call",

                    "tool":
                        function_name,

                    "arguments":
                        arguments,
                }
            )


            result = (
                execute_tool(
                    function_name,
                    arguments,
                )
            )


            ui_results.append(
                result
            )


            messages.append(
                {

                    "role":
                        "tool",

                    "tool_name":
                        function_name,

                    "content":
                        json.dumps(
                            result,
                            default=str,
                        ),
                }
            )


        final_response = chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )


        messages.append(
            final_response.message
        )


        return (
            final_response
            .message
            .content,

            ui_results,

            debug_info,
        )


    # ==================================================
    # 3. TEXT / JSON TOOL CALL RECOVERY
    # ==================================================

    text_tool_call = (
        parse_text_tool_call(
            response
            .message
            .content
        )
    )


    if text_tool_call:

        function_name = (
            text_tool_call[
                "name"
            ]
        )

        arguments = (
            text_tool_call[
                "arguments"
            ]
        )


        debug_info.append(
            {

                "routing":
                    "recovered_text_tool_call",

                "tool":
                    function_name,

                "arguments":
                    arguments,
            }
        )


        result = (
            execute_tool(
                function_name,
                arguments,
            )
        )


        ui_results.append(
            result
        )


        answer = (
            explain_tool_result(
                messages,
                function_name,
                arguments,
                result,
            )
        )


        return (
            answer,
            ui_results,
            debug_info,
        )


    # ==================================================
    # 4. NORMAL CONVERSATIONAL RESPONSE
    # ==================================================

    messages.append(
        response.message
    )


    return (
        response
        .message
        .content,

        ui_results,

        debug_info,
    )


# ==================================================
# UI HELPERS
# ==================================================

def safe_int(
    value,
):

    try:

        return int(
            value
        )

    except Exception:

        return 0


def safe_float(
    value,
):

    try:

        return float(
            value
        )

    except Exception:

        return 0.0


def format_goal_difference(
    value,
):

    value = (
        safe_int(
            value
        )
    )


    if value > 0:

        return f"+{value}"


    return str(
        value
    )


def render_kpis(
    items,
):

    if not items:

        return


    columns = st.columns(
        len(
            items
        )
    )


    for column, item in zip(
        columns,
        items,
    ):

        column.metric(
            item[
                "label"
            ],
            item[
                "value"
            ],
        )


# ==================================================
# MODEL VALIDATION DISPLAY
# ==================================================

def render_model_validation():

    st.markdown(
        "#### Model validation"
    )


    model_column, market_column = (
        st.columns(
            2
        )
    )


    with model_column:

        st.markdown(
            "##### Frozen Model 2"
        )

        st.metric(
            "1X2 accuracy",
            f"{MODEL_ACCURACY:.2f}%",
        )

        st.metric(
            "Log loss",
            f"{MODEL_LOG_LOSS:.4f}",
        )

        st.metric(
            "Brier score",
            f"{MODEL_BRIER:.4f}",
        )


    with market_column:

        st.markdown(
            "##### Historical market"
        )

        st.metric(
            "1X2 accuracy",
            f"{MARKET_ACCURACY:.2f}%",
        )

        st.metric(
            "Log loss",
            f"{MARKET_LOG_LOSS:.4f}",
        )

        st.metric(
            "Brier score",
            f"{MARKET_BRIER:.4f}",
        )


    st.caption(
        f"Walk-forward validation sample: "
        f"{VALIDATION_MATCHES:,} "
        "Premier League matches."
    )


    st.info(
        "The historical bookmaker market "
        "outperformed Model 2 overall. "
        "Historical model-market disagreements "
        "did not produce a persistent "
        "positive-return signal."
    )


# ==================================================
# FIXTURE RESULT RENDERER
# ==================================================

def render_fixture_prediction(
    result,
):

    home_team = (
        result.get(
            "home_team"
        )
    )

    away_team = (
        result.get(
            "away_team"
        )
    )


    home_probability = (
        safe_float(
            result.get(
                "home_win_probability"
            )
        )
    )

    draw_probability = (
        safe_float(
            result.get(
                "draw_probability"
            )
        )
    )

    away_probability = (
        safe_float(
            result.get(
                "away_win_probability"
            )
        )
    )


    home_xg = (
        safe_float(
            result.get(
                "expected_home_goals"
            )
        )
    )

    away_xg = (
        safe_float(
            result.get(
                "expected_away_goals"
            )
        )
    )


    # --------------------------------------------------
    # Predicted outcome
    # --------------------------------------------------

    outcome_probabilities = {

        home_team:
            home_probability,

        "Draw":
            draw_probability,

        away_team:
            away_probability,
    }


    predicted_outcome = max(
        outcome_probabilities,
        key=outcome_probabilities.get,
    )


    scorelines = (
        result.get(
            "most_likely_scores",
            [],
        )
    )


    most_likely_score = (
        scorelines[
            0
        ].get(
            "score"
        )
        if scorelines
        else
        "Unavailable"
    )


    # --------------------------------------------------
    # Header
    # --------------------------------------------------

    st.subheader(
        f"Prediction: "
        f"{home_team} vs {away_team}"
    )


    st.caption(
        "Frozen Model 2 statistical prediction"
    )


    # --------------------------------------------------
    # Outcome + score
    # --------------------------------------------------

    outcome_columns = (
        st.columns(
            2
        )
    )


    outcome_columns[
        0
    ].metric(
        "Predicted outcome",
        predicted_outcome,
    )


    outcome_columns[
        1
    ].metric(
        "Most likely scoreline",
        most_likely_score,
    )


    # --------------------------------------------------
    # Probabilities
    # --------------------------------------------------

    columns = (
        st.columns(
            3
        )
    )


    columns[
        0
    ].metric(
        f"{home_team} win",
        f"{home_probability:.1f}%",
    )


    columns[
        1
    ].metric(
        "Draw",
        f"{draw_probability:.1f}%",
    )


    columns[
        2
    ].metric(
        f"{away_team} win",
        f"{away_probability:.1f}%",
    )


    probability_df = (
        pd.DataFrame(
            {

                "Outcome": [
                    f"{home_team} win",
                    "Draw",
                    f"{away_team} win",
                ],

                "Probability %": [
                    home_probability,
                    draw_probability,
                    away_probability,
                ],
            }
        )
        .set_index(
            "Outcome"
        )
    )


    st.markdown(
        "#### Outcome probabilities"
    )


    st.bar_chart(
        probability_df,
        use_container_width=True,
    )


    # --------------------------------------------------
    # Expected goals
    # --------------------------------------------------

    st.markdown(
        "#### Expected goals"
    )


    xg_columns = (
        st.columns(
            2
        )
    )


    xg_columns[
        0
    ].metric(
        home_team,
        f"{home_xg:.2f}",
    )


    xg_columns[
        1
    ].metric(
        away_team,
        f"{away_xg:.2f}",
    )


    # --------------------------------------------------
    # Scorelines
    # --------------------------------------------------

    if scorelines:

        st.markdown(
            "#### Most likely scorelines"
        )


        scoreline_df = (
            pd.DataFrame(
                [
                    {

                        "Score":
                            item.get(
                                "score"
                            ),

                        "Probability %":
                            safe_float(
                                item.get(
                                    "probability_pct"
                                )
                            ),
                    }

                    for item
                    in scorelines
                ]
            )
        )


        st.dataframe(
            scoreline_df,
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------
    # Model details
    # --------------------------------------------------

    with st.expander(
        "About this prediction"
    ):

        model_info = (
            result.get(
                "model",
                {},
            )
        )


        st.write(
            "**Production model:** "
            f"{model_info.get('name', PRODUCTION_MODEL_NAME)}"
        )


        st.write(
            "**Method:** "
            f"{model_info.get('type', PRODUCTION_MODEL_TYPE)}"
        )


        st.write(
            "**Historical feature season:** "
            f"{result.get('feature_season', 'Unknown')}"
        )


        feature_metadata = (
            result.get(
                "feature_metadata",
                {},
            )
        )


        if (
            feature_metadata.get(
                "cold_start_used"
            )
        ):

            cold_start_teams = (
                feature_metadata.get(
                    "cold_start_teams",
                    [],
                )
            )


            st.info(
                "Promoted-team cold-start prior "
                "used for: "
                +
                ", ".join(
                    cold_start_teams
                )
            )


            st.write(
                "**Cold-start method:** "
                f"{feature_metadata.get('cold_start_method')}"
            )


        st.caption(
            "The model uses the latest available "
            "historical team data in the local "
            "dataset. It does not currently use "
            "future line-ups, injuries or live "
            "bookmaker odds."
        )


        render_model_validation()


        st.warning(
            "These probabilities are statistical "
            "estimates, not guarantees."
        )


# ==================================================
# TOOL RESULT RENDERER
# ==================================================

def render_tool_result(
    result,
):

    if not isinstance(
        result,
        dict,
    ):

        return


    if (
        "error"
        in result
    ):

        st.warning(
            result[
                "error"
            ]
        )

        return


    result_type = (
        result.get(
            "type"
        )
    )


    # ==================================================
    # FIXTURE PREDICTION
    # ==================================================

    if (
        result_type
        ==
        "fixture_prediction"
    ):

        render_fixture_prediction(
            result
        )

        return


    # ==================================================
    # STANDARD ANALYTICS
    # ==================================================

    data = (
        result.get(
            "data"
        )
    )


    if not data:

        return


    df = pd.DataFrame(
        data
    )


    # --------------------------------------------------
    # League table
    # --------------------------------------------------

    if (
        result_type
        ==
        "league_table"
    ):

        st.subheader(
            "League Table"
        )


        preferred_columns = [
            "Position",
            "Team",
            "Played",
            "Won",
            "Drawn",
            "Lost",
            "GoalsFor",
            "GoalsAgainst",
            "GoalDifference",
            "Points",
        ]


        display_columns = [
            column
            for column
            in preferred_columns
            if column
            in df.columns
        ]


        if display_columns:

            st.dataframe(
                df[
                    display_columns
                ],
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )


    # --------------------------------------------------
    # Recent matches
    # --------------------------------------------------

    elif (
        result_type
        ==
        "recent_form"
    ):

        st.subheader(
            "Recent Matches"
        )


        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------
    # Head to head
    # --------------------------------------------------

    elif (
        result_type
        ==
        "head_to_head"
    ):

        st.subheader(
            "Head-to-Head"
        )


        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------
    # Home / away
    # --------------------------------------------------

    elif (
        result_type
        ==
        "home_away_record"
    ):

        st.subheader(
            "Home vs Away"
        )


        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )


        required_columns = [
            "Venue",
            "PointsPerGame",
            "GoalsPerGame",
            "GoalsAgainstPerGame",
        ]


        if all(
            column
            in df.columns
            for column
            in required_columns
        ):

            chart_df = (
                df[
                    required_columns
                ]
                .set_index(
                    "Venue"
                )
            )


            st.markdown(
                "#### Performance comparison"
            )


            st.bar_chart(
                chart_df,
                use_container_width=True,
            )


    # --------------------------------------------------
    # Team comparison
    # --------------------------------------------------

    elif (
        result_type
        ==
        "team_comparison"
    ):

        st.subheader(
            "Team Comparison"
        )


        if (
            len(
                df
            )
            ==
            2
        ):

            team_1 = (
                df.iloc[
                    0
                ]
            )

            team_2 = (
                df.iloc[
                    1
                ]
            )


            column_1, column_2 = (
                st.columns(
                    2
                )
            )


            with column_1:

                st.markdown(
                    f"### "
                    f"{team_1['Team']}"
                )


                render_kpis(
                    [
                        {

                            "label":
                                "Points",

                            "value":
                                safe_int(
                                    team_1[
                                        "Points"
                                    ]
                                ),
                        },
                        {

                            "label":
                                "Wins",

                            "value":
                                safe_int(
                                    team_1[
                                        "Won"
                                    ]
                                ),
                        },
                        {

                            "label":
                                "PPG",

                            "value":
                                team_1.get(
                                    "PointsPerGame",
                                    "",
                                ),
                        },
                        {

                            "label":
                                "Win %",

                            "value":
                                (
                                    f"{team_1.get('WinPercentage', '')}%"
                                ),
                        },
                    ]
                )


            with column_2:

                st.markdown(
                    f"### "
                    f"{team_2['Team']}"
                )


                render_kpis(
                    [
                        {

                            "label":
                                "Points",

                            "value":
                                safe_int(
                                    team_2[
                                        "Points"
                                    ]
                                ),
                        },
                        {

                            "label":
                                "Wins",

                            "value":
                                safe_int(
                                    team_2[
                                        "Won"
                                    ]
                                ),
                        },
                        {

                            "label":
                                "PPG",

                            "value":
                                team_2.get(
                                    "PointsPerGame",
                                    "",
                                ),
                        },
                        {

                            "label":
                                "Win %",

                            "value":
                                (
                                    f"{team_2.get('WinPercentage', '')}%"
                                ),
                        },
                    ]
                )


        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )


        chart_columns = [
            "Team",
            "Points",
            "GoalsFor",
            "GoalsAgainst",
        ]


        if all(
            column
            in df.columns
            for column
            in chart_columns
        ):

            st.markdown(
                "#### Season comparison"
            )


            chart_df = (
                df[
                    chart_columns
                ]
                .set_index(
                    "Team"
                )
            )


            st.bar_chart(
                chart_df,
                use_container_width=True,
            )


    # --------------------------------------------------
    # Team record
    # --------------------------------------------------

    elif (
        result_type
        ==
        "team_record"
    ):

        st.subheader(
            "Team Record"
        )


        if (
            len(
                df
            )
            >
            0
        ):

            row = (
                df.iloc[
                    0
                ]
            )


            kpis = []


            if (
                "Points"
                in row.index
            ):

                kpis.append(
                    {

                        "label":
                            "Points",

                        "value":
                            safe_int(
                                row[
                                    "Points"
                                ]
                            ),
                    }
                )


            if (
                "Won"
                in row.index
            ):

                kpis.append(
                    {

                        "label":
                            "Wins",

                        "value":
                            safe_int(
                                row[
                                    "Won"
                                ]
                            ),
                    }
                )


            if (
                "WinPercentage"
                in row.index
            ):

                kpis.append(
                    {

                        "label":
                            "Win %",

                        "value":
                            (
                                f"{row['WinPercentage']}%"
                            ),
                    }
                )


            if (
                "GoalDifference"
                in row.index
            ):

                kpis.append(
                    {

                        "label":
                            "Goal Difference",

                        "value":
                            format_goal_difference(
                                row[
                                    "GoalDifference"
                                ]
                            ),
                    }
                )


            render_kpis(
                kpis
            )


            extra_kpis = []


            if (
                "PointsPerGame"
                in row.index
            ):

                extra_kpis.append(
                    {

                        "label":
                            "PPG",

                        "value":
                            row[
                                "PointsPerGame"
                            ],
                    }
                )


            if (
                "GoalsPerGame"
                in row.index
            ):

                extra_kpis.append(
                    {

                        "label":
                            "Goals / Game",

                        "value":
                            row[
                                "GoalsPerGame"
                            ],
                    }
                )


            if (
                "GoalsAgainstPerGame"
                in row.index
            ):

                extra_kpis.append(
                    {

                        "label":
                            "Goals Against / Game",

                        "value":
                            row[
                                "GoalsAgainstPerGame"
                            ],
                    }
                )


            render_kpis(
                extra_kpis
            )


        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------
    # Form summary
    # --------------------------------------------------

    elif (
        result_type
        ==
        "form_summary"
    ):

        st.subheader(
            "Form Summary"
        )


        if (
            len(
                df
            )
            >
            0
        ):

            row = (
                df.iloc[
                    0
                ]
            )


            kpis = []


            for column, label in [

                (
                    "Points",
                    "Points",
                ),

                (
                    "Won",
                    "Wins",
                ),

                (
                    "PointsPerGame",
                    "PPG",
                ),

                (
                    "WinPercentage",
                    "Win %",
                ),
            ]:

                if (
                    column
                    in row.index
                ):

                    value = (
                        row[
                            column
                        ]
                    )


                    if (
                        column
                        ==
                        "WinPercentage"
                    ):

                        value = (
                            f"{value}%"
                        )


                    kpis.append(
                        {

                            "label":
                                label,

                            "value":
                                value,
                        }
                    )


            render_kpis(
                kpis
            )


            extra_kpis = []


            for column, label in [

                (
                    "GoalsPerGame",
                    "Goals / Game",
                ),

                (
                    "GoalsAgainstPerGame",
                    "Goals Against / Game",
                ),

                (
                    "GoalDifference",
                    "Goal Difference",
                ),
            ]:

                if (
                    column
                    in row.index
                ):

                    extra_kpis.append(
                        {

                            "label":
                                label,

                            "value":
                                row[
                                    column
                                ],
                        }
                    )


            render_kpis(
                extra_kpis
            )


            if all(
                column
                in row.index
                for column
                in [
                    "Won",
                    "Drawn",
                    "Lost",
                ]
            ):

                form_chart = (
                    pd.DataFrame(
                        {

                            "Result": [
                                "Wins",
                                "Draws",
                                "Losses",
                            ],

                            "Matches": [
                                row[
                                    "Won"
                                ],
                                row[
                                    "Drawn"
                                ],
                                row[
                                    "Lost"
                                ],
                            ],
                        }
                    )
                    .set_index(
                        "Result"
                    )
                )


                st.markdown(
                    "#### Results breakdown"
                )


                st.bar_chart(
                    form_chart,
                    use_container_width=True,
                )


        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------
    # Generic fallback
    # --------------------------------------------------

    else:

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Football Copilot",
    page_icon="⚽",
    layout="wide",
)


# ==================================================
# SESSION STATE
# ==================================================

if (
    "llm_messages"
    not in st.session_state
):

    st.session_state.llm_messages = [
        {

            "role":
                "system",

            "content":
                SYSTEM_PROMPT,
        }
    ]


if (
    "conversation"
    not in st.session_state
):

    st.session_state.conversation = []


if (
    "debug_info"
    not in st.session_state
):

    st.session_state.debug_info = []


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header(
        "⚽ Football Copilot"
    )


    st.caption(
        "Premier League conversational "
        "analytics and prediction"
    )


    st.divider()


    st.markdown(
        "### Data"
    )


    st.write(
        "**Historical coverage:** "
        "2021/22 to 2025/26"
    )


    st.write(
        "**Live experiment:** "
        "2026/27"
    )


    st.write(
        "**Analytics database:** "
        "DuckDB"
    )


    st.divider()


    st.markdown(
        "### Prediction"
    )


    st.write(
        "**Production model:** "
        "Model 2"
    )


    st.write(
        "**Method:** "
        "Poisson regression"
    )


    st.write(
        f"**Validation matches:** "
        f"{VALIDATION_MATCHES:,}"
    )


    st.write(
        f"**Accuracy:** "
        f"{MODEL_ACCURACY:.2f}%"
    )


    st.divider()


    st.markdown(
        "### AI"
    )


    st.write(
        f"**Model:** "
        f"{MODEL}"
    )


    st.write(
        "**Runtime:** "
        "Ollama"
    )


    st.divider()


    if st.button(
        "New conversation",
        use_container_width=True,
    ):

        st.session_state.llm_messages = [
            {

                "role":
                    "system",

                "content":
                    SYSTEM_PROMPT,
            }
        ]


        st.session_state.conversation = []

        st.session_state.debug_info = []

        st.rerun()


    st.divider()


    with st.expander(
        "Developer debug"
    ):

        if (
            st.session_state.debug_info
        ):

            for item in (
                st.session_state.debug_info
            ):

                st.write(
                    "**Routing:** "
                    f"{item.get('routing', 'unknown')}"
                )


                st.write(
                    "**Tool:** "
                    f"{item.get('tool')}"
                )


                st.json(
                    item.get(
                        "arguments",
                        {},
                    )
                )


        else:

            st.caption(
                "No tool call yet."
            )


# ==================================================
# MAIN PAGE
# ==================================================

st.title(
    "⚽ Football Copilot"
)


st.caption(
    "Conversational Premier League analytics, "
    "statistically validated match prediction and "
    "live 2026/27 model monitoring."
)


# ==================================================
# STATUS CARDS
# ==================================================

status_1, status_2, status_3, status_4 = (
    st.columns(
        4
    )
)


status_1.metric(
    "Historical seasons",
    "5",
)


status_2.metric(
    "Production model",
    "Model 2",
)


status_3.metric(
    "Validation matches",
    f"{VALIDATION_MATCHES:,}",
)


status_4.metric(
    "Live season",
    "2026/27",
)


# ==================================================
# EXAMPLE QUESTIONS
# ==================================================

with st.expander(
    "What can I ask?"
):

    st.markdown(
        """
### Historical analytics

- How did Liverpool perform in 2025/26?
- How have Liverpool performed over their last 10 matches?
- Show me Liverpool's last five matches.
- Compare Liverpool and Arsenal in 2025/26.
- Compare Liverpool's home and away record in 2025/26.
- Show me the Premier League table for 2025/26.
- Show me Liverpool's matches against Arsenal.

### Predictive analytics

- Predict Liverpool vs Everton.
- Predict Liverpool vs Arsenal.
- What are Liverpool's chances of beating Arsenal at home?
- What is the most likely score for Liverpool vs Arsenal?
- Predict Arsenal vs Liverpool.
- What if Arsenal were at home?

### Model understanding

- How accurate is the prediction model?
- Is the model better than the bookmaker market?
"""
    )


# ==================================================
# MODEL PERFORMANCE
# ==================================================

with st.expander(
    "Production model performance"
):

    render_model_validation()


# ==================================================
# LIVE 2026/27 PREDICTIONS
# ==================================================

st.divider()


try:

    render_live_predictions()


except Exception as error:

    st.warning(
        "Live gameweek predictions could not "
        "be displayed."
    )


    with st.expander(
        "Live prediction technical details"
    ):

        st.exception(
            error
        )


st.divider()


# ==================================================
# CONVERSATIONAL COPILOT
# ==================================================

st.markdown(
    "## 💬 Ask Football Copilot"
)


st.caption(
    "Ask about historical Premier League performance "
    "or use Model 2 to generate a statistical fixture "
    "prediction."
)


# ==================================================
# RE-RENDER CONVERSATION
# ==================================================

for turn in (
    st.session_state.conversation
):

    if (
        turn[
            "role"
        ]
        ==
        "user"
    ):

        with st.chat_message(
            "user"
        ):

            st.markdown(
                turn[
                    "content"
                ]
            )


    elif (
        turn[
            "role"
        ]
        ==
        "assistant"
    ):

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                turn[
                    "content"
                ]
            )


            for result in turn.get(
                "ui_results",
                [],
            ):

                render_tool_result(
                    result
                )


# ==================================================
# CHAT INPUT
# ==================================================

prompt = st.chat_input(
    "Ask a Premier League analytics "
    "or prediction question..."
)


# ==================================================
# PROCESS NEW QUESTION
# ==================================================

if prompt:

    # --------------------------------------------------
    # Store user turn
    # --------------------------------------------------

    st.session_state.conversation.append(
        {

            "role":
                "user",

            "content":
                prompt,
        }
    )


    st.session_state.llm_messages.append(
        {

            "role":
                "user",

            "content":
                prompt,
        }
    )


    # --------------------------------------------------
    # Display user turn
    # --------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            prompt
        )


    # --------------------------------------------------
    # Execute Football Copilot
    # --------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Analysing football data..."
        ):

            try:

                (
                    answer,
                    ui_results,
                    debug_info,
                ) = run_agent(
                    st.session_state.llm_messages,
                    prompt,
                    st.session_state.conversation,
                )


                st.session_state.debug_info = (
                    debug_info
                )


                st.markdown(
                    answer
                )


                for result in (
                    ui_results
                ):

                    render_tool_result(
                        result
                    )


                st.session_state.conversation.append(
                    {

                        "role":
                            "assistant",

                        "content":
                            answer,

                        "ui_results":
                            ui_results,
                    }
                )


            except Exception as error:

                error_message = (
                    "I couldn't complete that "
                    "analysis successfully."
                )


                st.error(
                    error_message
                )


                st.session_state.conversation.append(
                    {

                        "role":
                            "assistant",

                        "content":
                            error_message,

                        "ui_results":
                            [],
                    }
                )


                with st.expander(
                    "Technical details"
                ):

                    st.exception(
                        error
                    )