from pathlib import Path
import sys
import json

from ollama import chat


# --------------------------------------------------
# Project setup
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


# --------------------------------------------------
# Import Football Copilot tools
# --------------------------------------------------

from agent.football_tools import (
    team_record,
    recent_form,
    form_summary,
    league_table,
    home_away_record,
    head_to_head,
    team_comparison,
)


# --------------------------------------------------
# Model configuration
# --------------------------------------------------

MODEL = "qwen3:4b"


# --------------------------------------------------
# System prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
You are Football Copilot, a conversational Premier League
analytics assistant.

CRITICAL RULE:

Your own training knowledge is NOT a valid source of football
information.

The football analytics tools and their database are the ONLY
source of truth for football results, statistics, seasons,
standings and team performance.

TOOL RULES:

1. If the user asks about the performance of a team, you MUST
   call an appropriate football analytics tool.

2. If the user asks about a specific season, you MUST query
   the tools. Never decide yourself whether that season has
   started, finished or exists.

3. If the user asks about results, form, wins, losses, draws,
   goals, points, league position, home performance, away
   performance, head-to-head performance or comparisons,
   you MUST use a tool.

4. Never answer a football data question from your own
   training knowledge.

5. Never state that a season has not started, is ongoing or
   has finished unless this information can be established
   from the provided data.

6. Treat all tool results as authoritative for this
   application.

7. Use conversation history to resolve follow-up questions
   such as:
   "What about Arsenal?"
   "What about away from home?"
   "Compare them."

8. After receiving tool results, explain the statistics in
   clear natural language.

9. Do not invent statistics that are not present in the
   tool results.

10. If the tools cannot answer a question, say that the
    available data cannot answer it. Do not substitute your
    own football knowledge.

11. Keep answers concise unless the user requests more
    detail.
"""


# --------------------------------------------------
# Tools exposed to the model
# --------------------------------------------------

TOOLS = [
    team_record,
    recent_form,
    form_summary,
    league_table,
    home_away_record,
    head_to_head,
    team_comparison,
]


# --------------------------------------------------
# Tool lookup dictionary
# --------------------------------------------------

TOOL_FUNCTIONS = {
    "team_record": team_record,
    "recent_form": recent_form,
    "form_summary": form_summary,
    "league_table": league_table,
    "home_away_record": home_away_record,
    "head_to_head": head_to_head,
    "team_comparison": team_comparison,
}


# --------------------------------------------------
# Run one conversational turn
# --------------------------------------------------

def run_turn(messages, user_message):
    """
    Send one user message to Ollama.

    If Ollama requests one or more analytical tools,
    execute them and send the results back to Ollama.

    Return the final assistant response.
    """

    messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    response = chat(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
    )

    messages.append(response.message)

    # --------------------------------------------------
    # Execute tool calls
    # --------------------------------------------------

    if response.message.tool_calls:

        for tool_call in response.message.tool_calls:

            function_name = tool_call.function.name
            arguments = tool_call.function.arguments
            print()
            print(
                f"[Tool: {function_name}]"
            )

            print(
                f"[Arguments: {arguments}]"
            )

            if function_name not in TOOL_FUNCTIONS:

                result = {
                    "error": (
                        f"Unknown tool requested: "
                        f"{function_name}"
                    )
                }

            else:

                function = TOOL_FUNCTIONS[
                    function_name
                ]

                try:

                    result = function(
                        **arguments
                    )

                except Exception as error:

                    result = {
                        "error": str(error)
                    }

            messages.append(
                {
                    "role": "tool",
                    "tool_name": function_name,
                    "content": json.dumps(
                        result,
                        default=str,
                    ),
                }
            )

        # --------------------------------------------------
        # Ask model to explain tool results
        # --------------------------------------------------

        final_response = chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )

        messages.append(
            final_response.message
        )

        return final_response.message.content

    # --------------------------------------------------
    # No tool required
    # --------------------------------------------------

    return response.message.content


# --------------------------------------------------
# Main chat loop
# --------------------------------------------------

def main():

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    print()
    print("FOOTBALL COPILOT")
    print("================")
    print(
        "Ask me questions about Premier League "
        "teams and results."
    )
    print()
    print(
        "Type 'exit' or 'quit' to finish."
    )
    print()

    while True:

        user_message = input(
            "You: "
        ).strip()

        if not user_message:
            continue

        if user_message.lower() in [
            "exit",
            "quit",
        ]:

            print()
            print(
                "Football Copilot: Goodbye."
            )
            break

        try:

            answer = run_turn(
                messages,
                user_message,
            )

            print()
            print(
                f"Football Copilot: {answer}"
            )
            print()

        except Exception as error:

            print()
            print(
                "Football Copilot encountered "
                "an error:"
            )
            print(error)
            print()


# --------------------------------------------------
# Start application
# --------------------------------------------------

if __name__ == "__main__":
    main()