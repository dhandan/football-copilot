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
# System prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
You are Football Copilot, a conversational
Premier League analytics assistant.

Your answers must be grounded in the football
analytics tools provided to you.

RULES:

1. Never invent football statistics.

2. When the user asks for numerical football
   information, use an appropriate tool.

3. Treat tool results as the source of truth.

4. Do not claim knowledge about matches or
   statistics that are not returned by the tools.

5. Use get-style analytical tools rather than
   attempting calculations yourself.

6. Explain statistics in clear natural language.

7. Highlight genuinely useful comparisons or trends
   when supported by the returned data.

8. Do not exaggerate conclusions.

9. If information is unavailable, say so clearly.

10. Keep answers concise unless the user asks
    for more detail.
"""

# --------------------------------------------------
# Question
# --------------------------------------------------

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    },
    {
        "role": "user",
        "content": "Compare Liverpool and Arsenal in 2025/26.",
    },
]


# --------------------------------------------------
# Tools available to the model
# --------------------------------------------------

tools = [
    team_record,
    recent_form,
    form_summary,
    league_table,
    home_away_record,
    head_to_head,
    team_comparison,
]


# --------------------------------------------------
# Map tool names to Python functions
# --------------------------------------------------

tool_functions = {
    "team_record": team_record,
    "recent_form": recent_form,
    "form_summary": form_summary,
    "league_table": league_table,
    "home_away_record": home_away_record,
    "head_to_head": head_to_head,
    "team_comparison": team_comparison,
}


# --------------------------------------------------
# Ask Ollama to interpret the question
# --------------------------------------------------

response = chat(
    model="qwen3:4b",
    messages=messages,
    tools=tools,
)

messages.append(response.message)


# --------------------------------------------------
# Check whether Ollama requested a tool
# --------------------------------------------------

if response.message.tool_calls:

    for tool_call in response.message.tool_calls:

        function_name = tool_call.function.name
        arguments = tool_call.function.arguments

        print("\nTOOL SELECTED")
        print("=============")
        print(f"Tool: {function_name}")
        print(f"Arguments: {arguments}")

        # Check that the tool exists
        if function_name not in tool_functions:
            raise ValueError(
                f"Unknown tool requested: {function_name}"
            )

        # Get the corresponding Python function
        function = tool_functions[function_name]

        # Execute the function
        try:
            result = function(**arguments)

        except Exception as error:
            result = {
                "error": str(error)
            }

        print("\nTOOL RESULT")
        print("===========")
        print(result)

        # Add the tool result to the conversation
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
    # Give the tool results back to Ollama
    # --------------------------------------------------

    final_response = chat(
        model="qwen3:4b",
        messages=messages,
        tools=tools,
    )

    print("\nFOOTBALL COPILOT")
    print("================")
    print(final_response.message.content)


# --------------------------------------------------
# No tool required
# --------------------------------------------------

else:

    print("\nFOOTBALL COPILOT")
    print("================")
    print(response.message.content)