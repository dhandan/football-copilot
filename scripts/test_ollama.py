from ollama import chat


response = chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "system",
            "content": """
You are part of a football analytics application.

Do not invent football statistics.

If you have not been given analytical data,
say that you need to query the football analytics tools.
"""
        },
        {
            "role": "user",
            "content": """
How many Premier League games did Liverpool
win in the 2025/26 season?
"""
        }
    ]
)


print(response.message.content)