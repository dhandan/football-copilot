from ollama import chat


print("\nTesting local Ollama model...")
print("=============================")


response = chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: Football Copilot is working."
        }
    ]
)


print("\nModel response:")
print(response.message.content)