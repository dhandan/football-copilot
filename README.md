# ⚽ Football Copilot

Football Copilot is an end-to-end conversational football analytics and prediction application built around Premier League data.

The project was developed from first principles to understand how an agentic analytics application works under the hood, deliberately avoiding orchestration frameworks such as LangChain and LangGraph.

It combines:

- Premier League historical match data
- DuckDB analytics
- Python analytical tools
- Local LLM inference with Ollama
- Deterministic tool routing
- Streamlit
- Statistical football prediction
- Walk-forward model validation
- Historical bookmaker market comparison


## Project Objective

The project had two main objectives.

### Part 1: Conversational football analytics

Allow a user to ask natural-language questions such as:

- How did Liverpool perform in 2025/26?
- Show Liverpool's last 10 matches.
- Compare Liverpool and Arsenal in 2025/26.
- Compare Liverpool's home and away performance.
- Show the Premier League table.

The LLM does not calculate these statistics itself.

Football Copilot routes the request to deterministic analytical tools that query the local football data and returns the result conversationally.


### Part 2: Predictive football analytics

Extend Football Copilot to estimate the probabilities of future match outcomes.

Example:

```text
Liverpool vs Arsenal

Liverpool win probability
Draw probability
Arsenal win probability

Expected goals
Most likely scorelines