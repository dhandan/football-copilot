# Football Copilot

Football Copilot is a personal AI and analytics experiment I built to deepen my practical understanding of how modern agentic AI applications work end-to-end.

Rather than using an orchestration framework such as LangChain or LangGraph, I built the core components from first principles to understand what is happening underneath the abstraction layer.

The project combines:

- local LLM inference with Qwen and Ollama
- deterministic tool routing
- conversational football analytics
- DuckDB
- Streamlit
- predictive modelling
- walk-forward validation
- model challenger testing
- historical bookmaker market benchmarking

I chose Premier League football as the domain because it provides a practical environment for experimenting with both conversational analytics and probabilistic prediction.

The objective was not simply to build a chatbot.  It was to understand where an LLM adds value, where deterministic analytics should remain in control, and how statistical models can be exposed through a conversational interface.

## What it demonstrates

Football Copilot demonstrates an end-to-end AI and analytics workflow:

```text
Natural-language question
        ↓
Intent detection
        ↓
Deterministic router / LLM tool calling
        ↓
Trusted analytical or prediction tool
        ↓
Structured result
        ↓
LLM explanation
        ↓
Streamlit visualisation
```

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
```

## Football Copilot in Action

### Conversational analytics

![Football Copilot team comparison](docs/images/team-comparison.png)

### Statistical match prediction

![Football Copilot match prediction](docs/images/match-prediction.png)

## Architecture

Football Copilot separates conversational AI from the analytical and predictive calculations.  The LLM interprets and explains, while trusted Python tools and statistical models produce the underlying results.

```mermaid
flowchart TD
    A["User<br/>Natural Language Question"] --> B["Streamlit<br/>Conversational Interface"]

    B --> C{"Intent Router"}

    C -->|"Recognised analytical intent"| D["Deterministic Routing"]
    C -->|"Conversational / ambiguous intent"| E["Qwen LLM<br/>via Ollama"]

    E --> F["Tool Calling"]
    D --> G["Football Tools"]
    F --> G

    G --> H{"Request Type"}

    H -->|"Historical Analytics"| I["Analytics Engine<br/>Python"]
    H -->|"Match Prediction"| J["Production Model 2<br/>Enhanced Poisson"]

    I --> K["DuckDB<br/>Premier League Data"]
    J --> L["Historical Features<br/>Form, Venue & Team Strength"]

    K --> M["Structured Analytical Result"]
    L --> N["Probabilities, xG<br/>& Scorelines"]

    M --> O["Conversational Explanation<br/>& Visualisation"]
    N --> O

    O --> B
```

### Design principle

The LLM is **not the source of truth for football statistics or predictions**.

Football Copilot uses the LLM for natural-language interpretation and explanation, while deterministic analytical tools and the statistical prediction model perform the calculations.

This architecture reduces the risk of hallucinated statistics while retaining a conversational user experience.