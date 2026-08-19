# Football Copilot 2026/27

## A live AI and predictive analytics experiment

Football Copilot is a personal experiment exploring how conversational AI, deterministic analytics and predictive modelling can work together in practice.

For the 2026/27 Premier League season I am taking the experiment a step further.

Before each gameweek, Football Copilot will generate and freeze predictions for every Premier League fixture.

After the matches have been played, those predictions will be evaluated against the actual results.

The objective is not to claim that the model can predict football perfectly.  It is to create a genuine prospective experiment in model monitoring, AI evaluation and continuous learning.

## What I'm tracking

For every fixture Football Copilot records:

- home win probability
- draw probability
- away win probability
- predicted outcome
- expected goals
- most likely scoreline
- model version
- prediction timestamp
- cold-start methodology where required

After each gameweek I will evaluate:

- 1X2 prediction accuracy
- Log Loss
- Brier score
- exact score predictions
- expected goals versus actual goals
- performance against historical market probabilities
- cumulative season performance

## 2026/27 Gameweeks

### [Gameweek 1](gameweeks/GW01.md)

Opening weekend predictions generated before any matches were played.

---

## Production model

**Model:** Model 2 v1.0  
**Method:** Enhanced Poisson regression

Historical walk-forward validation:

| Metric | Model 2 | Historical Market |
|---|---:|---:|
| Accuracy | 52.69% | 54.48% |
| Log Loss | 0.9927 | 0.9641 |
| Brier Score | 0.5927 | 0.5730 |

The historical bookmaker market outperformed Model 2 overall.

This project should therefore be viewed as an AI, analytics and model-evaluation experiment rather than a betting recommendation service.

## Project

[View the full Football Copilot source code on GitHub](../)