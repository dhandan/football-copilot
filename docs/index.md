# Football Copilot

## 2026/27 Premier League Live Prediction Experiment

Football Copilot is an AI-powered football analytics and prediction project built to explore how historical match data, statistical modelling and conversational AI can be combined into an interactive football analytics application.

For the 2026/27 Premier League season, Football Copilot is being run as a **prospective live modelling experiment**.

Predictions are generated and frozen before each Gameweek. Once the matches have been completed, the predictions are evaluated against the actual results and the model's performance is tracked throughout the season.

The purpose of the experiment is not simply to publish predictions. It is to understand where the model performs well, where it fails, and how the modelling approach can be improved using evidence from genuinely unseen matches.

---

## 2026/27 Live Prediction Journal

### Gameweek 2

**Status: Predictions frozen**

Football Copilot's second prospective test of the 2026/27 Premier League season.

The same Model 2 used for Gameweek 1 remains unchanged, allowing GW2 to provide another genuinely unseen test of the existing model.

**GW2 snapshot**

- 10 Premier League fixtures
- Predictions generated before the Gameweek
- Model 2 unchanged following GW1
- 2 fixtures use promoted-team cold-start handling
- 8 of 10 fixtures have 1-1 as the most likely individual scoreline
- The recurring 1-1 scoreline pattern remains under observation

**[View Gameweek 2 predictions](gameweeks/GW02.md)**

---

### Gameweek 1

**Status: Complete**

Gameweek 1 was the first live prospective test of Football Copilot Model 2.

The model correctly predicted the 1X2 outcome in 5 of the 10 fixtures.

**GW1 Model 2 performance**

| Metric | GW1 |
|---|---:|
| Matches | 10 |
| Correct 1X2 outcomes | 5 |
| 1X2 accuracy | 50.0% |
| Exact scores | 0/10 |
| Log Loss | 1.0212 |
| Brier Score | 0.6109 |
| Home goals MAE | 0.913 |
| Away goals MAE | 1.001 |
| Total goals MAE | 0.892 |

### GW1 baseline comparison

An important part of the live experiment is determining whether Model 2 actually adds predictive value over simple prediction strategies.

| Approach | Accuracy | Log Loss | Brier Score |
|---|---:|---:|---:|
| Model 2 | 50.0% | 1.0212 | 0.6109 |
| Always Home | 70.0% | — | — |
| Historical Majority Class | 70.0% | — | — |
| Historical Outcome Frequency | 70.0% | 0.9435 | 0.5604 |

The historical Premier League dataset used for the baseline contains 1,900 matches, with the following outcome distribution:

- Home win: 44.16%
- Draw: 23.89%
- Away win: 31.95%

GW1 was unusually home-win heavy, with 7 of the 10 matches won by the home team. This helped the simple home-based baselines outperform Model 2 during the Gameweek.

Ten matches are far too small a sample from which to conclude that the simple baseline is a better predictor. The comparison will therefore continue prospectively across subsequent Gameweeks.

**[View the completed Gameweek 1 analysis](gameweeks/GW01.md)**

---

## What the live experiment is monitoring

The first five Gameweeks are being treated as a live monitoring period rather than an opportunity to repeatedly tune the model after individual results.

Five key hypotheses are being tracked.

### 1. xG compression

The model may be producing expected-goal estimates that are too tightly concentrated around similar values.

This could prevent it from adequately representing the difference between closely matched fixtures and games where one team should have a substantial advantage.

### 2. 1-1 scoreline concentration

In GW1, 8 of the 10 fixtures had 1-1 as the model's most likely individual scoreline.

The same pattern appeared again in the GW2 predictions, with another 8 of 10 fixtures producing a 1-1 modal scoreline.

This does **not** mean the model predicts these matches to finish as draws.

The predicted 1X2 outcome is calculated from the combined probability of all possible home-win, draw and away-win scorelines. A 1-1 result can therefore be the single most likely individual score while the combined probability of a home or away victory is greater.

The persistence of the 1-1 modal score is nevertheless an important diagnostic being monitored.

### 3. Promoted-team cold starts

Newly promoted clubs have limited recent Premier League history.

Football Copilot therefore uses promoted-team priors to provide an initial estimate of their strength while sufficient Premier League evidence accumulates.

GW1 contained three fixtures involving cold-start handling.

Cold-start accuracy in GW1 was:

**33.3%**

This will continue to be monitored as the season develops.

### 4. Upset calibration

The experiment will monitor whether Model 2 assigns appropriate probabilities to unexpected results.

GW1's biggest model surprise was:

**Hull City 2-0 Manchester United**

The model assigned Hull City only a 17.5% probability of winning.

Tracking these misses will help determine whether the model is systematically too confident about favourites or whether they simply represent normal football uncertainty.

### 5. Incremental predictive value

The most important long-term question is whether Model 2's fixture-specific information actually improves predictions over simple historical benchmarks.

Model 2 will therefore be continuously compared with:

- Always Home
- Historical Majority Class
- Historical Premier League Outcome Frequency

Accuracy alone will not determine model quality.

Log Loss and Brier Score will also be tracked because they measure the quality and calibration of the probabilities generated by the model.

---

## Season performance

The season tracker will accumulate results as each Gameweek is completed.

Current position after GW1:

| Measure | Model 2 | Historical Outcome Frequency |
|---|---:|---:|
| Matches evaluated | 10 | 10 |
| 1X2 accuracy | 50.0% | 70.0% |
| Log Loss | 1.0212 | 0.9435 |
| Brier Score | 0.6109 | 0.5604 |

These numbers should not be interpreted as a meaningful model ranking after only ten matches.

The objective is to build a growing out-of-sample evidence base throughout the 2026/27 season.

---

## Experimental methodology

Each live Gameweek follows the same process:

1. Retrieve the upcoming Premier League fixtures.
2. Validate teams against the Football Copilot data model.
3. Run the prediction pipeline.
4. Freeze the official prediction snapshot before matches are played.
5. Publish the predictions to the Football Copilot journal.
6. Wait for the Gameweek to finish.
7. Retrieve the actual results.
8. Evaluate Model 2 against the actual outcomes.
9. Calculate Log Loss, Brier Score and goal prediction errors.
10. Compare Model 2 with simple historical baselines.
11. Update cumulative season performance.
12. Document findings and diagnostics.

This creates an auditable separation between **prediction** and **evaluation**.

---

## Model development roadmap

The live 2026/27 experiment forms part of a wider Football Copilot development roadmap.

### V1: Football Copilot

Build the analytics copilot from first principles, including:

- Premier League historical analytics
- conversational football queries
- match prediction
- probabilistic score modelling
- model validation
- live Gameweek prediction pipeline
- prospective model monitoring

### V1.5: Model improvement

After sufficient live evidence has accumulated, use the findings from the prospective experiment to improve prediction performance.

Potential areas include:

- probability calibration
- xG modelling
- scoreline modelling
- promoted-team adaptation
- recent form
- team-strength dynamics
- additional predictive features
- comparison with market probabilities

Changes will be evidence-led rather than made in response to individual Gameweeks.

### V2: Agent orchestration

Refactor Football Copilot using LangChain and LangGraph to explore:

- stateful agent workflows
- tool orchestration
- structured reasoning flows
- persistent analytical state
- specialised football analytics tools

### V3: Cloud deployment

Containerise Football Copilot and deploy the application and non-LLM services to a cloud platform.

The LLM will be treated as a replaceable inference layer so that different local or hosted models can be evaluated without redesigning the application.

### Future: Fantasy Football Copilot

Extend the platform into Fantasy Premier League analysis.

This will require a player-level dataset and could support:

- player performance analysis
- expected points
- transfer recommendations
- captain selection
- fixture difficulty
- squad optimisation
- differential identification
- risk and uncertainty analysis

---

## Why this project exists

Football Copilot is both a working football analytics application and a learning project.

The aim is to explore the complete lifecycle of an AI and analytics product:

**data → modelling → validation → application → live monitoring → agent orchestration → cloud deployment**

Publishing the live experiment also makes model performance transparent.

Successful predictions are recorded.

Failed predictions are recorded.

The model is judged against simple baselines rather than accuracy being viewed in isolation.

Most importantly, predictions are frozen before results are known.

---

## Follow the experiment

The journal will be updated throughout the 2026/27 Premier League season as each Gameweek is predicted and evaluated.

**[Gameweek 2: Frozen predictions](gameweeks/GW02.md)**

**[Gameweek 1: Completed analysis](gameweeks/GW01.md)**

---

## Football Copilot repository

The full Football Copilot source code, modelling pipeline and development history are available in the GitHub repository.

**[View the Football Copilot repository](https://github.com/dhandan/football-copilot)**