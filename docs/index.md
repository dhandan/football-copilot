# Football Copilot

## 2026/27 Premier League Live Prediction Experiment

Football Copilot is an AI-powered football analytics and prediction project built to explore how historical match data, statistical modelling and conversational AI can be combined into an interactive football analytics application.

For the 2026/27 Premier League season, Football Copilot is being run as a **prospective live modelling experiment**.

Predictions are generated and frozen before each Gameweek. Once the matches have been completed, the predictions are evaluated against the actual results and performance is tracked throughout the season.

The purpose is not simply to publish predictions. It is to understand where the model performs well, where it fails, and how the modelling approach can be improved using evidence from genuinely unseen matches.

---

## 2026/27 Live Prediction Journal

### Gameweek 2

**Status: Complete**

The second prospective test of Football Copilot Model 2 produced the same 1X2 accuracy as GW1.

**GW2 performance**

| Metric | Result |
|---|---:|
| Matches | 10 |
| Correct 1X2 outcomes | 5 |
| 1X2 accuracy | 50.0% |
| Exact scores | 2 |
| Log Loss | 1.0615 |
| Brier Score | 0.6378 |
| Home goals MAE | 1.213 |
| Away goals MAE | 0.817 |
| Total goals MAE | 1.876 |

Model 2 outperformed all three simple baselines during GW2.

| Approach | Accuracy | Log Loss | Brier Score |
|---|---:|---:|---:|
| Model 2 | **50.0%** | **1.0615** | **0.6378** |
| Always Home | 30.0% | — | — |
| Historical Majority Class | 30.0% | — | — |
| Historical Outcome Frequency | 30.0% | 1.1311 | 0.6903 |

The scoreline diagnostic remained notable. Eight of the ten fixtures again had 1-1 as their most likely individual scoreline. Two actually finished 1-1.

Model 2 made no 1X2 draw predictions, while three GW2 matches finished as draws.

**[View the completed Gameweek 2 analysis](gameweeks/GW02.md)**

---

### Gameweek 1

**Status: Complete**

Gameweek 1 was the first prospective test of Football Copilot Model 2.

| Metric | Result |
|---|---:|
| Matches | 10 |
| Correct 1X2 outcomes | 5 |
| 1X2 accuracy | 50.0% |
| Exact scores | 0 |
| Log Loss | 1.0212 |
| Brier Score | 0.6109 |

GW1 contained seven home victories, resulting in the simple home-based baselines outperforming Model 2 during that Gameweek.

| Approach | Accuracy | Log Loss | Brier Score |
|---|---:|---:|---:|
| Model 2 | 50.0% | 1.0212 | 0.6109 |
| Always Home | 70.0% | — | — |
| Historical Majority Class | 70.0% | — | — |
| Historical Outcome Frequency | 70.0% | 0.9435 | 0.5604 |

**[View the completed Gameweek 1 analysis](gameweeks/GW01.md)**

---

## Season performance

After two completed Gameweeks, Football Copilot has now been tested prospectively against 20 previously unseen Premier League fixtures.

| Metric | GW1 | GW2 | Cumulative |
|---|---:|---:|---:|
| Matches | 10 | 10 | **20** |
| Correct outcomes | 5 | 5 | **10** |
| 1X2 accuracy | 50.0% | 50.0% | **50.0%** |
| Log Loss | 1.0212 | 1.0615 | **1.0414** |
| Brier Score | 0.6109 | 0.6378 | **0.6243** |
| Exact scores | 0 | 2 | **2** |
| Predicted draws | 0 | 0 | **0** |
| Actual draws | 1 | 3 | **4** |
| Modal 1-1 predictions | 8 | 8 | **16** |
| Actual 1-1 results | 0 | 2 | **2** |

Twenty matches remain a small sample, so these results should not be treated as a definitive assessment of the model.

The purpose of the live experiment is to build this evidence base progressively across the season.

---

## What the live experiment is monitoring

Five hypotheses are currently being tracked.

### 1. xG compression

The model may be producing expected-goal estimates that are too tightly concentrated around similar values.

This could reduce its ability to represent the difference between closely matched fixtures and games where one team should have a substantial advantage.

### 2. 1-1 scoreline concentration

This is currently the strongest live diagnostic.

Across GW1 and GW2:

- **16 of 20 predictions (80%)** had 1-1 as the most likely individual scoreline.
- **2 of 20 actual matches (10%)** finished 1-1.

Importantly, the most likely individual scoreline and the predicted 1X2 outcome are different calculations.

The 1X2 probabilities aggregate all possible home-win, draw and away-win scorelines. Therefore, 1-1 can be the highest-probability individual score while a home or away victory has the greatest combined probability.

Nevertheless, the concentration is sufficiently pronounced to warrant specific investigation.

### 3. Draw behaviour

Across the first 20 fixtures:

- Model 2 predicted **0 draws** as the most likely 1X2 outcome.
- **4 matches** actually finished as draws.

This creates an interesting modelling diagnostic alongside the 1-1 concentration.

The model frequently identifies 1-1 as the most probable individual score but has not yet made Draw its highest aggregate 1X2 probability.

### 4. Promoted-team cold starts

Five fixtures across GW1 and GW2 have involved promoted-team cold-start handling.

Model 2 has correctly predicted two of those five outcomes, giving an early cold-start accuracy of **40%**.

The sample remains too small for conclusions, but performance will continue to be tracked as Premier League evidence accumulates for the promoted teams.

### 5. Incremental predictive value

Model 2 is being compared prospectively with:

- Always Home
- Historical Majority Class
- Historical Premier League Outcome Frequency

GW1 favoured the simple historical baselines.

GW2 favoured Model 2.

This illustrates why multiple Gameweeks are required before determining whether Model 2 consistently adds predictive value.

Accuracy alone will not determine model quality. Log Loss and Brier Score are also tracked to evaluate the quality and calibration of the probabilities.

---

## Model development approach

The official Model 2 remains frozen during the initial live monitoring period.

This preserves the integrity of the prospective 2026/27 experiment: official predictions are generated using the same model rather than retrospectively benefiting from information learned after the season began.

Model development continues separately through controlled historical experiments.

Following the draw and scoreline behaviour observed during GW1 and GW2, three challenger experiments have now been completed:

- **Model 3A — Dixon-Coles:** tested low-score dependence.
- **Model 3B — Draw calibration:** tested whether Model 2's draw probabilities required systematic adjustment.
- **Model 4 — Direct 1X2 Gradient Boosting:** tested a structurally different direct outcome model incorporating Elo and additional relative-strength features.

None produced sufficient improvement to replace Model 2.

The evidence now suggests that further algorithm changes using essentially the same underlying match-history information are unlikely to deliver the material improvement being sought.

The next model-development phase will therefore focus on **data enrichment**, beginning with genuine expected-goals and related chance-quality information.

Model 2 will continue generating the official live predictions while this research is conducted independently.

---

## Experimental methodology

Each Gameweek follows the same process:

1. Retrieve upcoming Premier League fixtures.
2. Validate teams against the Football Copilot data model.
3. Generate Model 2 predictions.
4. Freeze the official prediction snapshot before matches are played.
5. Publish the predictions.
6. Retrieve actual results after the Gameweek.
7. Evaluate the frozen predictions.
8. Calculate Accuracy, Log Loss, Brier Score and goal prediction errors.
9. Compare Model 2 against simple baselines.
10. Update cumulative season performance.
11. Record diagnostics and modelling hypotheses.

This maintains an auditable separation between **prediction** and **evaluation**.

---

## Roadmap

### V1 — Football Copilot

Build the football analytics copilot from first principles, including historical analytics, conversational queries, match prediction, probabilistic score modelling, validation and prospective live monitoring.

### V1.5 — Model improvement and data enrichment

Use evidence from historical walk-forward testing and the prospective live experiment to improve the prediction capability without changing the frozen live baseline retrospectively.

Completed challenger experiments:

- Model 3A — Dixon-Coles low-score correction: **rejected**
- Model 3B — training-fitted draw calibration: **rejected**
- Model 4 — direct 1X2 gradient boosting with Elo: **rejected**

These experiments tested increasingly substantial changes to the modelling architecture but did not materially improve on Model 2.

The next phase therefore moves from **algorithm iteration to data enrichment**.

Priority areas include:

- genuine expected-goals data
- rolling xG for and against
- xG difference and trend
- home and away xG strength
- shots and shots on target
- chance-quality information
- player availability and line-ups where practical
- richer team-strength signals

A new challenger will only be built once genuinely new predictive information has been introduced.

It will then be evaluated against the frozen Model 2 benchmark using the same leakage-safe walk-forward methodology and pre-agreed promotion thresholds.

If richer data still fails to generate material uplift, the project will shift emphasis from continued historical model optimisation toward calibration, prospective monitoring and product capability.

### V2 — Agent orchestration

Refactor Football Copilot using LangChain and LangGraph to explore stateful workflows, tool orchestration, persistent analytical state and specialised football analytics agents.

### V3 — Cloud deployment

Containerise Football Copilot and deploy the application and supporting services to a cloud platform while keeping the LLM inference layer replaceable.

### Future — Fantasy Football Copilot

Extend the platform using player-level data to support expected points, transfer recommendations, captain selection, fixture difficulty, squad optimisation and differential identification.

---

## Why this project exists

Football Copilot is both a working football analytics application and a practical AI/ML learning project.

The aim is to explore the complete lifecycle:

**data → modelling → validation → application → prospective monitoring → model improvement → agent orchestration → cloud deployment**

Successful predictions are recorded.

Failed predictions are recorded.

Predictions are frozen before results are known.

And the model is evaluated against simple benchmarks rather than accuracy being viewed in isolation.

---

## Follow the experiment

**[Gameweek 2: Completed analysis](gameweeks/GW02.md)**

**[Gameweek 1: Completed analysis](gameweeks/GW01.md)**

---

## Model Development

The live 2026/27 experiment is also being used to identify modelling questions that can be tested rigorously against historical data.

### Model 3A: Dixon-Coles Challenger

After GW2, Model 2 had produced 1-1 as the modal scoreline in 16 of 20 live predictions while selecting Draw as the most likely 1X2 outcome in none of them.

This led to a controlled Dixon-Coles experiment across 1,100 historical walk-forward predictions.

Dixon-Coles reduced the historical modal 1-1 rate from 71.18% to 64.55%, but did not improve the zero-draw behaviour. Accuracy remained 52.18%, while Log Loss and Brier Score deteriorated marginally.

**Decision: Model 3A rejected for shadow promotion.**

The result has redirected the next investigation towards xG separation and draw probability calibration.

[Read the Model 3A experiment](model3-dixon-coles.md)

### Model 3B: Draw Calibration

The Model 3A results suggested that low-score dependence alone did not explain Model 2's draw behaviour.

A second controlled experiment therefore tested whether Model 2 contained useful information about draw likelihood but systematically positioned Draw too low relative to Home and Away.

Model 3B applied a training-fitted multiplier to Model 2's draw probability before renormalising the Home / Draw / Away probabilities.

The multiplier was learned only from historical training data within the walk-forward process.

Across the three historical test periods, the fitted multipliers were:

- 2023/24: 0.9824
- 2024/25: 0.9820
- 2025/26: 1.0117

The mean multiplier was approximately **0.992**, effectively indicating that the optimiser did not support materially increasing Model 2's draw probabilities.

Across 1,100 historical test fixtures:

| Metric | Model 2 | Model 3B |
| --- | ---: | ---: |
| Accuracy | 52.18% | 52.18% |
| Log Loss | 1.0037 | 1.0037 |
| Brier Score | 0.5999 | 0.5999 |
| Predicted draws | 0 | 0 |

**Decision: Model 3B rejected.**

The experiment provided useful evidence that simply increasing Draw probability globally would not solve the underlying problem.

This led to Model 4, which tested a much larger structural change by predicting Home / Draw / Away directly.

## Model 4: Direct 1X2 Gradient Boosting

Following the results from Model 3A and Model 3B, Model 4 deliberately tested a much larger structural change.

Instead of estimating home and away goals and deriving match probabilities through a Poisson score matrix, Model 4 predicted **Home / Draw / Away directly** using gradient boosting.

The challenger retained the existing Model 2 feature base and added:

- leakage-safe pre-match Elo
- relative-strength features
- form-acceleration features
- attack and defence trend features
- nonlinear interactions through gradient boosting

### Pre-agreed success threshold

The performance hurdle was defined before running the experiment.

| Metric | Model 2 reference | Model 4 target |
| --- | ---: | ---: |
| Accuracy | 52.18% | >= 54.00% |
| Log Loss | 1.0037 | <= 0.980 |
| Brier Score | 0.5999 | <= 0.585 |

### Result

The controlled walk-forward backtest covered 1,100 test fixtures.

| Metric | Model 2 | Model 4 | Change |
| --- | ---: | ---: | ---: |
| Accuracy | 52.18% | 51.27% | -0.91pp |
| Log Loss | 1.0037 | 1.0319 | worse by 0.0281 |
| Brier Score | 0.5999 | 0.6158 | worse by 0.0159 |
| Predicted draws | 0 | 24 | behavioural change |
| Draw recall | 0.00% | 2.61% | |

Model 4 failed all three pre-agreed success thresholds and was worse than Model 2 on Accuracy, Log Loss and Brier Score in every historical test season.

Although the direct classifier began predicting some draws, this did not translate into improved overall performance.

**Decision: REJECT Model 4.**

Model 2 remains the frozen production baseline.

The combined evidence from Models 3A, 3B and 4 suggests that the next meaningful improvement is more likely to come from **genuinely new predictive information** than another algorithm operating on the existing match-history feature set.

The next modelling phase will therefore focus on **data enrichment**, beginning with expected-goals and related chance-quality data.

[Read the full Model 4 experiment](model4-direct-1x2.md)

## Football Copilot repository

The full source code, modelling pipeline and development history are available in the GitHub repository.

**[View the Football Copilot repository](https://github.com/dhandan/football-copilot)**