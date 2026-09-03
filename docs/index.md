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

Model development continues separately through controlled historical experiments and out-of-time validation.

Following the draw and scoreline behaviour observed during GW1 and GW2, four challenger experiments have now been completed:

- **Model 3A — Dixon-Coles:** tested low-score dependence.
- **Model 3B — Draw calibration:** tested whether Model 2's draw probabilities required systematic adjustment.
- **Model 4 — Direct 1X2 Gradient Boosting:** tested a structurally different direct outcome model incorporating Elo and additional relative-strength features.
- **Model 5 — Observed xG Enrichment:** tested whether genuinely new chance-quality information improved the existing Model 2 Poisson architecture.

Models 3A, 3B and 4 were rejected during historical testing.

Model 5 was different.

It passed its pre-registered historical promotion hurdle, improving Accuracy, Log Loss and Brier Score across the controlled 730-fixture historical test.

It was therefore promoted to a stronger out-of-time validation against the untouched 2025/26 Premier League season.

On those 370 common evaluation fixtures, Model 5 again improved Log Loss and Brier Score, providing evidence that observed xG contained incremental predictive information.

However, its Accuracy deteriorated by 1.62 percentage points relative to Model 2, exceeding the pre-agreed one percentage point tolerance.

Model 5 therefore failed its out-of-time promotion gate and was not promoted over Model 2.

This result is an important part of the development process. Historical improvement alone was not treated as sufficient evidence for model replacement, and the failed future-season result will not be used to retrospectively tune Model 5.

The 2025/26 season has now been consumed as a holdout for this experiment.

Model 2 will continue generating the official live 2026/27 predictions through the initial five-Gameweek monitoring period.

### GW3 prospective milestone

Following completion of the challenger experiments, the project returned to the prospective 2026/27 experiment with Model 2 remaining unchanged.

The official Gameweek 3 predictions were generated and frozen on **3 September 2026**, before the first GW3 fixture.

The process identified and corrected a stale GW2 fixture snapshot before any official GW3 predictions were saved. The correct GW3 fixture window was then retrieved, validated and passed through a dry run before the official prediction snapshot was created.

The frozen GW3 snapshot contains:

- 10 fixtures
- 10 successful predictions
- 0 prediction failures
- 3 fixtures using promoted-team cold-start priors
- 0 Draw selections as the highest-probability 1X2 outcome
- 6 fixtures with 1-1 as the individual modal scoreline

The official snapshot is stored at:

`data/live/predictions/2026_27_gw03_predictions.csv`

and was committed before the Gameweek began, preserving an auditable prospective record.

### What the challenger experiments have taught us

Models 3A through 5 tested increasingly different explanations for Model 2's limitations.

Model 3A showed that correcting low-score dependence did not materially improve the probabilities.

Model 3B showed that simply recalibrating Draw probability was not supported by the historical evidence.

Model 4 showed that replacing the Poisson architecture with a direct 1X2 gradient-boosting model did not improve performance using essentially the same underlying information.

Model 5 provided the strongest evidence of a potential improvement. Adding genuinely new observed xG information improved historical Accuracy, Log Loss and Brier Score and continued to improve probability quality on the untouched 2025/26 season. However, the improvement did not generalise to 1X2 accuracy strongly enough to pass the pre-registered promotion gate.

The collective conclusion is therefore **not to continue creating new model architectures simply to search for a higher backtest score**.

The next decision point will be after Gameweek 5, when Football Copilot will have accumulated approximately 50 genuinely prospective Premier League predictions.

At that point the project will perform a structured failure analysis to identify where Model 2 succeeds and fails, including areas such as prediction confidence, close Home/Away probabilities, Draw behaviour, promoted teams, home/away effects, disagreement with market benchmarks and other identifiable match characteristics.

Any subsequent challenger will be driven by evidence from that analysis and tested using historical data without retrospectively changing the frozen prospective baseline.

Where additional information is required, priority will remain on **free and reproducible data sources**. Potential enrichment includes derived rest and fixture-congestion features, Elo and other team-strength measures, richer match statistics, observed xG signals and, where reliable free sources exist, player-level or availability information.

This preserves a core Football Copilot design principle: model improvement should come from a clear hypothesis, genuinely useful information and out-of-sample evidence rather than repeated algorithm tuning.

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

Use evidence from historical testing, out-of-time validation and the prospective live experiment to improve the prediction capability without retrospectively changing the frozen live baseline.

Completed challenger experiments:

- Model 3A — Dixon-Coles low-score correction: **rejected**
- Model 3B — training-fitted draw calibration: **rejected**
- Model 4 — direct 1X2 gradient boosting with Elo: **rejected**
- Model 5 — observed xG enrichment: **passed historical promotion hurdle, failed out-of-time promotion gate**

Models 3A, 3B and 4 tested whether changes to the modelling architecture could extract materially better predictions from essentially the same underlying match-history information.

None produced sufficient improvement.

Model 5 therefore moved the project into **data enrichment** by introducing genuinely new observed expected-goals information while retaining the Model 2 Poisson architecture.

Historically, Model 5 improved all three primary metrics:

| Metric | Model 2 | Model 5 |
| --- | ---: | ---: |
| Accuracy | 53.5616% | **54.5205%** |
| Log Loss | 0.9867 | **0.9731** |
| Brier Score | 0.5881 | **0.5786** |

This passed the pre-registered historical promotion hurdle and advanced Model 5 to an untouched 2025/26 out-of-time test.

On the 370-fixture future-season cohort, Model 5 again improved probability quality but failed to reproduce the historical accuracy uplift:

| Metric | Model 2 | Model 5 |
| --- | ---: | ---: |
| Accuracy | **49.4595%** | 47.8378% |
| Log Loss | 1.0375 | **1.0323** |
| Brier Score | 0.6233 | **0.6196** |

The result provides evidence that observed xG adds useful information to the probability estimates, but not enough evidence to promote Model 5 as the new 1X2 prediction model.

The 2025/26 holdout will not now be used to tune Model 5 and then reused as untouched validation.

The immediate V1.5 priority returns to the **prospective 2026/27 experiment**.

Model 2 remains frozen as the official model through Gameweek 5. The live evidence accumulated across those five Gameweeks will then be reviewed alongside the completed challenger experiments before deciding the next modelling direction.

Potential future enrichment remains available, including:

- shots and shots on target
- richer chance-quality information
- player availability and line-ups
- player-level data
- additional team-strength signals

These are research directions rather than committed Model 6 features. A further challenger should only be created where there is a clear hypothesis and genuinely new predictive information rather than continuing model iteration for its own sake.

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

The combined evidence from Models 3A, 3B and 4 suggested that the next meaningful improvement was more likely to come from **genuinely new predictive information** than another algorithm operating on the existing match-history feature set.

This directly led to Model 5, which moved the project from algorithm iteration into **data enrichment** by introducing observed expected-goals information.

[Read the full Model 4 experiment](model4-direct-1x2.md)

## Model 5: Observed xG Enrichment

The results from Models 3A, 3B and 4 suggested that further changes to modelling architecture using essentially the same match-history information were unlikely to produce the material improvement being sought.

Model 5 therefore moved the research from **algorithm iteration to data enrichment**.

Rather than replacing Model 2's Poisson architecture, Model 5 retained the existing 25 Model 2 features and added 14 leakage-safe observed expected-goals features covering:

- rolling xG for and against
- five and ten-match xG form
- xG difference
- attacking xG trend
- defensive xG trend

This created a controlled test of whether genuinely new information about chance quality could improve the existing model.

### Historical validation

Model 5 was initially evaluated across a common 730-fixture historical cohort covering 2023/24 and 2024/25.

Promotion thresholds were defined before viewing the results:

| Metric | Promotion requirement |
| --- | ---: |
| Accuracy | >= 54.5% |
| Log Loss | <= 0.975 |
| Brier Score | <= 0.580 |
| Probability quality | Improve both Log Loss and Brier vs Model 2 |

Model 5 passed every threshold.

| Metric | Model 2 | Model 5 | Change |
| --- | ---: | ---: | ---: |
| Accuracy | 53.5616% | **54.5205%** | **+0.9589pp** |
| Log Loss | 0.9867 | **0.9731** | **+0.0136 improvement** |
| Brier Score | 0.5881 | **0.5786** | **+0.0095 improvement** |
| Predicted draws | 0 | 0 | No change |

**Historical decision: PROMOTE Model 5 to out-of-time validation.**

Importantly, this did not replace the frozen Model 2 used for the live 2026/27 experiment.

### Untouched 2025/26 validation

The historical result was then tested against the completely untouched 2025/26 Premier League season.

Because the historical xG source did not contain 2025/26, Understat fixture-level xG was acquired and transformed onto the historical FBref-derived xG scale using a provider bridge fitted exclusively on 2024/25 data.

The bridge and Model 5 feature set were frozen before performance was revealed.

After applying the existing Model 2 history requirements, the final common evaluation cohort contained:

**370 fixtures**

with zero missing values across the locked Model 5 features.

The out-of-time promotion criterion was also defined before viewing performance:

- Model 5 must improve Log Loss
- Model 5 must improve Brier Score
- Accuracy must not deteriorate by more than one percentage point

### Out-of-time result

| Metric | Model 2 | Model 5 | Change |
| --- | ---: | ---: | ---: |
| Accuracy | **49.4595%** | 47.8378% | **-1.6216pp** |
| Log Loss | 1.0375 | **1.0323** | **+0.0051 improvement** |
| Brier Score | 0.6233 | **0.6196** | **+0.0037 improvement** |
| Predicted draws | 0 | 0 | No change |

Model 5 reproduced its improvement in probability quality on genuinely future data.

However, the historical accuracy uplift did not generalise.

Accuracy deteriorated by **1.62 percentage points**, exceeding the pre-agreed one percentage point tolerance.

**Out-of-time decision: FAIL. Model 5 is not promoted over Model 2.**

### Why the result changed

Diagnostics showed that Model 5 changed Model 2's predicted 1X2 outcome on only:

**38 / 370 fixtures (10.27%)**

Those switches produced:

- 13 improved predictions
- 19 worsened predictions
- 6 predictions that changed but remained incorrect

The net effect was exactly:

**-6 correct fixtures**

All 38 classification switches occurred where Model 2 confidence was below 50%, and 34 occurred below 45%.

Observed xG therefore did not destabilise high-confidence predictions. Its classification impact was concentrated around uncertain fixtures.

Every classification change was also between Home and Away:

- Home → Away: 20
- Away → Home: 18
- switches into Draw: 0

### Probability quality still improved

Despite the lower classification accuracy, Model 5 produced better match-level:

- Log Loss on **196 / 370 fixtures (52.97%)**
- Brier Score on **194 / 370 fixtures (52.43%)**

The improvement was strongest for actual draws.

Across the 101 drawn fixtures, mean improvement was:

- Log Loss: **+0.012584**
- Brier Score: **+0.010503**

Yet Model 5 still selected Draw as the highest-probability outcome zero times.

This reinforces an important distinction between improving the probability assigned to an outcome and changing the final argmax classification.

### Model 5 conclusion

Model 5 demonstrated that observed xG contains incremental predictive information.

Its probability improvements replicated from historical testing into an untouched future season.

Its classification accuracy improvement did not.

The pre-registered promotion criterion was therefore correctly failed, and the 2025/26 holdout will not now be used to tune Model 5 and subsequently presented as untouched validation.

**Model 2 remains the frozen official model for the initial 2026/27 live experiment.**

[Read the full Model 5 experiment](model5-observed-xg.md)

## Football Copilot repository

The full source code, modelling pipeline and development history are available in the GitHub repository.

**[View the Football Copilot repository](https://github.com/dhandan/football-copilot)**