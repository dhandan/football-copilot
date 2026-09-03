# Model 4: Direct 1X2 Gradient Boosting

## Why Model 4 was tested

After Model 3A and Model 3B failed to produce meaningful improvements over Model 2, the next experiment deliberately moved away from incremental Poisson adjustments.

Model 4 tested a structurally different approach:

- predict Home / Draw / Away directly
- use gradient boosting rather than derive 1X2 probabilities from independent Poisson goal models
- retain the existing Model 2 feature base
- add leakage-safe pre-match Elo
- add relative-strength and form-acceleration features
- evaluate using the same walk-forward historical framework

The objective was to test a large structural challenger rather than continue with small Model 3 variants.

## Pre-agreed success threshold

The success threshold was defined before running the experiment.

Against the controlled Model 2 benchmark:

| Metric | Model 2 reference | Model 4 target |
| --- | ---: | ---: |
| Accuracy | 52.18% | >= 54.00% |
| Log Loss | 1.0037 | <= 0.980 |
| Brier Score | 0.5999 | <= 0.585 |

A meaningful challenger needed to improve both headline outcome accuracy and probability quality.

## Model architecture

### Model 2

```text
pre-match features
        |
        v
home goal model + away goal model
        |
        v
independent Poisson score matrix
        |
        v
P(Home) / P(Draw) / P(Away)
```

### Model 4

```text
existing Model 2 features
        +
pre-match Elo
        +
relative-strength features
        +
form acceleration
        +
attack / defence trend features
        |
        v
direct gradient-boosted multiclass model
        |
        v
P(Home) / P(Draw) / P(Away)
```

## Walk-forward design

The historical seasons remained:

- 2021/22
- 2022/23
- 2023/24
- 2024/25
- 2025/26

Evaluation remained strictly walk-forward:

- 2023/24 trained on 2021/22 and 2022/23
- 2024/25 trained on all prior seasons
- 2025/26 trained on all prior seasons

No future fixture information was used to construct the Model 4 pre-match features.

## Results

The controlled backtest contained 1,100 test fixtures.

### Overall results

| Metric | Model 2 | Model 4 | Change |
| --- | ---: | ---: | ---: |
| Accuracy | 52.18% | 51.27% | -0.91pp |
| Log Loss | 1.0037 | 1.0319 | worse by 0.0281 |
| Brier Score | 0.5999 | 0.6158 | worse by 0.0159 |
| Predicted draws | 0 | 24 | behavioural change |
| Correct predicted draws | 0 | 7 | |
| Draw recall | 0.00% | 2.61% | |
| Draw precision | N/A | 29.17% | |

Model 4 failed all three pre-agreed success thresholds.

### Results by season

| Season | Model 2 Accuracy | Model 4 Accuracy | Model 2 Log Loss | Model 4 Log Loss | Model 2 Brier | Model 4 Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023/24 | 54.72% | 53.89% | 0.9726 | 1.0034 | 0.5773 | 0.5926 |
| 2024/25 | 52.43% | 51.62% | 1.0003 | 1.0334 | 0.5986 | 0.6180 |
| 2025/26 | 49.46% | 48.38% | 1.0375 | 1.0582 | 0.6233 | 0.6363 |

Model 4 was worse than Model 2 in every test season on all three primary metrics.

## Draw behaviour

One reason for testing a direct 1X2 classifier was Model 2's unusual draw behaviour.

Across the controlled backtest:

- actual draws: 268
- Model 2 predicted draws: 0
- Model 4 predicted draws: 24
- Model 4 correct predicted draws: 7
- Model 4 draw recall: 2.61%
- Model 4 draw precision: 29.17%

Model 4 therefore changed the classification behaviour and did begin to predict draws.

However, this did not improve overall model performance.

The experiment demonstrates that solving the zero-draw symptom alone does not solve the underlying prediction problem.

## Feature importance

The strongest Model 4 feature was:

- EloDifference

Other relatively important features included:

- HomeElo
- SeasonPPGDifference
- TenMatchGoalDifferenceGap
- HomeAttackAcceleration
- RecentVsSeasonAway
- Away10GoalsAgainst
- HomeSeasonGoalDifferencePG

The feature-importance results suggest that dynamic team-strength information contains predictive signal.

However, the additional signal was not sufficient to make the overall Model 4 architecture outperform Model 2.

Elo remains a candidate input for future richer-data models.

## Decision

**REJECT Model 4.**

Model 4:

- failed all three agreed success thresholds
- reduced historical accuracy
- worsened Log Loss
- worsened Brier Score
- predicted some draws but did not discriminate them effectively
- underperformed Model 2 in every test season

Model 2 therefore remains the frozen production baseline.

## What the modelling experiments now tell us

The sequence from Model 3A through Model 4 provides evidence against several simple explanations for the current performance ceiling.

### Model 3A: Dixon-Coles

Dixon-Coles changed the low-score probability structure and reduced some of the extreme 1-1 modal-score concentration.

However, it did not improve overall prediction quality.

### Model 3B: Draw calibration

A training-fitted draw multiplier remained close to 1.0.

The calibration experiment produced no practical performance improvement.

### Model 4: Direct 1X2 gradient boosting

Model 4 removed the independent-Poisson structural constraint altogether.

It used direct multiclass outcome prediction, nonlinear feature interactions, Elo, relative strength and form acceleration.

Despite the structural change, performance deteriorated.

## Conclusion

The evidence increasingly suggests that the next meaningful improvement is more likely to come from **new predictive information** rather than another algorithm operating on the same underlying match-history data.

This marks the end of algorithm-only experimentation using the current feature set.

## Next phase: data enrichment

The next phase of Football Copilot will focus on enriching the modelling dataset.

The highest-priority candidate is genuine expected-goals data.

Potential new information includes:

- rolling xG for
- rolling xG against
- xG difference
- home xG strength
- away xG strength
- 5-match xG form
- 10-match xG form
- xG form acceleration
- actual goals versus expected goals
- shots
- shots on target
- chance-quality indicators
- player availability
- starting line-ups
- richer team-strength signals

The next major challenger should only be built after genuinely new predictive information has been added.

## Hard-stop principle

Football Copilot will not create new models purely to increase iteration count.

A challenger should only be pursued when there is a credible reason to expect meaningful improvement.

If richer data still fails to materially improve probability quality and predictive accuracy, the project will shift emphasis toward:

- probability calibration
- prospective live evaluation
- model monitoring
- explanation
- agentic product capability
- deployment architecture

rather than continued historical accuracy optimisation.

Model 4 therefore marks a deliberate decision point in the project rather than another failed experiment.