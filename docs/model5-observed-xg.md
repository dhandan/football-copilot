# Model 5: Observed xG Enrichment

## Objective

Model 5 tested whether adding observed expected-goals (xG) information to the existing Model 2 architecture could improve Premier League match prediction.

The experiment deliberately retained the existing Model 2 modelling framework.

Model 5 therefore remained a Poisson goal model and added observed xG features to the existing Model 2 feature set rather than replacing the modelling architecture.

The hypothesis was:

> Observed xG may contain information about underlying team performance that is not fully represented by goals, points and existing form features.

The experiment was conducted as a controlled challenger and did not alter the frozen live Model 2 used for the 2026/27 prospective experiment.


## Model Design

### Model 2

Model 2 uses 25 pre-match features covering:

- recent goals scored and conceded
- recent points per game
- 10-match form
- venue performance
- season performance
- relative team-strength measures
- attack-versus-defence measures

Separate Poisson regression models predict home and away goals.

The resulting expected-goal estimates are converted into Home / Draw / Away probabilities using an independent Poisson score matrix.


### Model 5

Model 5 retains all 25 Model 2 features and adds 14 observed-xG features:

- HomeXGForAvg5
- HomeXGAgainstAvg5
- AwayXGForAvg5
- AwayXGAgainstAvg5
- HomeXGForAvg10
- HomeXGAgainstAvg10
- AwayXGForAvg10
- AwayXGAgainstAvg10
- HomeXGDifferenceAvg5
- AwayXGDifferenceAvg5
- HomeXGForTrend
- AwayXGForTrend
- HomeXGAgainstTrend
- AwayXGAgainstTrend

All rolling xG features are calculated using prior matches only.

The modelling architecture remains:

- `PoissonRegressor(alpha=0.1, max_iter=1000)`
- separate home and away goal models
- independent Poisson H/D/A probability calculation
- maximum score matrix of eight goals

This allowed the incremental value of observed xG to be tested without simultaneously changing the model architecture.


## Historical xG Data

Historical observed xG was sourced from a Kaggle FBref-derived Premier League dataset covering 2021/22 to 2024/25.

The source contained 3,800 team-perspective rows across five seasons.

For the Football Copilot overlap period of 2021/22 to 2024/25:

- 1,520 canonical Premier League fixtures were expected
- 1,520 fixtures were reconstructed from the xG source
- 1,520 / 1,520 fixtures reconciled successfully
- no internal xG consistency discrepancies were found

The source was therefore accepted for historical Model 5 experimentation.


## Leakage-Safe Feature Engineering

Observed xG was transformed into rolling pre-match features.

Rolling calculations use:

`shift(1).rolling(...)`

before calculating each fixture's features.

This ensures that the current fixture's observed xG cannot enter its own prediction.

An independent validation subsequently tested 120 feature calculations and passed:

**120 / 120 leakage checks.**


## Historical Controlled Experiment

The initial controlled Model 5 experiment covered the 2023/24 and 2024/25 seasons.

The common Model 2 / Model 5 cohort contained:

**730 fixtures**

rather than the full 760 fixtures.

This resulted from the existing Model 2 minimum Premier League history rules for promoted teams and was retained rather than retrofitting the pipeline to increase the sample.


## Pre-Registered Historical Promotion Criteria

Before viewing Model 5 results, the following promotion hurdle was locked:

- Accuracy >= 54.5%
- Log Loss <= 0.975
- Brier <= 0.580
- Model 5 must improve both Log Loss and Brier versus Model 2


## Historical Results

| Metric | Model 2 | Model 5 |
| --- | ---: | ---: |
| Accuracy | 53.5616% | **54.5205%** |
| Log Loss | 0.9867 | **0.9731** |
| Brier | 0.5881 | **0.5786** |
| Predicted draws | 0 | 0 |

Accuracy improved by:

**+0.9589 percentage points**

Log Loss improved by:

**0.0136**

Brier improved by:

**0.0095**

All pre-registered historical promotion criteria passed.

### Historical verdict

**PROMOTE to out-of-time validation.**

This did not mean replacing the frozen live Model 2.

It meant Model 5 had earned a stronger future-season validation test.


## Historical Uplift Diagnostics

Model 5 changed the Model 2 classification on 59 of the 730 fixtures.

Of those switches:

- 27 improved the prediction
- 20 worsened the prediction
- 12 changed the prediction but remained incorrect

The net classification effect was:

**+7 correct fixtures**

All classification switches occurred on fixtures where Model 2 confidence was below 50%.

Model 5 also produced better:

- Log Loss on 410 / 730 fixtures
- Brier score on 412 / 730 fixtures

The historical result suggested that observed xG was adding information primarily around uncertain fixtures rather than destabilising strong Model 2 predictions.


# 2025/26 Out-of-Time Validation

The historical result was then subjected to a stronger test using an untouched future Premier League season.

No Model 5 parameters or features were tuned against 2025/26 before evaluation.


## 2025/26 xG Provider

The historical FBref-derived dataset did not contain 2025/26.

Understat was therefore used to acquire fixture-level observed xG for the complete 2025/26 Premier League season.

The Understat dataset contained:

**380 / 380 completed Premier League fixtures.**


## Provider Compatibility

Because historical Model 5 features were based on FBref-derived xG, raw Understat xG was not assumed to be directly interchangeable.

A provider comparison was performed using the overlapping 2024/25 season.

Fixture reconciliation was:

**380 / 380**

Understat and FBref xG were strongly correlated at approximately:

**0.932**

However, raw Understat xG values were approximately 12.9% higher on average.

A provider bridge was therefore fitted using 2024/25 only.


## Frozen Provider Bridge

The resulting transformations were:

### Home

`FBref-scale xG = 0.124993 + 0.806923 × Understat xG`

### Away

`FBref-scale xG = 0.123716 + 0.808671 × Understat xG`

The bridge reduced combined provider MAE from:

**0.2883**

to:

**0.2264**

The bridge was frozen before evaluating 2025/26.

The transformed values are therefore treated as:

> Understat xG transformed onto the historical FBref scale.

They are not treated as literal FBref xG.


## 2025/26 Fixture Reconciliation

The 2025/26 Understat dataset reconciled:

**380 / 380 fixtures**

against the canonical Football Copilot Premier League results.


## Cross-Season Feature History

An important preprocessing issue was identified before revealing Model 5 performance.

The original 2025/26 xG feature build reset rolling history at the start of the season.

The historical Model 5 feature pipeline, however, carried team xG history across season boundaries.

Using a season reset would therefore have changed the meaning of the features between historical training and future testing.

The 2025/26 feature pipeline was rebuilt using the complete chronology:

**2021/22–2024/25 historical FBref-scale xG → 2025/26 bridged Understat xG**

Rolling features were then calculated chronologically across team history before extracting the 2025/26 fixtures.

This preserved the historical Model 5 feature semantics while remaining leakage-safe.


## Locked Evaluation Cohort

The final Model 2 / Model 5 common 2025/26 cohort contained:

**370 fixtures**

Model 5 contained all 380 fixtures.

The ten excluded fixtures were Sunderland's first ten Premier League fixtures because the existing Model 2 minimum-history rule excluded them.

No Model 2 fixtures were missing from Model 5.

Feature completeness on the locked cohort was:

**0 missing values across all 14 Model 5 xG features.**

The 370-fixture cohort was frozen before viewing performance.


## Out-of-Time Generalisation Criteria

Before revealing 2025/26 performance, the following criterion was locked.

Model 5 must:

- improve Log Loss versus Model 2
- improve Brier versus Model 2
- avoid an accuracy deterioration greater than 1 percentage point

Draw prediction was reported separately but was not made a promotion requirement.


# Untouched 2025/26 Results

| Metric | Model 2 | Model 5 | Change |
| --- | ---: | ---: | ---: |
| Accuracy | **49.4595%** | 47.8378% | **-1.6216pp** |
| Log Loss | 1.0375 | **1.0323** | **+0.0051 improvement** |
| Brier | 0.6233 | **0.6196** | **+0.0037 improvement** |
| Predicted draws | 0 | 0 | No change |

Actual draws:

**101 / 370**

Model 5 therefore reproduced the historical improvement in probability quality but did not reproduce the historical accuracy uplift.

The accuracy deterioration exceeded the pre-registered one percentage point tolerance.


## Out-of-Time Verdict

**FAIL: Model 5 did not meet the locked 2025/26 generalisation criterion.**

Model 5 was therefore not promoted over Model 2.


# Out-of-Time Diagnostics

The failed promotion result was analysed without retraining or modifying either model.


## Classification Changes

Model 2 correctly predicted:

**183 / 370 fixtures (49.4595%)**

Model 5 correctly predicted:

**177 / 370 fixtures (47.8378%)**

Net change:

**-6 correct fixtures**

Model 5 changed Model 2's classification on:

**38 / 370 fixtures (10.2703%)**

Those switches consisted of:

- 13 improved predictions
- 19 worsened predictions
- 6 changed but remained incorrect

Net switch effect:

**-6**

This completely explains the observed accuracy deterioration.


## Confidence of Changed Predictions

Every one of the 38 classification switches occurred where Model 2 confidence was below 50%.

Furthermore:

**34 / 38**

occurred below 45% confidence.

Mean confidence on switched fixtures was:

- Model 2: 40.45%
- Model 5: 40.57%

Observed xG therefore did not destabilise high-confidence Model 2 predictions.

Its classification impact was concentrated around uncertain fixtures.


## Direction of Classification Changes

Every classification switch was between Home and Away:

- Home → Away: 20
- Away → Home: 18

There were:

**0 switches into Draw.**

This provides further evidence that the existing draw-classification issue is separate from the observed-xG enrichment experiment.


## Probability Quality

Despite the classification deterioration, Model 5 improved match-level Log Loss on:

**196 / 370 fixtures (52.9730%)**

and improved match-level Brier on:

**194 / 370 fixtures (52.4324%)**

Mean improvement was:

- Log Loss: +0.005136
- Brier: +0.003654

The aggregate probability improvement was therefore reasonably broad-based rather than being caused solely by a small number of extreme fixtures.


## Probability Improvement by Actual Result

| Actual result | Matches | Mean Log Loss improvement | Mean Brier improvement |
| --- | ---: | ---: | ---: |
| Home | 157 | +0.007987 | +0.006288 |
| Draw | 101 | **+0.012584** | **+0.010503** |
| Away | 112 | -0.005578 | -0.006215 |

Model 5 improved probability quality for Home and Draw outcomes but deteriorated for Away outcomes.

The largest improvement occurred on actual draws.

However, Model 5 still predicted zero draws.

This distinction is important.

Increasing the probability assigned to the correct draw outcome can improve Log Loss and Brier without making Draw the highest-probability class.


# Conclusion

Model 5 demonstrated that observed xG contains incremental predictive information.

Historically, it improved:

- Accuracy
- Log Loss
- Brier score

On the untouched 2025/26 season it again improved:

- Log Loss
- Brier score

However, the historical classification accuracy uplift did not generalise.

Observed xG changed only low-confidence Home/Away decisions and produced six fewer correct classifications on the future-season test.

The pre-registered promotion criterion was therefore correctly failed.


## Decision

**Model 5 is not promoted over Model 2.**

The failed out-of-time result will not be used to tune Model 5.

The 2025/26 season has now been consumed as a holdout and will not subsequently be presented as untouched validation for a modified version of the same experiment.

The frozen Model 2 remains the official live model through Gameweek 5 of the 2026/27 prospective experiment.


## Key Learning

The experiment demonstrates an important distinction between:

**better probability estimation**

and:

**better argmax classification accuracy**

Observed xG improved the quality of Model 5's probability distribution, including its probabilities for actual draws, but this did not translate into more correct 1X2 classifications.

The result also demonstrates why historical challenger performance should not be treated as sufficient evidence for model promotion.

A model can pass historical validation and still fail to reproduce that improvement on genuinely future data.