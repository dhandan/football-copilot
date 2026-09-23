# Post-GW5 Model Deep Dive

## Objective

After GW5, we performed a structured deep dive into Model 2 to understand its weaknesses and determine whether a materially stronger model could be developed for GW6.

Model 2 remained the frozen baseline throughout the investigation.

Controlled historical baseline:

- Accuracy: 52.18%
- Log Loss: 1.0037
- Brier Score: 0.5999
- Test population: 1,100 out-of-time Premier League fixtures

The target was not simply to create another model, but to find a challenger that produced a meaningful and repeatable improvement out of sample.

---

## H1-H6 Diagnosis

Six hypotheses were investigated.

### H1 - Draw discrimination is weak

Supported.

Model 2 assigned reasonable probability to draws but almost never made a draw its most likely outcome.

Across the 1,100-match test:

- Actual draws: 268
- Model 2 predicted draws: 0

Explicit draw modelling was subsequently tested, but did not improve overall prediction quality.

### H2 - Expected-goal / scoreline distribution is too compressed

Supported.

Model 2 produced an excessive concentration around similar expected-goal totals and 1-1 modal scorelines.

Alternative goal-model specifications increased scoreline diversity, but this did not translate into better outcome prediction.

### H3 - Team-strength adaptation is too slow

Partially supported.

Model 2 already contains recent 5/10-match form, venue form and season-strength features.

Additional shots, shots-on-target and underlying-performance features were tested. They provided useful information but did not consistently improve out-of-time performance.

### H4 - Independent Poisson modelling is limiting performance

Partially supported.

Alternative approaches included:

- Dixon-Coles
- Non-linear goal models
- Direct H/D/A classification
- Gradient boosting
- Ensemble models

Changing the modelling architecture alone did not produce a material improvement.

### H5 - Promoted-team cold-start priors can improve

Investigated but not identified as the main performance constraint.

The existing 75% translated Championship / 25% Premier League prior remained the strongest tested approach.

### H6 - Errors cluster in identifiable situations

Supported.

Errors were concentrated in:

- low-confidence fixtures
- closely matched teams
- draw-heavy situations
- fixtures where football features and market expectations disagreed

This became an important diagnostic finding for future development.

---

## Model 6

Model 6 moved beyond incremental Model 2 optimisation.

It introduced a richer information architecture containing:

- existing Model 2 features
- opening-market probabilities
- xG/xGA
- shots and shots on target
- exponentially weighted form
- venue performance
- matchup differences
- rest/context information

Closing-market information was excluded from prediction features.

All testing remained chronological and out of time.

### Opening Market Benchmark

The strongest new signal was the pre-match opening market.

Across 1,100 OOT fixtures:

| Model | Accuracy | Log Loss | Brier |
|---|---:|---:|---:|
| Model 2 | 52.18% | 1.0037 | 0.5999 |
| Opening Market | 54.09% | 0.9682 | 0.5763 |

Opening Market therefore improved:

- Accuracy by +1.91 percentage points
- Log Loss by 0.0355
- Brier by 0.0236

Probability performance improved across all three OOT seasons.

---

## Model 6 Variants

Several approaches were tested to determine whether our football information could improve on the market signal.

### Model 6B - Market + football gradient boosting

Rejected.

Accuracy fell to 48.91% and probability metrics deteriorated substantially.

### Model 6C - Market prior + football residual model

Rejected.

Accuracy: 54.18%

This represented only one additional correct prediction across 1,100 fixtures and produced worse Log Loss and Brier than the opening market.

The improvement was not material.

### Model 6D - Selective market override

Rejected.

A model attempted to identify market mistakes and selectively replace uncertain market predictions.

Accuracy fell to 53.00%.

### Model 6E - Explicit draw architecture

Rejected.

A two-stage model separately estimated draw probability while retaining the market's conditional home/away judgement.

Results:

- Accuracy: 53.82%
- Log Loss: 0.9815
- Brier: 0.5822
- Predicted draws: 17

Although it generated draw predictions, all primary metrics were worse than the opening market.

No further Model 6 variants were developed to avoid research overfitting.

---

## Market Error Diagnosis

The opening market's errors were analysed separately.

Important findings included:

- Market accuracy: 54.09%
- Actual draws: 268
- Predicted draws: 0
- Strong market favourites performed considerably better than uncertain fixtures.
- Errors clustered heavily in competitive fixtures.
- When football and market signals agreed, market accuracy was 57.67%.
- When they disagreed, market accuracy fell to 45.51%.

This suggests that the remaining problem is increasingly one of missing information rather than simply model architecture.

---

## Protected Historical GW6 Test

Before viewing results, a separate historical GW6 holdout was frozen:

- 2023/24 GW6: 10 fixtures
- 2024/25 GW6: 10 fixtures
- 2025/26 GW6: 10 fixtures
- Total: 30 fixtures

Opening Market results:

- 12/30 correct
- Accuracy: 40.00%
- Log Loss: 0.9960
- Brier: 0.6000
- Actual draws: 9
- Predicted draws: 0

Season accuracy:

- 2023/24: 50%
- 2024/25: 40%
- 2025/26: 30%

The 30-match sample is small and does not outweigh the 1,100-match OOT evidence, but it provides an important warning against immediately promoting the market model.

The holdout will not be used for further tuning.

---

## GW6 Champion / Challenger

For live 2026/27 GW6:

**Champion: Model 2**

Model 2 remains the production model.

**Shadow Challenger: Opening Market**

Opening Market will run independently alongside Model 2.

Both models will be frozen before GW6 matches are played and evaluated using:

- Accuracy
- Log Loss
- Brier Score

The GW7 decision will consider the complete evidence rather than making a decision from only 10 live fixtures:

1. 1,100-match historical OOT test
2. 30-match protected historical GW6 test
3. prospective live GW6 performance

---

## Main Learning

The deep dive did not demonstrate that a different algorithm using our existing football information can materially outperform Model 2.

The largest improvement came from introducing a genuinely new information source: opening-market expectations.

This suggests that the next major opportunity is not further algorithm optimisation against the same data.

It is improving the information available to the model.

Potential future signals include:

- player availability and injuries
- suspensions
- starting XI strength
- player-level performance
- transfers
- manager changes
- schedule congestion
- European competition/rest effects
- richer opponent-adjusted performance
- improved promoted-team information

The market can also be used as a diagnostic benchmark: fixtures where Model 2 and the market strongly disagree provide a useful population for investigating what information our football model is missing.

---

## Next Steps

1. Freeze Model 2 and Opening Market for GW6.
2. Generate both sets of predictions before kickoff.
3. Record H/D/A probabilities as well as predicted result.
4. Evaluate both after GW6 using Accuracy, Log Loss and Brier.
5. Make the formal champion/challenger decision for GW7 using cumulative evidence.
6. Do not tune against the revealed 30-match historical GW6 holdout.
7. Shift the next model-development cycle from algorithm optimisation toward information enrichment.
8. Introduce LangChain/LangGraph after the GW6 model lifecycle is frozen, separating prediction from the conversational/orchestration layer.

## Current Status

Model 2 remains production champion.

Opening Market is the strongest challenger discovered during the post-GW5 deep dive.

Models 3A/3B/4/5 and Model 6B/6C/6D/6E have not demonstrated sufficient evidence for promotion.

The project now moves from retrospective model optimisation into a prospective champion/challenger experiment for GW6.