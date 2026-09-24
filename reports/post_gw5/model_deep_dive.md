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

## GW6 Operational Workflow

Following the post-GW5 deep dive, the existing live prediction workflow was reviewed before making any implementation changes.

### Champion: Model 2

Model 2 remains the production Champion for GW6.

The existing production workflow remains unchanged:

1. `scripts/fetch_live_fixtures.py`
   - Fetches the live Premier League fixture snapshot.
   - Uses the official API matchday.
   - Normalises team names.
   - Saves a timestamped fixture snapshot.

2. `scripts/generate_live_gameweek_predictions.py`
   - Reads the latest fixture snapshot.
   - Generates Model 2 predictions through `predict_fixture()`.
   - Saves the protected official pre-match prediction snapshot.
   - Prevents an existing official snapshot from being overwritten.

3. `scripts/evaluate_gameweek.py`
   - Joins the frozen prediction snapshot to the actual results by `FixtureId`.
   - Evaluates Accuracy, Log Loss and Brier Score.
   - Retains the existing Model 2 goal, scoreline, draw and cold-start diagnostics.

No changes are required to this production workflow for GW6.

### Shadow Challenger: Opening Market

The Opening Market candidate remains the formal Shadow Challenger for GW6.

It will run separately from the production Model 2 workflow and will not modify the official Model 2 prediction or evaluation files.

The existing historical market pipeline already provides the required probability calculation:

`1X2 odds -> implied probabilities -> remove bookmaker overround -> normalised H/D/A probabilities`

For the live experiment, a separate immutable pre-match market snapshot will be frozen for the same GW6 fixtures.

The market snapshot must record:

- Fixture identifier
- Home and away teams
- Odds capture timestamp
- Odds source
- Home, draw and away odds
- Normalised Home, Draw and Away probabilities
- Modal 1X2 prediction

The live market source and input format must be verified before implementation. No assumptions about the live odds schema will be built into the project.

### GW6 Evaluation

After GW6 results are available, Champion and Challenger will be compared on the same fixtures using:

- 1X2 Accuracy
- Log Loss
- Brier Score

Model 2-specific outputs such as expected goals and exact scorelines will remain part of the existing Model 2 evaluation but will not be required from the market challenger.

GW6 is prospective evidence rather than a standalone promotion test.

The GW7 Champion/Challenger decision will consider the combined evidence from:

1. 1,100-match historical temporal out-of-time testing
2. 30-match sealed historical GW6 simulation
3. Live prospective 2026/27 GW6 performance

No further Model 6 algorithm variants will be developed before this prospective test. The next modelling research cycle will focus on additional information rather than further tuning of the existing feature/model combinations.

---

## Follow-up: Opening Market Comparison and Frozen Shadow Challenger

**Status: Completed 24 September 2026**

The original post-GW5 deep dive identified the Opening Market as the next Shadow Challenger and deliberately left the live market implementation open until a suitable source and schema had been verified.

Subsequent work completed that investigation and refined the challenger design before any GW6 results were available.

The original analysis above remains unchanged as the contemporaneous post-GW5 decision record. This follow-up records the additional development work that followed it.

### GW1-GW5 Opening Market Comparison

The 50 frozen Model 2 predictions from GW1-GW5 were compared fixture-by-fixture with historical opening-market probabilities.

The market evidence came from the frozen Football-Data 2026/27 dataset using:

- `AvgH`
- `AvgD`
- `AvgA`

These are the average opening Home, Draw and Away decimal odds.

Closing prices (`AvgCH`, `AvgCD`, `AvgCA`) were deliberately excluded from the experiment.

For each fixture, the opening prices were converted to fair probabilities using:

`decimal odds -> inverse probabilities -> normalise bookmaker overround -> H/D/A probabilities`

This produced the following aggregate comparison across the 50-match GW1-GW5 development period:

| Signal | Accuracy | Log Loss | Brier Score |
| --- | ---: | ---: | ---: |
| Model 2 | 42.0% | 1.0598 | 0.6371 |
| Opening Market | 44.0% | 1.0522 | 0.6340 |

The market was narrowly stronger in aggregate, but neither signal dominated consistently across individual Gameweeks.

### Gameweek Comparison

| Gameweek | Model 2 Accuracy | Market Accuracy | Model 2 Log Loss | Market Log Loss | Model 2 Brier | Market Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GW1 | 50.0% | 60.0% | 1.0212 | 0.9859 | 0.6108 | 0.5757 |
| GW2 | 50.0% | 50.0% | 1.0612 | 0.9759 | 0.6378 | 0.5800 |
| GW3 | 30.0% | 30.0% | 1.1209 | 1.0466 | 0.6761 | 0.6346 |
| GW4 | 30.0% | 40.0% | 1.1292 | 1.1731 | 0.6874 | 0.7313 |
| GW5 | 50.0% | 40.0% | 0.9665 | 1.0796 | 0.5732 | 0.6485 |
| **Overall** | **42.0%** | **44.0%** | **1.0598** | **1.0522** | **0.6371** | **0.6340** |

The market performed better during some Gameweeks while Model 2 produced better probability quality during others, particularly GW4 and GW5.

The signals therefore appeared complementary rather than one being a clear replacement for the other.

### Model 2 / Opening Market Blend Experiment

A controlled blend sweep was then performed across the same 50 fixtures.

For each candidate weight:

`Blended probability = Model 2 weight x Model 2 probability + Market weight x Opening Market probability`

The sweep tested Model 2 weights from 100% to 0% in five-percentage-point increments, with the market receiving the complementary weight.

Selected results were:

| Model 2 Weight | Market Weight | Accuracy | Log Loss | Brier Score |
| ---: | ---: | ---: | ---: | ---: |
| 100% | 0% | 42.0% | 1.0598 | 0.6371 |
| 55% | 45% | 42.0% | 1.0487 | 0.6297 |
| 50% | 50% | 42.0% | 1.0483 | 0.6295 |
| 45% | 55% | 46.0% | 1.0480 | 0.6294 |
| 40% | 60% | 46.0% | 1.0478 | 0.6294 |
| 35% | 65% | 46.0% | 1.0478 | 0.6296 |
| 0% | 100% | 44.0% | 1.0522 | 0.6340 |

The result was a broad, shallow optimum rather than evidence for a uniquely optimal precise weight.

The strongest region was approximately:

- 35%-45% Model 2
- 55%-65% Opening Market

A **40% Model 2 / 60% Opening Market** specification was selected within that region.

### Frozen Shadow Challenger

The selected challenger is frozen as:

`Model2_OpeningMarket_40_60_v1.0`

Its specification is:

- Champion: `Model2_v1.0`
- Model 2 weight: 40%
- Opening Market weight: 60%
- Status: Shadow Challenger
- Development period: GW1-GW5
- Development matches: 50
- Prospective validation start: GW6
- Retuning using GW6 or later results: prohibited

Development-period performance was:

| Signal | Accuracy | Log Loss | Brier Score |
| --- | ---: | ---: | ---: |
| Model 2 | 42.0% | 1.0598 | 0.6371 |
| Opening Market | 44.0% | 1.0522 | 0.6340 |
| 40/60 Shadow Challenger | **46.0%** | **1.0478** | **0.6294** |

The 40/60 challenger matched or exceeded Model 2's 1X2 accuracy in each of the five development Gameweeks:

| Gameweek | Model 2 | Opening Market | 40/60 Blend |
| --- | ---: | ---: | ---: |
| GW1 | 50.0% | 60.0% | 60.0% |
| GW2 | 50.0% | 50.0% | 50.0% |
| GW3 | 30.0% | 30.0% | 30.0% |
| GW4 | 30.0% | 40.0% | 40.0% |
| GW5 | 50.0% | 40.0% | 50.0% |

These results must not be interpreted as prospective validation of the selected blend because GW1-GW5 were used to select its specification.

### Prospective Market Source

The prospective market source has now been verified as **The Odds API**.

The live market pipeline uses:

- Sport: `soccer_epl`
- Region: `uk`
- Market: `h2h`
- Odds format: decimal

For each fixture, only bookmakers supplying a complete Home / Draw / Away market are included.

Available bookmaker decimal prices are averaged separately for Home, Draw and Away. The resulting consensus prices are then converted to fair probabilities using inverse decimal odds and normalisation to remove the aggregate overround.

This mirrors the probability methodology used for the historical Football-Data opening-market benchmark, while recognising that the prospective bookmaker panel is a different data source and may not contain the same bookmakers.

### Prospective Snapshot Protocol

From GW6 onward, one official market snapshot will be frozen for each Gameweek approximately **24 hours before the first Premier League fixture of that Gameweek**.

At the official decision point the process is:

1. refresh the Gameweek fixture snapshot
2. generate and freeze the official `Model2_v1.0` predictions
3. capture and freeze the official market snapshot
4. generate and freeze the `Model2_OpeningMarket_40_60_v1.0` shadow predictions

The market snapshot will not subsequently be refreshed for that Gameweek's official prospective comparison.

This means Champion and Shadow Challenger are evaluated from the same pre-match information point.

### Production Isolation

The existing Model 2 production workflow remains unchanged.

The official prediction generator remains:

`scripts/generate_live_gameweek_predictions.py`

The market challenger operates through separate scripts:

- `scripts/fetch_opening_market_snapshot.py`
- `scripts/generate_market_shadow_predictions.py`
- `scripts/evaluate_market_shadow.py`

The shadow generator consumes the already-frozen official Model 2 probabilities and the separately frozen market snapshot. It does not retrain Model 2 or call the Model 2 prediction process itself.

The challenger outputs and evaluations are also stored separately from the official Model 2 prediction and evaluation files.

Safeguards prevent official snapshots from being silently overwritten and require the expected model and challenger versions.

### GW6 Infrastructure Validation

The prospective workflow was tested on **24 September 2026** before the official GW6 decision point.

The test:

- retrieved all 10 GW6 fixtures
- obtained complete Home / Draw / Away prices from 18 UK bookmakers for each fixture in the test snapshot
- successfully calculated consensus no-vig market probabilities
- confirmed that the shadow generator refuses to run without a frozen official Model 2 snapshot
- confirmed that the shadow evaluator refuses to run without a frozen official shadow prediction file

No official GW6 Model 2 prediction file was generated during this test.

The market file created during the test is explicitly marked as a test artefact and must not be used for prospective evaluation.

### GW6 Official Decision Point

The first GW6 fixture is:

**Arsenal vs Leeds, Saturday 10 October 2026 at 12:30 UK time**

The target official snapshot point is therefore:

**Friday 9 October 2026 at approximately 12:30 UK time**

At that point the GW6 fixtures will be refreshed and both the Champion and Shadow Challenger inputs will be frozen according to the prospective protocol.

GW6 and subsequent Gameweeks will provide the genuinely unseen evidence required to assess whether the 40/60 challenger generalises beyond the 50-match development period.

No GW6 or later result will be used to retune the frozen 40/60 weight.