# Model 3A: Dixon-Coles Shadow Challenger

## Why Model 3A was investigated

Football Copilot is currently running Model 2 as the frozen production baseline for the 2026/27 Premier League live experiment.

After the first two Gameweeks, an unusual pattern emerged in the prospective predictions.

Across the first 20 live fixtures:

- Model 2 achieved 50.0% 1X2 accuracy.
- 4 of the 20 matches ended in draws.
- Model 2 selected Draw as the highest-probability 1X2 outcome in 0 of 20 fixtures.
- Model 2 produced 1-1 as its modal exact score in 16 of 20 fixtures.
- Only 2 of the 20 fixtures actually finished 1-1.

This created an interesting apparent contradiction.

The model frequently identified 1-1 as the single most likely exact scoreline, while the combined probability of all draw scorelines was never sufficiently high for Draw to become the most likely overall 1X2 outcome.

Rather than modifying the live model immediately, this behaviour was treated as an empirical modelling question.

---

## Hypothesis

The initial hypothesis was:

> Model 2's independent Poisson formulation may inadequately represent dependence between home and away scoring in low-scoring football matches. A Dixon-Coles adjustment may improve the scoreline distribution and draw probability calibration without sacrificing overall probability quality.

Dixon-Coles was selected because it modifies the probability assigned to four low-scoring outcomes:

- 0-0
- 1-0
- 0-1
- 1-1

This made it directly relevant to the behaviour observed in the live experiment.

---

## Experimental design

Model 3A was deliberately implemented as a controlled challenger rather than a completely new prediction model.

The Model 2 feature engineering and expected-goal models were left unchanged.

The experiment therefore followed this architecture:

Model 2 features  
↓  
Model 2 home and away Poisson regressors  
↓  
Home xG + Away xG  
↓  
Independent Poisson score matrix = Model 2  
↓  
Dixon-Coles low-score correction = Model 3A  
↓  
H / D / A probabilities and modal scoreline

This design isolates the effect of the Dixon-Coles correction.

If Model 3A improved performance, the improvement could therefore be attributed much more confidently to the probability adjustment rather than to simultaneous changes in features, expected-goal modelling or training methodology.

---

## Walk-forward methodology

The challenger was evaluated using the same walk-forward principle used for Model 2.

For each test season, only previous seasons were used for training.

The historical test periods were:

| Test season | Training data |
|---|---|
| 2023/24 | 2021/22 + 2022/23 |
| 2024/25 | 2021/22 to 2023/24 |
| 2025/26 | 2021/22 to 2024/25 |

The Dixon-Coles rho parameter was estimated using training data only.

This prevents information from the test season leaking into the challenger.

The fitted rho values were:

| Test season | Rho |
|---|---:|
| 2023/24 | 0.03331 |
| 2024/25 | 0.01572 |
| 2025/26 | 0.01519 |

---

## Evaluation criteria

The challenger was not judged simply on whether it predicted more draws or produced fewer 1-1 scorelines.

The primary model-quality measures remained:

- 1X2 accuracy
- Log Loss
- Brier Score

Additional diagnostics included:

- actual draw rate
- predicted draw rate
- draw recall
- average predicted draw probability
- average draw probability when a draw actually occurred
- average draw probability when the match did not finish level
- modal 1-1 frequency
- actual 1-1 frequency
- goal MAE

The goal was therefore to improve probability quality, not artificially force the model to predict more draws.

---

## Historical results

The walk-forward experiment evaluated 1,100 previously unseen matches.

| Metric | Model 2 | Model 3A |
|---|---:|---:|
| 1X2 Accuracy | 52.18% | 52.18% |
| Log Loss | 1.0037 | 1.0041 |
| Brier Score | 0.5999 | 0.6001 |
| Predicted Draws | 0 | 0 |
| Predicted Draw Rate | 0.00% | 0.00% |
| Actual Draws | 268 | 268 |
| Actual Draw Rate | 24.36% | 24.36% |
| Draw Recall | 0.00% | 0.00% |
| Mean P(Draw) | 22.81% | 22.35% |
| Modal 1-1 Rate | 71.18% | 64.55% |

Actual 1-1 results occurred in 124 of the 1,100 matches, equivalent to 11.27%.

---

## What changed?

Dixon-Coles did have a measurable effect on the exact-score distribution.

Model 2 selected 1-1 as its modal scoreline in:

**71.18% of historical test matches**

Model 3A reduced this to:

**64.55%**

The actual historical 1-1 rate was:

**11.27%**

The adjustment therefore moved the modal-score behaviour in the expected direction, but the concentration remained extremely high.

More importantly, the change did not translate into improved 1X2 draw behaviour.

Both models predicted:

**0 draws from 1,100 matches**

despite:

**268 actual draws**

The Dixon-Coles adjustment therefore did not resolve the underlying draw-prediction issue.

---

## Probability quality

Model 3A also failed the primary promotion criteria.

Accuracy was unchanged:

**52.18% → 52.18%**

Log Loss became marginally worse:

**1.0037 → 1.0041**

Brier Score also became marginally worse:

**0.5999 → 0.6001**

The differences are small, but there is no evidence that the challenger improves probability quality.

---

## Conclusion

### Hypothesis partially rejected

The experiment provides evidence that low-score dependence contributes to the unusual modal-score distribution, because Dixon-Coles reduced the concentration of 1-1 modal predictions.

However, it does not explain the larger draw-calibration problem.

The absence of predicted draws persisted completely after the Dixon-Coles correction.

Model 3A also failed to improve either Log Loss or Brier Score.

It will therefore **not be promoted to prospective shadow deployment**.

Model 2 remains the frozen official model for the 2026/27 live experiment.

---

## What we learned

This experiment narrowed the investigation.

The original observation combined two behaviours:

1. excessive 1-1 modal scorelines
2. no Draw selections at the 1X2 level

The Dixon-Coles experiment suggests these should not be treated as the same problem.

Low-score correction changed the first behaviour but did not solve the second.

The next investigation will therefore move upstream and examine:

- home versus away xG separation
- total expected goals
- predicted draw probability
- actual draw frequency
- probability calibration
- the conditions under which Draw can become the highest-probability 1X2 outcome

This analysis will determine the design of the next challenger rather than selecting another model architecture in advance.

---

## Model status

**Model 2:** Frozen official live model  
**Model 3A:** Historical challenger tested  
**Model 3A outcome:** Rejected for shadow promotion  
**Next step:** Draw/xG diagnostics and calibration investigation