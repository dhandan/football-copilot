# Model 7 A/B/C historical empirical screen

**Status:** NO-GO full model build: no qualified +2pp historical result

Evaluated on the identical 1,100 saved historical fixtures, with earlier-season-only training.
Results are **exploratory**: earlier historical tests have repeatedly examined this cohort.
Hyperparameters and event features were fixed in the script before testing; no outcome-directed retuning was performed.
The raw external event source provides post-match records; all team feature averages shift by one full match before rolling.

**Fixed comparator:** 40% original Model 2 plus 60% saved market; 594/1100, 54.0%

| Arm | Forecast | Correct | Accuracy | Uplift | Log Loss | Brier | Paired uplift 95% bootstrap interval |
|---|---|---:|---:|---:|---:|---:|---|
| A | standalone | 567/1100 | 51.545% | -2.455pp | 1.00958 | 0.60448 | [-4.182, -0.818] |
| A | fixed_blend | 584/1100 | 53.091% | -0.909pp | 0.97632 | 0.58129 | [-1.727, -0.091] |
| B | standalone | 577/1100 | 52.455% | -1.545pp | 0.9975 | 0.59546 | [-3.091, -0.091] |
| B | fixed_blend | 593/1100 | 53.909% | -0.091pp | 0.97512 | 0.58037 | [-1.0, 0.727] |
| C | standalone | 594/1100 | 54.0% | +0.000pp | 1.00567 | 0.59994 | [-1.636, 1.636] |
| C | fixed_blend | 602/1100 | 54.727% | +0.727pp | 0.97616 | 0.5807 | [-0.273, 1.727] |

## Interpretation

The screening threshold is 616 correct of 1,100 while retaining Log Loss and Brier quality. A pass is only a reason to set up new prospective validation, not proof of future accuracy.
This is not an untouched holdout and does not establish the expected uplift with confidence.
The shared 2026/27 100-match archive has been feasibility-audited, but 2026/27 outcomes were not used for training or scoring this historical experiment.
Data source rights, publication lag, and stable live supply must be resolved before production deployment.

## Audit files
- reports/model7_screen/results.json
- reports/model7_screen/predictions.csv
- experiments/model7_abc/bounded_screen.py
