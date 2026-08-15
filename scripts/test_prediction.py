from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.append(
    str(PROJECT_ROOT)
)


from prediction.poisson_predictor import (
    predict_match
)


# --------------------------------------------------
# Load our historical feature dataset
# --------------------------------------------------

df = pd.read_csv(
    "data/processed/prediction_features.csv"
)


# --------------------------------------------------
# Pick one historical match
# --------------------------------------------------

row = df.iloc[-1]


feature_values = {
    "HomeRecentGoalsFor":
        row["HomeRecentGoalsFor"],

    "HomeRecentGoalsAgainst":
        row["HomeRecentGoalsAgainst"],

    "HomeRecentPPG":
        row["HomeRecentPPG"],

    "AwayRecentGoalsFor":
        row["AwayRecentGoalsFor"],

    "AwayRecentGoalsAgainst":
        row["AwayRecentGoalsAgainst"],

    "AwayRecentPPG":
        row["AwayRecentPPG"],
}


prediction = predict_match(
    feature_values
)


print("\nMATCH")
print("=====")

print(
    row["HomeTeam"],
    "vs",
    row["AwayTeam"],
)


print("\nACTUAL RESULT")
print("=============")

print(
    f"{row['HomeGoals']} - "
    f"{row['AwayGoals']}"
)


print("\nMODEL PREDICTION")
print("================")

print(prediction)