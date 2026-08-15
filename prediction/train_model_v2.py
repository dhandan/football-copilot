from pathlib import Path
import pickle

import pandas as pd
from sklearn.linear_model import PoissonRegressor


# --------------------------------------------------
# Files
# --------------------------------------------------

INPUT_FILE = (
    "data/processed/"
    "prediction_features_v2.csv"
)

MODEL_FILE = (
    "models/"
    "poisson_model_v2.pkl"
)


# --------------------------------------------------
# Features
# --------------------------------------------------

MODEL_2_FEATURES = [

    "HomeRecentGoalsFor",
    "HomeRecentGoalsAgainst",
    "HomeRecentPPG",

    "AwayRecentGoalsFor",
    "AwayRecentGoalsAgainst",
    "AwayRecentPPG",

    "Home10GoalsFor",
    "Home10GoalsAgainst",
    "Home10PPG",

    "Away10GoalsFor",
    "Away10GoalsAgainst",
    "Away10PPG",

    "HomeVenuePPG",
    "HomeVenueGoalsFor",

    "AwayVenuePPG",
    "AwayVenueGoalsFor",

    "HomeSeasonPPG",
    "HomeSeasonGoalDifferencePG",

    "AwaySeasonPPG",
    "AwaySeasonGoalDifferencePG",

    "RecentPPGDifference",
    "TenMatchPPGDifference",
    "SeasonPPGDifference",

    "AttackVsDefenceHome",
    "AttackVsDefenceAway",
]


# --------------------------------------------------
# Load data
# --------------------------------------------------

print()
print("TRAINING MODEL 2")
print("================")

df = pd.read_csv(
    INPUT_FILE
)

X = df[
    MODEL_2_FEATURES
]

y_home = df[
    "HomeGoals"
]

y_away = df[
    "AwayGoals"
]


# --------------------------------------------------
# Train models
# --------------------------------------------------

home_model = PoissonRegressor(
    alpha=0.1,
    max_iter=1000,
)

away_model = PoissonRegressor(
    alpha=0.1,
    max_iter=1000,
)


home_model.fit(
    X,
    y_home,
)

away_model.fit(
    X,
    y_away,
)


# --------------------------------------------------
# Save model bundle
# --------------------------------------------------

Path("models").mkdir(
    parents=True,
    exist_ok=True,
)


model_bundle = {
    "home_model": home_model,
    "away_model": away_model,
    "features": MODEL_2_FEATURES,

    "metadata": {
        "model_name": "Model 2",
        "model_type": "Poisson regression",
        "walk_forward_accuracy": 52.18,
        "walk_forward_log_loss": 1.0037,
        "walk_forward_brier": 0.5999,
        "walk_forward_matches": 1100,
    },
}


with open(
    MODEL_FILE,
    "wb",
) as file:

    pickle.dump(
        model_bundle,
        file,
    )


print(
    f"Saved model to: {MODEL_FILE}"
)

print(
    f"Training rows: {len(df)}"
)

print()
print("MODEL 2 TRAINING COMPLETE")
print("=========================")