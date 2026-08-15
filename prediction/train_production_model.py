from pathlib import Path
import pickle

import pandas as pd
from sklearn.linear_model import PoissonRegressor


# --------------------------------------------------
# Files
# --------------------------------------------------

INPUT_FILE = (
    "data/processed/"
    "prediction_features_v3.csv"
)

MODEL_FILE = (
    "models/"
    "production_model_v2.pkl"
)


# --------------------------------------------------
# Frozen Model 2 specification
# --------------------------------------------------

MODEL_NAME = "Model 2"

MODEL_TYPE = "Poisson regression"

MODEL_ALPHA = 0.1


MODEL_FEATURES = [

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
# Final validation metadata
# --------------------------------------------------

VALIDATION_METADATA = {

    "validation_matches": 1061,

    "model_accuracy_pct": 52.69,
    "model_log_loss": 0.9927,
    "model_brier": 0.5927,

    "market_accuracy_pct": 54.48,
    "market_log_loss": 0.9641,
    "market_brier": 0.5730,

    "five_percent_edge_2023_24_roi_pct": -9.48,
    "five_percent_edge_2024_25_roi_pct": -0.02,
    "five_percent_edge_2025_26_roi_pct": -8.40,
}


# --------------------------------------------------
# Load feature data
# --------------------------------------------------

print()
print("TRAINING PRODUCTION MODEL 2")
print("===========================")


df = pd.read_csv(
    INPUT_FILE
)


X = df[
    MODEL_FEATURES
]


y_home = df[
    "HomeGoals"
]


y_away = df[
    "AwayGoals"
]


# --------------------------------------------------
# Train final home-goals model
# --------------------------------------------------

home_model = PoissonRegressor(
    alpha=MODEL_ALPHA,
    max_iter=1000,
)


home_model.fit(
    X,
    y_home,
)


# --------------------------------------------------
# Train final away-goals model
# --------------------------------------------------

away_model = PoissonRegressor(
    alpha=MODEL_ALPHA,
    max_iter=1000,
)


away_model.fit(
    X,
    y_away,
)


# --------------------------------------------------
# Save final production bundle
# --------------------------------------------------

Path(
    "models"
).mkdir(
    parents=True,
    exist_ok=True,
)


model_bundle = {

    "model_name":
        MODEL_NAME,

    "model_type":
        MODEL_TYPE,

    "model_alpha":
        MODEL_ALPHA,

    "features":
        MODEL_FEATURES,

    "home_model":
        home_model,

    "away_model":
        away_model,

    "validation":
        VALIDATION_METADATA,
}


with open(
    MODEL_FILE,
    "wb",
) as file:

    pickle.dump(
        model_bundle,
        file,
    )


print()
print(
    f"Training rows: {len(df)}"
)

print(
    f"Features: {len(MODEL_FEATURES)}"
)

print(
    f"Saved to: {MODEL_FILE}"
)

print()
print("PRODUCTION MODEL TRAINING COMPLETE")
print("==================================")