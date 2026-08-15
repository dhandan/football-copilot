from pathlib import Path
import pickle

import pandas as pd
from sklearn.linear_model import PoissonRegressor


INPUT_FILE = (
    "data/processed/prediction_features.csv"
)

MODEL_FILE = (
    "models/poisson_models.pkl"
)


print("\nTraining Poisson models...")
print("===========================")


# --------------------------------------------------
# Load features
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)


FEATURE_COLUMNS = [
    "HomeRecentGoalsFor",
    "HomeRecentGoalsAgainst",
    "HomeRecentPPG",
    "AwayRecentGoalsFor",
    "AwayRecentGoalsAgainst",
    "AwayRecentPPG",
]


X = df[
    FEATURE_COLUMNS
]


# --------------------------------------------------
# Target variables
# --------------------------------------------------

y_home = df[
    "HomeGoals"
]

y_away = df[
    "AwayGoals"
]


# --------------------------------------------------
# Create models
# --------------------------------------------------

home_model = PoissonRegressor(
    alpha=0.1,
    max_iter=1000,
)

away_model = PoissonRegressor(
    alpha=0.1,
    max_iter=1000,
)


# --------------------------------------------------
# Train
# --------------------------------------------------

home_model.fit(
    X,
    y_home,
)

away_model.fit(
    X,
    y_away,
)


# --------------------------------------------------
# Save models
# --------------------------------------------------

Path("models").mkdir(
    parents=True,
    exist_ok=True,
)


model_bundle = {
    "home_model": home_model,
    "away_model": away_model,
    "features": FEATURE_COLUMNS,
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
    f"\nModels saved to: {MODEL_FILE}"
)


# --------------------------------------------------
# Basic diagnostics
# --------------------------------------------------

home_prediction = home_model.predict(
    X
)

away_prediction = away_model.predict(
    X
)


print(
    "\nAverage actual home goals:",
    round(
        y_home.mean(),
        3,
    ),
)

print(
    "Average predicted home goals:",
    round(
        home_prediction.mean(),
        3,
    ),
)

print(
    "\nAverage actual away goals:",
    round(
        y_away.mean(),
        3,
    ),
)

print(
    "Average predicted away goals:",
    round(
        away_prediction.mean(),
        3,
    ),
)