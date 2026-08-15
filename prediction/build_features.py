from pathlib import Path
import pandas as pd


INPUT_FILE = "data/processed/matches_clean.csv"
OUTPUT_FILE = "data/processed/prediction_features.csv"


print("\nBuilding prediction features...")
print("===============================")


# --------------------------------------------------
# Load matches
# --------------------------------------------------

matches = pd.read_csv(INPUT_FILE)

matches["Date"] = pd.to_datetime(
    matches["Date"],
    errors="coerce",
)

matches = matches.sort_values(
    "Date"
).reset_index(drop=True)


# --------------------------------------------------
# Helper: previous team matches
# --------------------------------------------------

def get_previous_matches(
    data,
    team,
    current_date,
    n=5,
):
    """
    Return the team's previous n matches
    before current_date.
    """

    previous = data[
        (
            (data["HomeTeam"] == team)
            |
            (data["AwayTeam"] == team)
        )
        &
        (data["Date"] < current_date)
    ].copy()

    previous = previous.sort_values(
        "Date"
    ).tail(n)

    return previous


# --------------------------------------------------
# Helper: calculate recent team stats
# --------------------------------------------------

def calculate_recent_stats(
    previous_matches,
    team,
):
    """
    Calculate recent goals and points
    from the team's perspective.
    """

    if len(previous_matches) == 0:

        return {
            "games": 0,
            "goals_for": 0.0,
            "goals_against": 0.0,
            "points": 0.0,
        }

    goals_for = 0
    goals_against = 0
    points = 0

    for _, match in previous_matches.iterrows():

        if match["HomeTeam"] == team:

            gf = match["FTHG"]
            ga = match["FTAG"]

        else:

            gf = match["FTAG"]
            ga = match["FTHG"]

        goals_for += gf
        goals_against += ga

        if gf > ga:
            points += 3

        elif gf == ga:
            points += 1

    games = len(previous_matches)

    return {
        "games": games,
        "goals_for": goals_for / games,
        "goals_against": goals_against / games,
        "points": points / games,
    }


# --------------------------------------------------
# Build one feature row per match
# --------------------------------------------------

feature_rows = []


for index, match in matches.iterrows():

    current_date = match["Date"]

    home_team = match["HomeTeam"]
    away_team = match["AwayTeam"]

    home_previous = get_previous_matches(
        matches,
        home_team,
        current_date,
        n=5,
    )

    away_previous = get_previous_matches(
        matches,
        away_team,
        current_date,
        n=5,
    )

    home_stats = calculate_recent_stats(
        home_previous,
        home_team,
    )

    away_stats = calculate_recent_stats(
        away_previous,
        away_team,
    )

    feature_rows.append(
        {
            "Season": match["Season"],
            "Date": current_date,

            "HomeTeam": home_team,
            "AwayTeam": away_team,

            "HomeGoals": match["FTHG"],
            "AwayGoals": match["FTAG"],

            "HomeRecentGoalsFor": (
                home_stats["goals_for"]
            ),

            "HomeRecentGoalsAgainst": (
                home_stats["goals_against"]
            ),

            "HomeRecentPPG": (
                home_stats["points"]
            ),

            "AwayRecentGoalsFor": (
                away_stats["goals_for"]
            ),

            "AwayRecentGoalsAgainst": (
                away_stats["goals_against"]
            ),

            "AwayRecentPPG": (
                away_stats["points"]
            ),

            "HomeHistoryGames": (
                home_stats["games"]
            ),

            "AwayHistoryGames": (
                away_stats["games"]
            ),
        }
    )


features = pd.DataFrame(
    feature_rows
)


# --------------------------------------------------
# Remove matches without enough history
# --------------------------------------------------

features = features[
    (features["HomeHistoryGames"] >= 5)
    &
    (features["AwayHistoryGames"] >= 5)
].copy()


# --------------------------------------------------
# Save
# --------------------------------------------------

Path("data/processed").mkdir(
    parents=True,
    exist_ok=True,
)


features.to_csv(
    OUTPUT_FILE,
    index=False,
)


print(
    f"\nRows created: {len(features)}"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)

print("\nFirst rows:")
print(features.head())