import pandas as pd


file_path = "data/raw/E0_2526.csv"

df = pd.read_csv(file_path)


team = "Liverpool"


home_games = df[df["HomeTeam"] == team]

away_games = df[df["AwayTeam"] == team]


wins = 0
draws = 0
losses = 0


# Home results
for _, match in home_games.iterrows():

    if match["FTR"] == "H":
        wins += 1
    elif match["FTR"] == "D":
        draws += 1
    else:
        losses += 1


# Away results
for _, match in away_games.iterrows():

    if match["FTR"] == "A":
        wins += 1
    elif match["FTR"] == "D":
        draws += 1
    else:
        losses += 1


print(f"\n{team} 2025/26")
print("================")

print(f"Wins:   {wins}")
print(f"Draws:  {draws}")
print(f"Losses: {losses}")
print(f"Games:  {wins + draws + losses}")


# Points
Win    = 3
Draw   = 1
Loss   = 0

points = wins * 3 + draws
print(f"Points: {points}")

# Goals
goals_for = (
    home_games["FTHG"].sum()
    + away_games["FTAG"].sum()
)

goals_against = (
    home_games["FTAG"].sum()
    + away_games["FTHG"].sum()
)

print(f"Goals scored:    {goals_for}")
print(f"Goals conceded:  {goals_against}")
print(f"Goal difference: {goals_for - goals_against}")

