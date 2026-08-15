import pandas as pd


df = pd.read_csv(
    "data/processed/premier_league_matches.csv"
)


home_teams = set(
    df["HomeTeam"].dropna().unique()
)

away_teams = set(
    df["AwayTeam"].dropna().unique()
)


all_teams = sorted(
    home_teams | away_teams
)


print("\nALL TEAMS")
print("=========")


for team in all_teams:
    print(team)


print(
    f"\nTotal unique teams: {len(all_teams)}"
)

only_home = home_teams - away_teams

only_away = away_teams - home_teams


print("\nTeams appearing only as home team:")
print(only_home)


only_home = home_teams - away_teams

only_away = away_teams - home_teams


print("\nTeams appearing only as home team:")
print(only_home)


print("\nTeams appearing only as away team:")
print(only_away)