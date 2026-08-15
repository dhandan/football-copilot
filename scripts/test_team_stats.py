import pandas as pd


df = pd.read_csv(
    "data/processed/team_match_stats.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"]
)


team = "Liverpool"


team_data = df[
    df["Team"] == team
].sort_values("Date")


print(f"\n{team}")
print("=" * len(team))


print("\nLast 10 matches:")

print(
    team_data[
        [
            "Date",
            "Opponent",
            "Venue",
            "GoalsFor",
            "GoalsAgainst",
            "Result",
            "Points"
        ]
    ].tail(10)
)


last_10 = team_data.tail(10)


print("\nLast 10 summary:")

print(
    last_10[
        [
            "Win",
            "Draw",
            "Loss",
            "GoalsFor",
            "GoalsAgainst",
            "Points"
        ]
    ].sum()
)

home_data = team_data[
    team_data["Venue"] == "Home"
]

away_data = team_data[
    team_data["Venue"] == "Away"
]


print("\nHome record:")

print(
    home_data[
        [
            "Win",
            "Draw",
            "Loss",
            "GoalsFor",
            "GoalsAgainst",
            "Points"
        ]
    ].sum()
)


print("\nAway record:")

print(
    away_data[
        [
            "Win",
            "Draw",
            "Loss",
            "GoalsFor",
            "GoalsAgainst",
            "Points"
        ]
    ].sum()
)

opponent = "Arsenal"


head_to_head = team_data[
    team_data["Opponent"] == opponent
]


print(
    f"\n{team} vs {opponent}:"
)

print(
    head_to_head[
        [
            "Season",
            "Date",
            "Venue",
            "GoalsFor",
            "GoalsAgainst",
            "Result",
            "Points"
        ]
    ]
)


print("\nHead-to-head summary:")

print(
    head_to_head[
        [
            "Win",
            "Draw",
            "Loss",
            "GoalsFor",
            "GoalsAgainst",
            "Points"
        ]
    ].sum()
)