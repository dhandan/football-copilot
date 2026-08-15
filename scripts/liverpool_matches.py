import pandas as pd


df = pd.read_csv(
    "data/processed/matches_clean.csv"
)


team = "Liverpool"


matches = df[
    (df["HomeTeam"] == team)
    |
    (df["AwayTeam"] == team)
].copy()


matches["Date"] = pd.to_datetime(
    matches["Date"]
)


matches = matches.sort_values(
    "Date"
)


print(
    matches[
        [
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
        ]
    ].tail(20)
)