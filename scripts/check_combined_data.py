import pandas as pd


file_path = "data/processed/premier_league_matches.csv"


df = pd.read_csv(file_path)


print("\nDATA QUALITY CHECK")
print("==================")


print("\nTotal matches:")
print(len(df))


print("\nMatches by season:")
print(
    df["Season"]
    .value_counts()
    .sort_index()
)


print("\nNumber of unique home teams:")
print(df["HomeTeam"].nunique())


print("\nNumber of unique away teams:")
print(df["AwayTeam"].nunique())


print("\nMissing home teams:")
print(df["HomeTeam"].isna().sum())


print("\nMissing away teams:")
print(df["AwayTeam"].isna().sum())


print("\nMissing full-time results:")
print(df["FTR"].isna().sum())


print("\nFirst five rows:")
print(
    df[
        [
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "FTR",
        ]
    ].head()
)