import pandas as pd


df = pd.read_csv(
    "data/processed/matches_clean.csv"
)


print("\nCLEAN DATASET")
print("=============")


print("\nRows:")
print(len(df))


print("\nColumns:")
print(df.columns.tolist())


print("\nSeason counts:")
print(
    df["Season"]
    .value_counts()
    .sort_index()
)


print("\nFirst five matches:")
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


print("\nLast five matches:")
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
    ].tail()
)