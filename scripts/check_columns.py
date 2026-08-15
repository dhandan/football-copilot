import pandas as pd


df = pd.read_csv(
    "data/processed/premier_league_matches.csv"
)


important_columns = [
    "Season",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HTHG",
    "HTAG",
    "HTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
]


print("\nIMPORTANT FOOTBALL COLUMNS")
print("==========================")


for column in important_columns:

    if column in df.columns:

        missing = df[column].isna().sum()

        print(
            f"{column:10} "
            f"present   "
            f"missing: {missing}"
        )

    else:

        print(
            f"{column:10} "
            f"MISSING"
        )