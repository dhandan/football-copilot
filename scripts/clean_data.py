import pandas as pd
from pathlib import Path


INPUT_FILE = (
    "data/processed/"
    "premier_league_matches.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "matches_clean.csv"
)


df = pd.read_csv(INPUT_FILE)


columns_to_keep = [
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


available_columns = [
    column
    for column in columns_to_keep
    if column in df.columns
]


clean_df = df[
    available_columns
].copy()


clean_df["Date"] = pd.to_datetime(
    clean_df["Date"],
    dayfirst=True,
    errors="coerce"
)


clean_df = clean_df.sort_values(
    ["Date", "HomeTeam"]
)


clean_df = clean_df.reset_index(
    drop=True
)


Path(
    "data/processed"
).mkdir(
    parents=True,
    exist_ok=True
)


clean_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nClean dataset created.")

print(f"Rows: {len(clean_df)}")

print(f"Columns: {len(clean_df.columns)}")

print(f"Saved to: {OUTPUT_FILE}")