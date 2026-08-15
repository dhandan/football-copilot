from pathlib import Path
import pandas as pd


RAW_FOLDER = Path("data/raw")
PROCESSED_FOLDER = Path("data/processed")


PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)


SEASONS = {
    "E0_2122.csv": "2021/22",
    "E0_2223.csv": "2022/23",
    "E0_2324.csv": "2023/24",
    "E0_2425.csv": "2024/25",
    "E0_2526.csv": "2025/26",
}


all_seasons = []


for filename, season in SEASONS.items():

    file_path = RAW_FOLDER / filename

    print(f"Reading {season}...")

    df = pd.read_csv(file_path)

    df["Season"] = season

    all_seasons.append(df)


combined_df = pd.concat(
    all_seasons,
    ignore_index=True
)


output_file = PROCESSED_FOLDER / "premier_league_matches.csv"


combined_df.to_csv(
    output_file,
    index=False
)


print("\nCombined dataset created.")

print(f"Saved to: {output_file}")

print(f"Total rows: {len(combined_df)}")

print(f"Total columns: {len(combined_df.columns)}")