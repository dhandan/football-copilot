from pathlib import Path
import requests


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "championship"
)


SEASONS = {
    "2021_22": "2122",
    "2022_23": "2223",
    "2023_24": "2324",
    "2024_25": "2425",
    "2025_26": "2526",
}


BASE_URL = (
    "https://www.football-data.co.uk/"
    "mmz4281/{season}/E1.csv"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("FOOTBALL COPILOT")
print("CHAMPIONSHIP DATA DOWNLOAD")
print("==========================")
print()


for season_name, season_code in SEASONS.items():

    url = BASE_URL.format(
        season=season_code
    )

    output_file = (
        OUTPUT_DIR
        /
        f"championship_{season_name}.csv"
    )

    print(
        f"Downloading {season_name}..."
    )

    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent":
                "FootballCopilot/1.0"
        },
    )

    response.raise_for_status()

    output_file.write_bytes(
        response.content
    )

    print(
        f"Saved: {output_file}"
    )


print()
print("DOWNLOAD COMPLETE")