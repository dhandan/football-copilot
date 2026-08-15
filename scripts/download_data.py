from pathlib import Path
from urllib.request import urlopen
import ssl
import certifi


DATA_FOLDER = Path("data/raw")

DATA_FOLDER.mkdir(parents=True, exist_ok=True)


SEASONS = {
    "2021/22": "2122",
    "2022/23": "2223",
    "2023/24": "2324",
    "2024/25": "2425",
    "2025/26": "2526",
}


ssl_context = ssl.create_default_context(
    cafile=certifi.where()
)


for season_name, season_code in SEASONS.items():

    url = (
        f"https://www.football-data.co.uk/"
        f"mmz4281/{season_code}/E0.csv"
    )

    output_file = DATA_FOLDER / f"E0_{season_code}.csv"

    print(f"Downloading Premier League {season_name}...")

    with urlopen(url, context=ssl_context) as response:
        data = response.read()

    with open(output_file, "wb") as file:
        file.write(data)

    print(f"Saved to: {output_file}")


print("\nAll downloads complete.")