from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT)
)


from analytics.football_analytics import (
    get_team_record,
    get_team_form,
    get_league_table
)


print("\nERROR HANDLING TESTS")
print("====================")


try:

    get_team_record(
        "Liverpol",
        "2025/26"
    )

except ValueError as error:

    print("\nBad team:")
    print(error)


try:

    get_team_form(
        "Liverpool",
        -5
    )

except ValueError as error:

    print("\nBad game count:")
    print(error)


try:

    get_league_table(
        "2035/36"
    )

except ValueError as error:

    print("\nBad season:")
    print(error)