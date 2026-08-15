from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT)
)


from agent.football_tools import (
    form_summary
)


result = form_summary(
    "Liverpool",
    10
)


print("\nTOOL TEST")
print("=========")

print(result)