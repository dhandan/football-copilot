from pathlib import Path
import sys
import pprint


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.append(
    str(PROJECT_ROOT)
)


from prediction.fixture_predictor import (
    predict_fixture
)


prediction = predict_fixture(
    "Liverpool",
    "Arsenal",
)


print()
print("FIXTURE PREDICTION")
print("==================")

pprint.pprint(
    prediction
)