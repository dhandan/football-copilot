from pathlib import Path
import subprocess
import sys


# ==================================================
# PROJECT ROOT
# ==================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


# ==================================================
# HELPER
# ==================================================

def run_step(
    number,
    name,
    script_path,
):
    """
    Run one project-build script.

    Stop immediately if any stage fails.
    """

    print()
    print("=" * 70)
    print(f"STEP {number}: {name}")
    print("=" * 70)

    command = [
        sys.executable,
        script_path,
    ]

    print()
    print("Running:")
    print(" ".join(command))
    print()

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:

        print()
        print("=" * 70)
        print(f"BUILD FAILED AT STEP {number}")
        print("=" * 70)

        print(f"Stage: {name}")
        print(f"Script: {script_path}")

        sys.exit(
            result.returncode
        )

    print()
    print(f"STEP {number} COMPLETE")


# ==================================================
# START
# ==================================================

print()
print("=" * 70)
print("FOOTBALL COPILOT")
print("FULL PROJECT BUILD")
print("=" * 70)

print()
print(
    f"Project root: "
    f"{PROJECT_ROOT}"
)

print(
    f"Python: "
    f"{sys.executable}"
)


# ==================================================
# DATA PIPELINE
# ==================================================

run_step(
    1,
    "Download Premier League data",
    "scripts/download_data.py",
)


run_step(
    2,
    "Combine season data",
    "scripts/combine_data.py",
)


run_step(
    3,
    "Clean match data",
    "scripts/clean_data.py",
)


run_step(
    4,
    "Build DuckDB analytics database",
    "scripts/build_database.py",
)


# ==================================================
# PREDICTION FEATURES
# ==================================================

run_step(
    5,
    "Build Model 2 feature dataset",
    "prediction/build_features_v2.py",
)


run_step(
    6,
    "Build common validation feature dataset",
    "prediction/build_features_v3.py",
)


# ==================================================
# PROMOTED-TEAM PIPELINE
# ==================================================

run_step(
    7,
    "Build Championship team-season summaries",
    "prediction/build_promoted_team_priors.py",
)


run_step(
    8,
    "Calculate promotion translation factors",
    "prediction/calculate_promotion_translation.py",
)


run_step(
    9,
    "Generate 2026/27 promoted-team priors",
    "prediction/generate_2026_27_promoted_priors.py",
)


# ==================================================
# HISTORICAL MARKET DATA
# ==================================================

run_step(
    10,
    "Build historical bookmaker dataset",
    "prediction/build_market_data.py",
)


run_step(
    11,
    "Calculate fair market probabilities",
    "prediction/calculate_market_probabilities.py",
)


# ==================================================
# PRODUCTION MODEL
# ==================================================

run_step(
    12,
    "Train frozen production Model 2",
    "prediction/train_production_model.py",
)


# ==================================================
# FINAL VALIDATION
# ==================================================

run_step(
    13,
    "Run final model validation",
    "prediction/final_model_validation.py",
)


# ==================================================
# COMPLETE
# ==================================================

print()
print("=" * 70)
print("FOOTBALL COPILOT BUILD COMPLETE")
print("=" * 70)

print()
print("The following have now been rebuilt:")

print(
    """
- historical Premier League match data
- cleaned analytics data
- DuckDB analytics database
- Model 2 features
- validation features
- Championship team-season summaries
- promotion translation factors
- 2026/27 promoted-team priors
- historical bookmaker data
- fair market probabilities
- frozen production Model 2
- final validation outputs
"""
)

print(
    "Start Football Copilot with:"
)

print()
print(
    "streamlit run app/streamlit_app.py"
)
print()
