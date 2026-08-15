#!/bin/bash

# Move into the Football Copilot project folder
cd "$(dirname "$0")"

# Check that the virtual environment exists
if [ ! -d ".venv" ]; then
    echo ""
    echo "ERROR: .venv does not exist."
    echo ""
    echo "Create the environment first with:"
    echo ""
    echo "python3 -m venv .venv"
    echo ""
    exit 1
fi

# Activate Python environment
source .venv/bin/activate

echo ""
echo "Starting Football Copilot..."
echo ""

# Start Streamlit
streamlit run app/streamlit_app.py