#!/usr/bin/env bash
# User-triggered GitHub Codespaces session only; no persistent cloud deployment.
set -euo pipefail

if ! python -c 'import streamlit, duckdb, langgraph' >/dev/null 2>&1; then
  echo "Preview dependencies are not installed yet. Check Codespace setup logs."
  exit 1
fi

# Explicit restart is needed after a git pull: browser refresh and Streamlit
# hot reload can leave the old agent/langgraph_v2 Python module in memory.
if [[ "\${1:-}" == "--restart" ]]; then
  echo "Stopping the existing V2 preview to load updated Python modules..."
  pkill -f 'python -m streamlit run app/v2_groq_preview.py' || true
  for attempt in {1..20}; do
    if ! (echo > /dev/tcp/127.0.0.1/8501) >/dev/null 2>&1; then
      break
    fi
    sleep 0.25
  done
fi

if (echo > /dev/tcp/127.0.0.1/8501) >/dev/null 2>&1; then
  echo "Existing preview detected on port 8501. Pass --restart to reload updated code."
else
  echo "Starting the PRIVATE experimental V2 preview on port 8501."
  nohup python -m streamlit run app/v2_groq_preview.py \
    --server.address 0.0.0.0 --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false \
    > /tmp/football-copilot-v2-preview.log 2>&1 < /dev/null &
fi

echo "Git branch: $(git branch --show-current)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "In the app sidebar look for: Build: H07 topic-switch fix"
echo "In GitHub Codespaces → PORTS, keep port 8501 Visibility PRIVATE."
echo "Stop the Codespace when finished to conserve free allowance."
