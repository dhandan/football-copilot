#!/usr/bin/env bash
# User-triggered GitHub Codespaces session only; no persistent cloud deployment.
set -euo pipefail
if ! python -c 'import streamlit, duckdb, langgraph' >/dev/null 2>&1; then
  echo "Preview dependencies are not installed yet. Check Codespace setup logs."
  exit 1
fi
if (echo > /dev/tcp/127.0.0.1/8501) >/dev/null 2>&1; then
  echo "Existing preview detected on port 8501."
else
  echo "Starting the PRIVATE experimental V2 preview on port 8501."
  nohup python -m streamlit run app/v2_groq_preview.py     --server.address 0.0.0.0 --server.port 8501     --server.headless true     --browser.gatherUsageStats false     > /tmp/football-copilot-v2-preview.log 2>&1 < /dev/null &
fi
echo "In GitHub Codespaces → PORTS, set port 8501 Visibility to PRIVATE."
echo "Set GROQ_API_KEY in repository Codespaces secrets (not Actions secrets)."
echo "Stop the Codespace when finished to conserve free allowance."
