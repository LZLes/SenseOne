#!/usr/bin/env bash
# One-command setup for SenseOne (Gemini fork): creates a venv, installs
# deps, and checks for a Gemini API key. Safe to re-run.
set -euo pipefail

if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "GEMINI_API_KEY is not set."
    echo "Get a free key at https://aistudio.google.com/apikey, then:"
    echo "  export GEMINI_API_KEY=..."
    echo "and re-run this script (or just set it before running agent.py/app.py)."
    exit 1
fi

echo "== Setting up Python environment =="
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt

cat <<'EOF'

Setup complete.

Activate the environment in new shells with:
  source .venv/bin/activate

Make sure GEMINI_API_KEY is exported in that shell too, then run either:
  python agent.py          # terminal chat
  streamlit run app.py     # local web GUI (localhost:8501)

Add your own CV/CA data to reference_data/ and electrode photos to
reference_images/ -- see the README in each for the expected layout.
EOF
