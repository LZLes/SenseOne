#!/usr/bin/env bash
# One-command setup for SenseOne: pulls the two Ollama models, creates a
# venv, and installs Python deps. Safe to re-run.
set -euo pipefail

if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama not found on PATH. Install it first: https://ollama.com/download"
    exit 1
fi

echo "== Pulling models (qwen3:8b ~5.2GB, qwen2.5vl:7b ~6.0GB) -- this can take a while on first run =="
ollama pull qwen3:8b
ollama pull qwen2.5vl:7b

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

Then run either:
  python agent.py          # terminal chat
  streamlit run app.py     # local web GUI (localhost:8501)

Add your own CV/CA data to reference_data/ and electrode photos to
reference_images/ -- see the README in each for the expected layout.
EOF
