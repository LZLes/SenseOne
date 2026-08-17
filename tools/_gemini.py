"""Shared Gemini client + model config for every tool-calling and vision
call in this fork. One place to swap the model or key-handling instead of
duplicating it across agent.py and every tool file that talks to Gemini
directly (image_qc.py, literature_figures.py, vault_maintenance.py).
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# One model for everything (text tool-calling, thinking, and vision) --
# unlike the local Ollama version, which needed a separate vision model.
# Check https://ai.google.dev/gemini-api/docs/models for the current
# recommended free-tier flash model if this one has been deprecated by the
# time you're reading this -- model names/tiers here move fast.
MODEL = "gemini-3.6-flash"

_client = None

# Bounds every Gemini call so a stalled request can't hang a session
# forever, and auto-retries transient/rate-limit errors instead of failing
# a whole turn on the first 429/5xx -- likely with several people sharing
# one GEMINI_API_KEY during a demo.
_HTTP_OPTIONS = types.HttpOptions(
    timeout=60_000,  # ms
    retry_options=types.HttpRetryOptions(
        attempts=3, initial_delay=1.0, max_delay=10.0,
        http_status_codes=[429, 500, 502, 503, 504],
    ),
)


def api_key_configured() -> bool:
    return _resolve_api_key() is not None


def _resolve_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    # Streamlit Community Cloud: a key set via the Cloud UI's secrets editor
    # lands in st.secrets, not necessarily os.environ (only top-level TOML
    # keys get mirrored there) -- fall back to it so a hosted deploy doesn't
    # silently fail with no explanation.
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


def client() -> genai.Client:
    global _client
    if _client is None:
        api_key = _resolve_api_key()
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey, then "
                "`export GEMINI_API_KEY=...` (or set it in Streamlit's secrets "
                "manager when hosted) before running."
            )
        _client = genai.Client(api_key=api_key, http_options=_HTTP_OPTIONS)
    return _client
