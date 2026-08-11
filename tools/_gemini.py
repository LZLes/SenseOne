"""Shared Gemini client + model config for every tool-calling and vision
call in this fork. One place to swap the model or key-handling instead of
duplicating it across agent.py and every tool file that talks to Gemini
directly (image_qc.py, literature_figures.py, vault_maintenance.py).
"""

import os

from google import genai

# One model for everything (text tool-calling, thinking, and vision) --
# unlike the local Ollama version, which needed a separate vision model.
# Check https://ai.google.dev/gemini-api/docs/models for the current
# recommended free-tier flash model if this one has been deprecated by the
# time you're reading this -- model names/tiers here move fast.
MODEL = "gemini-2.5-flash"

_client = None


def client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey, then "
                "`export GEMINI_API_KEY=...` before running."
            )
        _client = genai.Client(api_key=api_key)
    return _client
