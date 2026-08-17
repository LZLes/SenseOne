"""Shared filesystem-path-component sanitization.

electrode_notes.py and literature.py both build file paths out of strings
the LLM supplies directly as tool arguments (batch, electrode_code,
paper_id) -- without sanitizing those first, a value containing '../' or an
absolute path could escape the intended directory. One implementation here
instead of two near-identical copies, so a future tightening of the rule
only has to happen in one place.
"""

import re

_UNSAFE_PATH_CHARS_RE = re.compile(r"[^A-Za-z0-9_.\-]+")


def safe_path_component(s: str, fallback: str = "unknown") -> str:
    """Collapse anything that isn't alnum/./-/_ into '_', and strip leading/
    trailing '_'/'.' (so a value can't reduce to '..' or hide a leading dot-
    file). Never returns '' -- falls back to `fallback` instead.
    """
    cleaned = _UNSAFE_PATH_CHARS_RE.sub("_", str(s).strip()).strip("_.")
    return cleaned or fallback
