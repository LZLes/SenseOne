"""
SenseOne (Gemini fork) -- web GUI (Streamlit).

Same presentation layer as the local Ollama version, adapted to the
Gemini API's Content/Part message shape (see agent.py's docstring for
why this fork exists). Reuses agent.py's SYSTEM_PROMPT (via
GENERATE_CONFIG), TOOLS, and run_tool_call directly rather than
reimplementing the tool-calling loop.

Run:
    export GEMINI_API_KEY=...   # https://aistudio.google.com/apikey
    streamlit run app.py
"""

import io
import json
import os
import random
import time
import uuid
from pathlib import Path

import streamlit as st
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image

import agent
import dashboard_ui

UPLOAD_DIR = Path("uploads")
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # keep in sync with .streamlit/config.toml's server.maxUploadSize

# Shown while waiting for the first token of a hop -- covers the real
# network/model latency before anything else is on screen, and disappears
# the instant real content (thought or answer) starts streaming in.
_LOADING_MESSAGES = (
    "Scanning the electrodes...",
    "Cross-referencing the literature vault...",
    "Warming up the potentiostat...",
    "Running the numbers...",
    "Consulting prior QC history...",
    "Checking for peaks...",
    "Polling the reference batch...",
    "Reticulating the roughness plots...",
    "Herding electrons...",
    "Calibrating...",
    "Percolating...",
)

st.set_page_config(page_title="SenseOne", page_icon="\U0001f52c", layout="wide")


def _find_image_paths(obj, found=None, _depth=0, _max_depth=25):
    """Recursively pull out every *_path-keyed string in a tool result that
    points at an existing image file, so it can be shown inline -- tool
    results nest these at different depths (e.g. image_qc's own image_path
    vs its nested surface_analysis.plot_path). _max_depth is a defensive cap
    -- tool results are our own JSON, not attacker-controlled, but nothing
    here should be able to blow the recursion stack regardless.
    """
    if found is None:
        found = []
    if _depth >= _max_depth:
        return found
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and key.endswith("path") and value.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                if os.path.exists(value):
                    found.append((key, value))
            else:
                _find_image_paths(value, found, _depth + 1, _max_depth)
    elif isinstance(obj, list):
        for item in obj:
            _find_image_paths(item, found, _depth + 1, _max_depth)
    return found


def _scalarize(value):
    """Flattens a value too complex for a single table cell into a short
    readable string -- a plain list becomes a comma-joined line, anything
    else falls back to compact JSON, so the properties table never has to
    drop a field just because it wasn't a flat scalar.
    """
    if isinstance(value, list):
        if all(isinstance(v, (str, int, float, bool, type(None))) for v in value):
            return ", ".join(str(v) for v in value) if value else "—"
        return json.dumps(value, default=str)
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    return value


def _render_as_table(result: dict) -> None:
    """Fallback for tool results with no bespoke dashboard_ui renderer --
    a table reads far faster than a nested JSON blob for the common shapes
    these tools return (a list of records, or a dict-of-records keyed by
    id). List-of-dict fields (e.g. list_electrode_notes' "notes",
    search_literature's "results") become their own st.dataframe; dict-of-
    dict fields (e.g. get_batch_metadata's "sheets") become a table keyed
    by their own dict key. Everything else -- scalars, and anything too
    irregular to tabulate -- lands in one small properties table at the
    end, so nothing from the raw result is ever silently dropped.
    """
    scalars = {}
    for key, value in result.items():
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            st.caption(key.replace("_", " "))
            st.dataframe(value, width="stretch", hide_index=True)
        elif isinstance(value, dict) and value and all(isinstance(v, dict) for v in value.values()):
            st.caption(key.replace("_", " "))
            rows = [{"id": k, **v} for k, v in value.items()]
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            scalars[key] = value
    if scalars:
        st.dataframe(
            [{"field": k.replace("_", " "), "value": _scalarize(v)} for k, v in scalars.items()],
            width="stretch", hide_index=True,
        )


def _render_tool_call(call, result_str, collected_images):
    name = call["function"]["name"]
    args = call["function"]["arguments"]
    with st.status(f"\U0001f527 {name}", expanded=False) as status:
        st.caption("arguments")
        st.json(args)
        try:
            result = json.loads(result_str)
        except (json.JSONDecodeError, TypeError):
            result = None

        if isinstance(result, dict):
            renderer = dashboard_ui.DASHBOARD_RENDERERS.get(name)
            if renderer is not None:
                try:
                    renderer(result)
                except Exception:
                    # Dashboard rendering is a presentation nicety on top of
                    # the real result -- a formatting bug in it (e.g. an
                    # unexpected field shape) should never hide the actual
                    # QC data, so fall through to the table view below.
                    _render_as_table(result)
                else:
                    with st.expander("raw result"):
                        st.json(result)
            else:
                try:
                    _render_as_table(result)
                except Exception:
                    # Same belt-and-suspenders logic as above -- an
                    # unexpectedly-shaped result should never hide the data.
                    st.json(result)
            images = _find_image_paths(result)
            collected_images.extend(images)
            status.update(label=f"\U0001f527 {name} -- {result.get('status', 'done')}", state="complete")
        else:
            st.text(result_str)
            status.update(label=f"\U0001f527 {name}", state="complete")


def _render_history(contents: list, turn_images: dict) -> None:
    """Replays session history into the chat UI, including every past tool
    call -- not just the final answer text. Without this, a tool-call panel
    (arguments, dashboard card/table) only ever rendered for the single
    script run where it streamed in live; the very next rerun (a new
    message, a retry, anything) silently dropped it, even though the
    underlying function_call/function_response Content entries were still
    sitting right there in `contents` the whole time. Reconstructing the
    display straight from `contents` (rather than caching a second, parallel
    display log) keeps there being exactly one source of truth.

    Walks contents as a flat index rather than a for-loop since one logical
    assistant turn can span several Content entries (one model-with-
    function_call + one user-with-function_response per tool-call round),
    all of which need grouping into a single chat bubble to match how the
    turn looked live.
    """
    i, n = 0, len(contents)
    while i < n:
        content = contents[i]
        if content.role == "user" and not any(getattr(p, "function_response", None) for p in content.parts):
            with st.chat_message("user"):
                st.markdown("".join(p.text for p in content.parts if p.text))
            i += 1
        elif content.role == "model":
            with st.chat_message("assistant"):
                final_text, final_index = None, i
                while i < n and contents[i].role == "model":
                    model_content = contents[i]
                    calls = [p.function_call for p in model_content.parts if getattr(p, "function_call", None)]
                    if calls:
                        i += 1
                        pending = {}
                        if i < n and contents[i].role == "user":
                            for p in contents[i].parts:
                                fr = getattr(p, "function_response", None)
                                if fr:
                                    pending.setdefault(fr.name, []).append(fr.response)
                            i += 1
                        for fc in calls:
                            args = dict(fc.args) if fc.args else {}
                            call_display = {"function": {"name": fc.name, "arguments": args}}
                            bucket = pending.get(fc.name)
                            result = bucket.pop(0) if bucket else {}
                            _render_tool_call(call_display, json.dumps(result, default=str), [])
                    else:
                        text = "".join(p.text for p in model_content.parts if p.text and not getattr(p, "thought", False))
                        if text:
                            final_text, final_index = text, i
                        i += 1
                if final_text:
                    st.markdown(final_text)
                _render_image_gallery(turn_images.get(final_index, []))
        else:
            i += 1  # stray function_response-only content with no preceding call -- shouldn't happen, skip defensively


def _render_image_gallery(images):
    """images: list of (label, path). Shown directly in the main chat flow,
    not nested inside a collapsed tool-call box -- the analysis is the
    point, it shouldn't require a click to see.
    """
    if not images:
        return
    seen = set()
    unique = [im for im in images if not (im[1] in seen or seen.add(im[1]))]
    st.caption(f"\U0001f4ca {len(unique)} analysis image(s)")
    cols = st.columns(min(len(unique), 3))
    for i, (label, path) in enumerate(unique):
        with cols[i % len(cols)]:
            st.image(path, caption=label)


def _stream_turn(
    contents: list,
    collected_images: list,
    generate_config=None,
    show_thinking: bool = True,
    max_hops: int = None,
) -> tuple:
    """Streams each hop live (thinking token-by-token, then the answer), runs
    any function calls that come back, and loops until the model gives a
    final answer with no further function calls (or max_hops is hit). Only
    the final, function-call-free hop's text is treated as the real answer
    -- intermediate hops' content is a partial/incomplete read meant to lead
    into a tool call, not a chat message.

    thinking_placeholder is reserved *before* content_slot each hop so the
    reasoning always renders above the answer regardless of arrival timing
    -- reversed, this used to reserve the answer's slot first, leaving an
    empty box sitting above where the (usually earlier-arriving) reasoning
    would later appear once the expander was created, which read as
    out-of-order/glitchy as text filled in above-then-below out of sequence.

    Never lets a Gemini-call failure (network error, quota/rate-limit,
    safety block) raise out into Streamlit's default traceback view -- shows
    st.error instead and returns whatever text streamed before the failure.
    """
    generate_config = generate_config or agent.GENERATE_CONFIG
    max_hops = max_hops or agent.MAX_HOPS
    turn_start_len = len(contents)

    for hop in range(max_hops):
        # Covers the real latency before the first token of this hop arrives
        # -- cleared the instant anything real shows up, whichever placeholder
        # ends up using it.
        loading_placeholder = st.empty()
        loading_placeholder.markdown(f"_{random.choice(_LOADING_MESSAGES)}_")
        first_part_seen = False

        thinking_placeholder = st.empty()  # reserved first -> renders above the answer
        content_slot = st.empty()
        thinking_slot = None
        thinking_acc = ""
        content_acc = ""
        parts_acc = []
        function_calls = []

        try:
            # Held for the live duration of the stream, not just the initial
            # call -- an open stream still counts against the shared key's
            # concurrent load, and this is what actually caps it (see
            # tools/_gemini.py's request_slot docstring). Chunks still render
            # token-by-token as they arrive; nothing is buffered.
            with agent.request_slot():
                stream = agent.client().models.generate_content_stream(
                    model=agent.MODEL, contents=contents, config=generate_config,
                )
                for chunk in stream:
                    if not chunk.candidates or not chunk.candidates[0].content:
                        continue
                    chunk_parts = chunk.candidates[0].content.parts
                    if not chunk_parts:
                        continue
                    for part in chunk_parts:
                        if not first_part_seen:
                            loading_placeholder.empty()
                            first_part_seen = True
                        parts_acc.append(part)
                        if getattr(part, "thought", False) and part.text:
                            if show_thinking:
                                if thinking_slot is None:
                                    thinking_slot = thinking_placeholder.expander("\U0001f9e0 thinking", expanded=True).empty()
                                thinking_acc += part.text
                                thinking_slot.markdown(thinking_acc)
                        elif getattr(part, "function_call", None):
                            function_calls.append(part.function_call)
                        elif part.text:
                            content_acc += part.text
                            content_slot.markdown(content_acc + "▌")
        except agent.RequestQueueFullError as e:
            loading_placeholder.empty()
            content_slot.empty()
            st.warning(str(e))
            if len(contents) == turn_start_len and not parts_acc:
                if contents and contents[-1].role == "user":
                    contents.pop()
            return content_acc, True
        except genai_errors.APIError as e:
            loading_placeholder.empty()
            content_slot.empty()
            if getattr(e, "code", None) == 429:
                st.warning(
                    "Gemini's rate limit was hit (this demo shares one free-tier API key across "
                    "every visitor) -- wait a few seconds and try again."
                )
            else:
                st.error(f"Gemini API error ({getattr(e, 'code', '?')}): {getattr(e, 'message', e)}")
            if len(contents) == turn_start_len and not parts_acc:
                if contents and contents[-1].role == "user":
                    contents.pop()
            return content_acc, True
        except Exception as e:
            loading_placeholder.empty()
            content_slot.empty()
            st.error(f"Gemini request failed: {e}")
            if len(contents) == turn_start_len and not parts_acc:
                # Nothing was appended for this turn yet and nothing usable
                # streamed back -- drop the dangling user message so history
                # stays well-formed and the researcher can just retry.
                if contents and contents[-1].role == "user":
                    contents.pop()
            return content_acc, True

        if function_calls:
            content_slot.empty()  # intermediate hop -- don't display raw content as a chat message
        else:
            content_slot.markdown(content_acc) if content_acc else content_slot.empty()

        contents.append(types.Content(role="model", parts=parts_acc))

        if not function_calls:
            return content_acc, False

        response_parts = []
        for fc in function_calls:
            args = dict(fc.args) if fc.args else {}
            call_display = {"function": {"name": fc.name, "arguments": args}}
            result = agent.run_tool_call(fc.name, args)
            result_str = json.dumps(result, default=str)
            _render_tool_call(call_display, result_str, collected_images)
            response_parts.append(types.Part.from_function_response(name=fc.name, response=result))

        # The API only accepts "user"/"model" roles -- function responses
        # ride back as a "user" turn, distinguished from real user input by
        # containing function_response parts instead of text.
        contents.append(types.Content(role="user", parts=response_parts))

    # Hit the round-trip cap without a final answer. Every function_call
    # above already got a matching function_response appended, so contents
    # stays well-formed -- stop asking for more hops and say why.
    st.warning(f"Stopped after {max_hops} tool-call round-trips without a final answer.")
    stop_text = (
        f"(Stopped after {max_hops} tool-call round-trips without a final answer -- "
        "try breaking the request into smaller steps.)"
    )
    contents.append(types.Content(role="model", parts=[types.Part.from_text(text=stop_text)]))
    return stop_text, True


MAX_MESSAGE_CHARS = 4000  # a runaway paste shouldn't be able to single-handedly burn a big chunk of shared quota
RATE_LIMIT_WINDOW_S = 60
RATE_LIMIT_MAX_MESSAGES = 10  # generous for real use, tight enough to blunt a script hammering the public URL
MAX_UPLOADS_PER_SESSION = 15  # caps disk use if someone uploads many files in one session


def _rate_limited() -> str:
    """Returns a warning string if this session should be throttled, else "".
    The global request_slot cap (tools/_gemini.py) protects the shared key
    across every session at once; this protects it from any single
    session -- an accidental loop, or someone scripting against the public
    URL -- sending a burst on its own.
    """
    now = time.time()
    st.session_state.message_timestamps = [t for t in st.session_state.message_timestamps if now - t < RATE_LIMIT_WINDOW_S]
    if len(st.session_state.message_timestamps) >= RATE_LIMIT_MAX_MESSAGES:
        wait = RATE_LIMIT_WINDOW_S - (now - st.session_state.message_timestamps[0])
        return f"Sending messages a little fast -- please wait about {max(1, int(wait))}s (keeps the shared demo responsive for everyone else too)."
    return ""


def chat_page() -> None:
    if "contents" not in st.session_state:
        st.session_state.contents = []
    if "turn_images" not in st.session_state:
        st.session_state.turn_images = {}  # contents index -> [(label, path), ...]
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "pending_upload_path" not in st.session_state:
        st.session_state.pending_upload_path = None
    if "session_id" not in st.session_state:
        # Scopes this session's uploads to their own subfolder so two visitors
        # (e.g. two judges) uploading a same-named file (very plausible --
        # "photo.jpg" straight off a phone) never silently overwrite each other.
        st.session_state.session_id = uuid.uuid4().hex[:8]
    if "message_timestamps" not in st.session_state:
        st.session_state.message_timestamps = []
    if "retry_pending" not in st.session_state:
        st.session_state.retry_pending = None  # (display_text, model_text, caption) of the last failed turn, or None

    with st.sidebar:
        st.header("SenseOne")
        st.caption(f"model: `{agent.MODEL}` (Gemini API)")
        if not agent.api_key_configured():
            st.error(
                "GEMINI_API_KEY is not set -- see README for setup "
                "(local: `.env`/`export`; hosted: Streamlit's Secrets manager)."
            )
        col_new, col_check = st.columns(2)
        if col_new.button("New conversation"):
            st.session_state.contents = []
            st.session_state.turn_images = {}
            st.session_state.retry_pending = None
            st.rerun()
        if col_check.button("\U0001f50c Check connection", help="Sends one small test request to confirm the API key/quota are actually working -- worth a click right before demoing."):
            if not agent.api_key_configured():
                st.error("No API key configured.")
            else:
                with st.spinner("Pinging Gemini..."):
                    try:
                        with agent.request_slot():
                            probe = agent.client().models.generate_content(
                                model=agent.MODEL, contents="Reply with exactly one word: OK",
                                config=types.GenerateContentConfig(temperature=0, thinking_config=types.ThinkingConfig(include_thoughts=False)),
                            )
                        if probe.text and "OK" in probe.text.upper():
                            st.success(f"Connected -- {agent.MODEL} responded normally.")
                        else:
                            st.warning(f"Got a response, but unexpected content: {probe.text!r}")
                    except Exception as e:
                        st.error(f"Connection check failed: {e}")

        with st.expander("⚙️ Advanced settings"):
            temperature = st.slider(
                "Temperature", 0.0, 1.0, agent.DEFAULT_TEMPERATURE, 0.05,
                help=(
                    "Lower = more consistent, literal answers (recommended for QC -- "
                    f"the system prompt is tuned around the default, {agent.DEFAULT_TEMPERATURE}). "
                    "Higher = more varied phrasing, more prone to embellishing beyond what a tool actually reported."
                ),
            )
            show_thinking = st.checkbox(
                "Show reasoning", value=True,
                help="Stream the model's visible thinking before its answer. Turn off for a cleaner, answer-only view.",
            )
            max_hops = st.slider(
                "Max tool-call rounds / turn", 1, 30, agent.MAX_HOPS,
                help="Safety cap -- stops a confused model from looping on tool calls indefinitely and burning shared API quota.",
            )

        st.divider()
        st.caption("upload a photo")
        uploaded = st.file_uploader(
            "electrode photo", type=["png", "jpg", "jpeg", "bmp"],
            key=f"uploader_{st.session_state.uploader_key}", label_visibility="collapsed",
        )
        if uploaded is not None:
            data = uploaded.getvalue()
            if len(data) > MAX_UPLOAD_BYTES:
                st.error(f"That file is {len(data) / 1e6:.1f} MB -- max {MAX_UPLOAD_BYTES / 1e6:.0f} MB.")
            else:
                try:
                    # Validate the bytes are actually a decodable image before
                    # saving/using them -- the uploader's type= filter is
                    # client-side only, and this file's path eventually gets
                    # read and sent to the Gemini API as "image data" by
                    # tools/image_qc.py, so a mistyped/corrupted upload should
                    # fail here with a clear message, not deep inside a tool call.
                    with Image.open(io.BytesIO(data)) as probe:
                        probe.verify()
                except Exception as e:
                    st.error(f"That doesn't look like a valid image file ({e}).")
                else:
                    session_dir = UPLOAD_DIR / st.session_state.session_id
                    session_dir.mkdir(parents=True, exist_ok=True)
                    # Cap how many files one session accumulates -- ephemeral
                    # storage on a hosted instance is finite, and nothing here
                    # ever deletes an old upload otherwise.
                    existing = sorted(session_dir.glob("*"), key=lambda p: p.stat().st_mtime)
                    for stale in existing[:max(0, len(existing) - MAX_UPLOADS_PER_SESSION + 1)]:
                        stale.unlink(missing_ok=True)
                    # Path(...).name strips any directory components a crafted
                    # filename might carry -- never trust a browser-supplied
                    # filename as a bare relative path.
                    safe_name = Path(uploaded.name).name or "upload"
                    save_path = session_dir / safe_name
                    save_path.write_bytes(data)
                    st.session_state.pending_upload_path = str(save_path)
                    st.image(str(save_path), caption=uploaded.name, width=150)
                    st.caption(f"Will be attached to your next message as `{save_path}`.")

        st.divider()
        st.caption("batches on record")
        img_root = Path("reference_images")
        if img_root.exists():
            for batch_dir in sorted(p for p in img_root.iterdir() if p.is_dir()):
                n = len(list(batch_dir.glob("*")))
                st.text(f"{batch_dir.name}  ({n} photos)")

    st.title("\U0001f52c SenseOne")
    st.caption("Gemini-backed research assistant for the electrochemical biosensor lab.")

    _render_history(st.session_state.contents, st.session_state.turn_images)

    def _handle_turn(display_text: str, model_text: str, caption: str = None) -> None:
        """Runs one full user turn: appends the user message, streams the
        assistant's reply, and tracks whether it needs a retry affordance.
        Wrapped in a last-resort try/except -- every Gemini-specific failure
        mode is already handled inside _stream_turn, so anything that reaches
        here is genuinely unanticipated (e.g. a rendering bug), and should
        never surface Streamlit's raw crash page in front of a live audience.
        """
        st.session_state.message_timestamps.append(time.time())
        st.session_state.contents.append(types.Content(role="user", parts=[types.Part.from_text(text=model_text)]))
        with st.chat_message("user"):
            st.markdown(display_text)
            if caption:
                st.caption(caption)
        with st.chat_message("assistant"):
            images = []
            try:
                _, failed = _stream_turn(
                    st.session_state.contents, images,
                    generate_config=agent.build_generate_config(temperature=temperature, include_thoughts=show_thinking),
                    show_thinking=show_thinking, max_hops=max_hops,
                )
            except Exception as e:
                st.error(f"Something unexpected went wrong ({type(e).__name__}). Try \"New conversation\" if this persists.")
                failed = True
            _render_image_gallery(images)
            if images:
                # final assistant turn is the last one appended
                st.session_state.turn_images[len(st.session_state.contents) - 1] = images

        st.session_state.retry_pending = (display_text, model_text, caption) if failed else None

    rate_limit_msg = _rate_limited()
    if rate_limit_msg:
        st.warning(rate_limit_msg)

    if st.session_state.retry_pending:
        retry_display, _, _ = st.session_state.retry_pending
        label = retry_display if len(retry_display) <= 60 else retry_display[:57] + "..."
        if st.button(f"\U0001f504 Retry: “{label}”", disabled=bool(rate_limit_msg)):
            display_text, model_text, caption = st.session_state.retry_pending
            st.session_state.retry_pending = None
            _handle_turn(display_text, model_text, caption)
            st.rerun()

    user_input = st.chat_input(
        "Ask about a sensor, image, paper, or batch..."
        if agent.api_key_configured() else "Set GEMINI_API_KEY to start chatting (see sidebar)",
        disabled=not agent.api_key_configured() or bool(rate_limit_msg),
        max_chars=MAX_MESSAGE_CHARS,
        submit_mode="disable",  # disables the input while a turn is in flight -- blocks double-submit spam on the shared key
    )
    if user_input:
        pending = st.session_state.pending_upload_path
        model_input = f"{user_input}\n\n[Uploaded image saved at: {pending}]" if pending else user_input
        st.session_state.pending_upload_path = None
        st.session_state.uploader_key += 1  # force a fresh, empty uploader widget so this file isn't re-attached to later messages

        _handle_turn(user_input, model_input, caption=f"\U0001f4ce {pending}" if pending else None)
        st.rerun()


def dashboard_page() -> None:
    dashboard_ui.render_batch_dashboard_page()


def guide_page() -> None:
    st.title("\U0001f4d6 User Guide")

    st.header("Why this exists")
    st.markdown(
        "Screen-printed electrodes (SPEs) for electrochemical biosensors are normally QC'd by "
        "actually running them -- cyclic voltammetry (CV) and chronoamperometry (CA) -- which "
        "costs real bench time (potentiostat time, reagents, sample) per electrode. That's the "
        "ground-truth signal, but it's slow, and it happens *after* fabrication, when a bad print "
        "can no longer be caught cheaply.\n\n"
        "SenseOne is built around a faster signal that's available before any electrochemistry "
        "runs at all: a photo of the printed electrode. The open question is whether that photo "
        "actually predicts electrical performance -- do defects, ink coverage, or surface "
        "roughness visible in a photo correlate with CV peak behavior -- and if so, by how much, "
        "starting from how little data. It's also a lab record-keeper: raw CV/CA exports, "
        "electrode photos, fabrication notes, and literature accumulate across sessions in a way "
        "that's easy to lose track of by hand."
    )

    st.header("What it can do")
    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            "**QC electrochemical data**\n\n"
            "Point it at a CV/CA/SWV CSV export -- peak current, ΔEp, noise, LOD, sensitivity, "
            "scan-to-scan stability."
        )
        st.markdown(
            "**QC electrode photos**\n\n"
            "Vision-model defect read, plus a luminance-based surface roughness proxy, plus "
            "unsupervised outlier detection against the rest of the batch."
        )
    with cols[1]:
        st.markdown(
            "**Predict performance**\n\n"
            "Correlates visual features against CV outcomes, and only asserts a real prediction "
            "once that correlation clears statistical significance -- otherwise it says so, "
            "explicitly, rather than guessing."
        )
        st.markdown(
            "**Remember, automatically**\n\n"
            "Every QC check writes a per-electrode/per-batch record without being asked, so "
            "history is there next session -- no manual logging."
        )
    with cols[2]:
        st.markdown(
            "**Search the literature**\n\n"
            "Local vault (PubMed/arXiv, cached, open-access prioritized) plus live web search, "
            "with a real clickable citation on every claim."
        )
        st.markdown(
            "**Suggest next steps**\n\n"
            "When a QC check flags something, it can pull literature-backed remediation ideas -- "
            "always cited, never a generic \"try a different ink\" guess."
        )

    st.header("How the agent actually works")
    st.markdown(
        "SenseOne is a **tool-calling agent**, not a fixed pipeline or a fine-tuned model -- "
        f"there's no hardcoded \"if photo, do X\" logic. A single Gemini model "
        f"(`{agent.MODEL}`) reads your message, decides for itself which of "
        f"{len(agent.TOOLS)} tools it actually needs (and in what order), calls them, reads the "
        "results, and either calls more tools or answers. That loop can run several rounds in "
        "one turn -- e.g. checking a photo's framing, then reading its defects, then pulling up "
        "the batch's fabrication metadata to explain *why* a defect might have happened, then "
        "searching the literature for a fix -- all before it writes a single word back to you. "
        "It stops once it has no more tool calls left to make, or after a hop-count safety cap "
        "if something goes wrong (protects the shared API quota from a runaway loop)."
    )
    st.graphviz_chart("""
        digraph {
            rankdir=TB;
            bgcolor="transparent";
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, margin="0.18,0.1", color="#4472C4"];
            edge [fontname="Helvetica", fontsize=10, color="#666666", fontcolor="#666666"];

            input [label="Researcher message\n(+ optional photo)", fillcolor="#E8EEF9"];
            agent [label="Gemini reasons:\nwhich tool(s) does this need?", fillcolor="#FFF2CC", color="#BF9000"];
            tools [label="Tool(s) called\n(sensor_qc, image_qc,\nsearch_literature, ...)", fillcolor="#E2F0D9", color="#548235"];
            result [label="Result fed back\ninto the conversation", fillcolor="#E2F0D9", color="#548235"];
            answer [label="Grounded final answer\n(every number traces to a call)", fillcolor="#E8EEF9"];

            input -> agent;
            agent -> tools [label="needs a tool"];
            tools -> result;
            result -> agent [label="loop: more tools needed?", style=dashed];
            agent -> answer [label="no more tools needed"];
        }
    """)
    st.caption("One turn can loop through this several times before you see an answer -- e.g. framing check, then defect read, then a literature search for a fix, all in one response.")
    st.markdown(
        "**Nothing is hidden.** Every tool call streams into its own expandable panel in the "
        "chat -- the exact arguments sent and the exact result returned, before the final answer "
        "even appears. If you ever want to know where a number came from, that panel is where it "
        "actually is; the agent is instructed to trace every number, status, and citation in its "
        "final answer back to one of those calls, or flag it plainly as its own inference instead "
        "of stating it as fact. You can also toggle on \"Show reasoning\" in the sidebar to watch "
        "its visible thinking before it commits to an answer."
    )
    st.markdown(
        "**Some of this is automatic and deliberately invisible until it matters.** Every photo, "
        "for instance, silently goes through the framing-quality gate described below as the "
        "*first* step of any photo tool -- you never call that check yourself; a bad photo just "
        "gets rejected with a specific reason before any analysis is attempted, rather than "
        "quietly producing a guess from unusable input."
    )

    st.header("Uploading a photo")
    st.markdown(
        "Every photo goes through an automatic framing check before any analysis happens -- a "
        "photo that fails is rejected outright with a specific reason, not silently guessed at. "
        "Subject validity is checked first, on every photo, before anything else -- there's no "
        "point measuring the blur of a photo of the wrong thing entirely:"
    )
    st.graphviz_chart("""
        digraph {
            rankdir=TB;
            bgcolor="transparent";
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.15,0.08", color="#4472C4"];
            edge [fontname="Helvetica", fontsize=9, color="#666666", fontcolor="#666666"];

            photo [label="Photo submitted", fillcolor="#E8EEF9"];
            subject [label="Is this a recognizable\nelectrode at all?", shape=diamond, fillcolor="#FFF2CC", color="#BF9000"];
            reject1 [label="Rejected: not an electrode", fillcolor="#FCE4E4", color="#C00000"];
            cheap [label="Blur / lighting /\noff-centre / resolution", shape=diamond, fillcolor="#FFF2CC", color="#BF9000"];
            reject2 [label="Rejected, specific reason\n(e.g. \\"too dark\\")", fillcolor="#FCE4E4", color="#C00000"];
            vision [label="Angle / overlap / flipped /\ntampered / mixed types", shape=diamond, fillcolor="#FFF2CC", color="#BF9000"];
            reject3 [label="Rejected, specific reason\n(e.g. \\"electrodes overlap\\")", fillcolor="#FCE4E4", color="#C00000"];
            analyze [label="Proceeds to analysis", fillcolor="#E2F0D9", color="#548235"];

            photo -> subject;
            subject -> reject1 [label="no"];
            subject -> cheap [label="yes"];
            cheap -> reject2 [label="fails"];
            cheap -> vision [label="passes"];
            vision -> reject3 [label="fails"];
            vision -> analyze [label="passes"];
        }
    """)
    st.markdown("In full, a photo needs to be:")
    st.markdown(
        "- **One electrode, correctly identified as one** -- a real SPE strip, not a batch sheet, "
        "a design mockup, or something else entirely\n"
        "- **Front side up** -- printed conductive pads clearly visible, not the blank/reverse side\n"
        "- **In focus and well-lit** -- not blurry, not too dark, no blown-out glare\n"
        "- **Centered in frame**, not cut off at the edges\n"
        "- **At least 640×480px** -- not a thumbnail or heavily downscaled export\n"
        "- **Undamaged and untampered** -- no visible cuts, gouges, or foreign objects on the strip\n"
        "- **Not overlapping another electrode**, and not mixed with a different electrode design "
        "in the same frame"
    )
    st.caption(
        "If a photo is rejected, the message says exactly which of these failed and how to "
        "retake it -- the electrode simply isn't analyzed rather than guessed at from a bad photo."
    )

    st.header("How to ask")
    st.markdown(
        "Talk to it like a lab colleague, not a command line -- it figures out which tool(s) a "
        "question needs. A few examples:"
    )
    st.code(
        "Run QC on reference_data/20260707_cv/707-A1-1.csv\n"
        "Does this photo look unusual compared to the rest of batch 20260707?\n"
        "What does the literature say about improving ink adhesion on flexible substrates?\n"
        "Is electrode A5's roughness related to its CV peak current at all?",
        language=None,
    )
    st.markdown(
        "Attach a photo via the uploader in the sidebar, then mention it in your message -- it's "
        "attached to your next turn automatically. Every tool call it makes along the way is "
        "shown in an expandable panel with the exact arguments and result, so any number in the "
        "final answer can be traced back to where it actually came from."
    )

    st.header("Worth knowing")
    st.markdown(
        "- **Predictions are gated on real statistics.** A \"preliminary\" result means the "
        "correlation isn't statistically significant yet with the data collected so far -- not "
        "that the tool is unsure how to phrase it.\n"
        "- **Roughness is a luminance proxy**, not calibrated physical roughness -- useful for "
        "comparing electrodes to each other, not as an absolute spec.\n"
        "- **This runs on a shared hosted API key.** During a busy demo you may occasionally see "
        "a rate-limit warning -- wait a few seconds and retry.\n"
        "- Check the **Batch Dashboard** tab for an at-a-glance pass/fail view of an entire batch "
        "without spending a model call."
    )


pg = st.navigation([
    st.Page(chat_page, title="Chat", icon="\U0001f4ac", default=True),
    st.Page(dashboard_page, title="Batch Dashboard", icon="\U0001f4ca"),
    st.Page(guide_page, title="User Guide", icon="\U0001f4d6"),
])
pg.run()
