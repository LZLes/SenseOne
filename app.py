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
                    # QC data, so fall through to the plain JSON view below.
                    st.caption("result")
                    st.json(result)
                else:
                    with st.expander("raw result"):
                        st.json(result)
            else:
                st.caption("result")
                st.json(result)
            images = _find_image_paths(result)
            collected_images.extend(images)
            status.update(label=f"\U0001f527 {name} -- {result.get('status', 'done')}", state="complete")
        else:
            st.text(result_str)
            status.update(label=f"\U0001f527 {name}", state="complete")


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

    for i, content in enumerate(st.session_state.contents):
        if content.role == "user" and not any(getattr(p, "function_response", None) for p in content.parts):
            with st.chat_message("user"):
                st.markdown("".join(p.text for p in content.parts if p.text))
        elif content.role == "model" and not any(getattr(p, "function_call", None) for p in content.parts):
            text = "".join(p.text for p in content.parts if p.text and not getattr(p, "thought", False))
            if text:
                with st.chat_message("assistant"):
                    st.markdown(text)
                    _render_image_gallery(st.session_state.turn_images.get(i, []))

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


pg = st.navigation([
    st.Page(chat_page, title="Chat", icon="\U0001f4ac", default=True),
    st.Page(dashboard_page, title="Batch Dashboard", icon="\U0001f4ca"),
])
pg.run()
