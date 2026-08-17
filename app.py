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
import uuid
from pathlib import Path

import streamlit as st
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image

import agent

UPLOAD_DIR = Path("uploads")
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # keep in sync with .streamlit/config.toml's server.maxUploadSize

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


def _stream_turn(contents: list, collected_images: list) -> str:
    """Streams each hop live (thinking token-by-token, then content), runs
    any function calls that come back, and loops until the model gives a
    final answer with no further function calls (or MAX_HOPS is hit). Only
    the final, function-call-free hop's text is treated as the real answer
    -- intermediate hops' content is a partial/incomplete read meant to lead
    into a tool call, not a chat message.

    Never lets a Gemini-call failure (network error, quota/rate-limit,
    safety block) raise out into Streamlit's default traceback view -- shows
    st.error instead and returns whatever text streamed before the failure.
    """
    turn_start_len = len(contents)

    for hop in range(agent.MAX_HOPS):
        thinking_slot = None
        content_slot = st.empty()
        thinking_acc = ""
        content_acc = ""
        parts_acc = []
        function_calls = []

        try:
            stream = agent.client().models.generate_content_stream(
                model=agent.MODEL, contents=contents, config=agent.GENERATE_CONFIG,
            )
            for chunk in stream:
                if not chunk.candidates or not chunk.candidates[0].content:
                    continue
                chunk_parts = chunk.candidates[0].content.parts
                if not chunk_parts:
                    continue
                for part in chunk_parts:
                    parts_acc.append(part)
                    if getattr(part, "thought", False) and part.text:
                        if thinking_slot is None:
                            thinking_slot = st.expander("\U0001f9e0 thinking", expanded=True).empty()
                        thinking_acc += part.text
                        thinking_slot.markdown(thinking_acc)
                    elif getattr(part, "function_call", None):
                        function_calls.append(part.function_call)
                    elif part.text:
                        content_acc += part.text
                        content_slot.markdown(content_acc + "▌")
        except genai_errors.APIError as e:
            content_slot.empty()
            st.error(f"Gemini API error ({getattr(e, 'code', '?')}): {getattr(e, 'message', e)}")
            if len(contents) == turn_start_len and not parts_acc:
                if contents and contents[-1].role == "user":
                    contents.pop()
            return content_acc
        except Exception as e:
            content_slot.empty()
            st.error(f"Gemini request failed: {e}")
            if len(contents) == turn_start_len and not parts_acc:
                # Nothing was appended for this turn yet and nothing usable
                # streamed back -- drop the dangling user message so history
                # stays well-formed and the researcher can just retry.
                if contents and contents[-1].role == "user":
                    contents.pop()
            return content_acc

        if function_calls:
            content_slot.empty()  # intermediate hop -- don't display raw content as a chat message
        else:
            content_slot.markdown(content_acc) if content_acc else content_slot.empty()

        contents.append(types.Content(role="model", parts=parts_acc))

        if not function_calls:
            return content_acc

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
    st.warning(f"Stopped after {agent.MAX_HOPS} tool-call round-trips without a final answer.")
    stop_text = (
        f"(Stopped after {agent.MAX_HOPS} tool-call round-trips without a final answer -- "
        "try breaking the request into smaller steps.)"
    )
    contents.append(types.Content(role="model", parts=[types.Part.from_text(text=stop_text)]))
    return stop_text


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

with st.sidebar:
    st.header("SenseOne")
    st.caption(f"model: `{agent.MODEL}` (Gemini API)")
    if not agent.api_key_configured():
        st.error(
            "GEMINI_API_KEY is not set -- see README for setup "
            "(local: `.env`/`export`; hosted: Streamlit's Secrets manager)."
        )
    if st.button("New conversation"):
        st.session_state.contents = []
        st.session_state.turn_images = {}
        st.rerun()

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

user_input = st.chat_input(
    "Ask about a sensor, image, paper, or batch..."
    if agent.api_key_configured() else "Set GEMINI_API_KEY to start chatting (see sidebar)",
    disabled=not agent.api_key_configured(),
)
if user_input:
    pending = st.session_state.pending_upload_path
    model_input = f"{user_input}\n\n[Uploaded image saved at: {pending}]" if pending else user_input
    st.session_state.pending_upload_path = None
    st.session_state.uploader_key += 1  # force a fresh, empty uploader widget so this file isn't re-attached to later messages

    st.session_state.contents.append(types.Content(role="user", parts=[types.Part.from_text(text=model_input)]))
    with st.chat_message("user"):
        st.markdown(user_input)
        if pending:
            st.caption(f"\U0001f4ce {pending}")
    with st.chat_message("assistant"):
        images = []
        _stream_turn(st.session_state.contents, images)
        _render_image_gallery(images)
        if images:
            # final assistant turn is the last one appended
            st.session_state.turn_images[len(st.session_state.contents) - 1] = images
    st.rerun()
