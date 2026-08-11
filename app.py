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

import json
import os
from pathlib import Path

import streamlit as st
from google.genai import types

import agent

UPLOAD_DIR = Path("uploads")

st.set_page_config(page_title="SenseOne", page_icon="\U0001f52c", layout="wide")


def _find_image_paths(obj, found=None):
    """Recursively pull out every *_path-keyed string in a tool result that
    points at an existing image file, so it can be shown inline -- tool
    results nest these at different depths (e.g. image_qc's own image_path
    vs its nested surface_analysis.plot_path).
    """
    if found is None:
        found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and key.endswith("path") and value.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                if os.path.exists(value):
                    found.append((key, value))
            else:
                _find_image_paths(value, found)
    elif isinstance(obj, list):
        for item in obj:
            _find_image_paths(item, found)
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
    final answer with no further function calls. Only the final,
    function-call-free hop's text is treated as the real answer --
    intermediate hops' content is a partial/incomplete read meant to lead
    into a tool call, not a chat message.
    """
    while True:
        thinking_slot = None
        content_slot = st.empty()
        thinking_acc = ""
        content_acc = ""
        parts_acc = []
        function_calls = []

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


if "contents" not in st.session_state:
    st.session_state.contents = []
if "turn_images" not in st.session_state:
    st.session_state.turn_images = {}  # contents index -> [(label, path), ...]
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "pending_upload_path" not in st.session_state:
    st.session_state.pending_upload_path = None

with st.sidebar:
    st.header("SenseOne")
    st.caption(f"model: `{agent.MODEL}` (Gemini API)")
    if not os.environ.get("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY is not set -- see README for setup.")
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
        UPLOAD_DIR.mkdir(exist_ok=True)
        save_path = UPLOAD_DIR / uploaded.name
        save_path.write_bytes(uploaded.getvalue())
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

user_input = st.chat_input("Ask about a sensor, image, paper, or batch...")
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
