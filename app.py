"""
SenseOne -- local web GUI (Streamlit).

Thin presentation layer over agent.py: reuses its SYSTEM_PROMPT, TOOLS,
run_tool_call, and _sized_options directly rather than re-implementing the
tool-calling loop. Beyond the CLI, this adds: inline display of the
images/plots the tools already generate (surface plots, batch-diff
visualizations, electrode photos) directly in the chat, a live
token-by-token view of the model's thinking instead of a wall of silence
until it's done, and a way to upload a new photo from the browser (the
tools all take a file path -- an uploaded image is saved to uploads/ and
its path handed to the model).

Run:
    streamlit run app.py
"""

import json
import os
from pathlib import Path

import streamlit as st

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


def _stream_turn(messages: list, collected_images: list) -> str:
    """Streams each hop live (thinking token-by-token, then content), runs
    any tool calls that come back, and loops until the model gives a final
    answer with no further tool calls. Only the final, tool-call-free hop's
    content is treated as the real answer -- intermediate hops sometimes
    stream raw tool-call syntax through the content field before Ollama
    finishes parsing it into a structured tool call, which isn't meant to
    be shown as a chat message.
    """
    while True:
        thinking_slot = None
        content_slot = st.empty()
        thinking_acc = ""
        content_acc = ""
        tool_calls = None

        stream = agent.ollama.chat(
            model=agent.MODEL, messages=messages, tools=agent.TOOLS,
            think=True, options=agent._sized_options(messages), stream=True,
        )
        for chunk in stream:
            m = chunk["message"]
            if m.get("thinking"):
                if thinking_slot is None:
                    thinking_slot = st.expander("\U0001f9e0 thinking", expanded=True).empty()
                thinking_acc += m["thinking"]
                thinking_slot.markdown(thinking_acc)
            if m.get("content"):
                content_acc += m["content"]
                content_slot.markdown(content_acc + "▌")
            if m.get("tool_calls"):
                tool_calls = m["tool_calls"]

        assistant_msg = {"role": "assistant", "content": content_acc}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
            content_slot.empty()  # intermediate hop -- don't display raw/leaked content as a chat message
        else:
            content_slot.markdown(content_acc) if content_acc else content_slot.empty()

        messages.append(assistant_msg)

        if not tool_calls:
            return content_acc

        for call in tool_calls:
            tool_result = agent.run_tool_call(call)
            _render_tool_call(call, tool_result, collected_images)
            messages.append({"role": "tool", "name": call["function"]["name"], "content": tool_result})


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": agent.SYSTEM_PROMPT}]
if "turn_images" not in st.session_state:
    st.session_state.turn_images = {}  # user-message index (in st.session_state.messages) -> [(label, path), ...]
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "pending_upload_path" not in st.session_state:
    st.session_state.pending_upload_path = None

with st.sidebar:
    st.header("SenseOne")
    st.caption(f"model: `{agent.MODEL}`")
    if st.button("New conversation"):
        st.session_state.messages = [{"role": "system", "content": agent.SYSTEM_PROMPT}]
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
st.caption("Local, Ollama-backed research assistant for the electrochemical biosensor lab.")

for i, msg in enumerate(st.session_state.messages[1:], start=1):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant" and msg.get("content") and not msg.get("tool_calls"):
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            _render_image_gallery(st.session_state.turn_images.get(i, []))

user_input = st.chat_input("Ask about a sensor, image, paper, or batch...")
if user_input:
    pending = st.session_state.pending_upload_path
    model_input = f"{user_input}\n\n[Uploaded image saved at: {pending}]" if pending else user_input
    st.session_state.pending_upload_path = None
    st.session_state.uploader_key += 1  # force a fresh, empty uploader widget so this file isn't re-attached to later messages

    st.session_state.messages.append({"role": "user", "content": model_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        if pending:
            st.caption(f"\U0001f4ce {pending}")
    with st.chat_message("assistant"):
        images = []
        _stream_turn(st.session_state.messages, images)
        _render_image_gallery(images)
        if images:
            # final assistant message is the last one appended
            st.session_state.turn_images[len(st.session_state.messages) - 1] = images
    st.rerun()
