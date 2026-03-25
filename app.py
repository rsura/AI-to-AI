"""
app.py — AI-to-AI: two models conversing with each other.

Supports two providers:
  - Ollama  — local models via the Ollama daemon (keep_alive=-1 keeps them loaded)
  - OpenRouter — cloud models via OpenRouter's OpenAI-compatible SSE endpoint

Conversation loop (driven by st.rerun):
  1. Render completed messages from display history.
  2. If running and not paused, stream the current speaker's response.
  3. Append the completed response to both models' private histories.
  4. Either pause (if requested) or flip the speaker and rerun.

Each model holds its own message list from its own perspective:
  - Its own replies    → role: "assistant"
  - The other's replies → role: "user"
"""

from datetime import datetime

import streamlit as st

from constants import DEFAULTS
from ollama_utils import is_farewell, stream_response
from openrouter_utils import stream_response_openrouter
from pdf_export import generate_pdf
from sidebar import render_sidebar


# ─────────────────────────────────────────────────────────────────────────────
# Session state — initialise missing keys on first load
# ─────────────────────────────────────────────────────────────────────────────

for _key, _val in DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


# ─────────────────────────────────────────────────────────────────────────────
# Page config (must come before any other st.* calls that render output)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="2 AIs Talking",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 2 AIs Talking")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

render_sidebar()


# ─────────────────────────────────────────────────────────────────────────────
# Download PDF button — top of main area
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.display_messages:
    pdf_bytes = generate_pdf()
    if pdf_bytes:
        fname = f"2-ais-talking-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf"
        st.download_button(
            label="📄 Download PDF",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
        )
    else:
        st.button(
            "📄 Download PDF",
            disabled=True,
            help="Install fpdf2 to enable PDF export: `pip install fpdf2`",
        )
else:
    st.button(
        "📄 Download PDF",
        disabled=True,
        help="Start a conversation to enable PDF download",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chat display — render completed messages from history
# ─────────────────────────────────────────────────────────────────────────────

def render_history():
    """Render all messages in display_messages using Streamlit's chat UI."""
    avatar1 = st.session_state.get("inp_avatar1", "⬛️")
    avatar2 = st.session_state.get("inp_avatar2", "⬜️")
    avatars = [avatar1, avatar2]

    for msg in st.session_state.display_messages:
        spk = msg["speaker"]
        if spk == -1:
            with st.chat_message("user", avatar="📝"):
                st.markdown(f"**Opening prompt:** {msg['content']}")
        else:
            with st.chat_message("assistant", avatar=avatars[spk]):
                st.markdown(f"**{msg['model']}**")
                st.write(msg["content"])


render_history()


# ─────────────────────────────────────────────────────────────────────────────
# Conversation-ended banner
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.conversation_ended:
    st.warning(
        "The conversation has concluded — one of the AIs said farewell. "
        "Click **🔄 Reset everything** in the sidebar to start a new conversation.",
        icon="👋",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Butt-in UI — shown only while paused
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.paused and st.session_state.running:
    st.info("⏸ Conversation paused. Inject a message as either AI, then resume.", icon="✋")

    m1_name = st.session_state.model1 or "AI 1"
    m2_name = st.session_state.model2 or "AI 2"
    avatar1 = st.session_state.get("inp_avatar1", "⬛️")
    avatar2 = st.session_state.get("inp_avatar2", "⬜️")

    def inject_as(speaker_idx: int, text: str) -> None:
        """
        Inject text into the conversation as if speaker_idx said it.

        Updates both models' histories consistently:
          - Appended as role:"assistant" for the speaking model
            (so it "remembers" having said it).
          - Appended as role:"user" for the other model
            (so the other model sees it as an incoming message).

        Hands the next turn to the other model.
        """
        other_idx = 1 - speaker_idx
        msgs = [st.session_state.messages_ai1, st.session_state.messages_ai2]
        name  = [m1_name, m2_name][speaker_idx]

        msgs[speaker_idx].append({"role": "assistant", "content": text})
        msgs[other_idx].append({"role": "user", "content": text})

        st.session_state.display_messages.append(
            {"speaker": speaker_idx, "model": f"{name} (you)", "content": text}
        )
        st.session_state.update(
            current_speaker=other_idx,
            paused=False,
            pause_requested=False,
        )

    col_a, col_b = st.columns(2)

    with col_a:
        with st.expander(f"💬 Butt in as {avatar1} {m1_name}", expanded=True):
            butin1_text = st.text_area("Your message as AI 1", key="butin1_text", height=100)
            if st.button(f"Send as {m1_name}", key="butin1_send", use_container_width=True):
                if butin1_text.strip():
                    inject_as(0, butin1_text.strip())
                    st.rerun()

    with col_b:
        with st.expander(f"💬 Butt in as {avatar2} {m2_name}", expanded=True):
            butin2_text = st.text_area("Your message as AI 2", key="butin2_text", height=100)
            if st.button(f"Send as {m2_name}", key="butin2_send", use_container_width=True):
                if butin2_text.strip():
                    inject_as(1, butin2_text.strip())
                    st.rerun()

    col_resume, _ = st.columns([1, 2])
    with col_resume:
        if st.button("▶ Resume without injecting", use_container_width=True):
            st.session_state.update(paused=False, pause_requested=False)
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Streaming conversation turn
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.running and not st.session_state.paused:
    spk      = st.session_state.current_speaker
    model    = [st.session_state.model1, st.session_state.model2][spk]
    messages = [st.session_state.messages_ai1, st.session_state.messages_ai2][spk]
    other_spk = 1 - spk

    avatar1 = st.session_state.get("inp_avatar1", "⬛️")
    avatar2 = st.session_state.get("inp_avatar2", "⬜️")
    avatar  = [avatar1, avatar2][spk]

    with st.chat_message("assistant", avatar=avatar):
        st.markdown(f"**{model}**")
        try:
            if st.session_state.get("provider") == "OpenRouter":
                api_key = st.session_state.get("openrouter_api_key", "")
                full_response: str = st.write_stream(
                    stream_response_openrouter(api_key, model, messages)
                )
            else:
                full_response: str = st.write_stream(stream_response(model, messages))
        except Exception as exc:
            st.session_state.update(running=False, paused=False, pause_requested=False)
            st.error(f"**Error from {st.session_state.get('provider', 'provider')}:** {exc}")
            st.stop()

    messages.append({"role": "assistant", "content": full_response})

    other_msgs = [st.session_state.messages_ai1, st.session_state.messages_ai2][other_spk]
    other_msgs.append({"role": "user", "content": full_response})

    st.session_state.display_messages.append(
        {"speaker": spk, "model": model, "content": full_response}
    )

    if is_farewell(full_response):
        st.session_state.update(
            running=False,
            paused=False,
            pause_requested=False,
            conversation_ended=True,
        )
        st.rerun()
    elif st.session_state.pause_requested:
        st.session_state.update(paused=True, pause_requested=False)
        st.rerun()
    else:
        st.session_state.current_speaker = other_spk
        st.rerun()
