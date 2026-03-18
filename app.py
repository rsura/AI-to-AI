"""
app.py — AI-to-AI: two Ollama models conversing with each other.

Conversation loop (driven by st.rerun):
  1. Render completed messages from display history.
  2. If running and not paused, stream the current speaker's response.
  3. Append the completed response to both models' private histories.
  4. Either pause (if requested) or flip the speaker and rerun.

Each model holds its own Ollama message list from its own perspective:
  - Its own replies  → role: "assistant"
  - The other's replies → role: "user"

Both models are kept in RAM between turns via keep_alive=-1.
"""

import streamlit as st
import ollama
import psutil

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Visual identifiers for AI 1 and AI 2 throughout the UI.
AVATARS = ["🤖", "🦾"]

# Default session-state values; applied once on first page load.
DEFAULTS: dict = {
    "running": False,           # whether the conversation loop is active
    "paused": False,            # whether we are currently in the butt-in pause state
    "pause_requested": False,   # flag set by the sidebar toggle; honoured after current turn
    "current_speaker": 0,       # index into AVATARS / model list (0 = AI1, 1 = AI2)
    "messages_ai1": [],         # Ollama message dicts from AI1's perspective
    "messages_ai2": [],         # Ollama message dicts from AI2's perspective
    "display_messages": [],     # unified list of {speaker, model, content} for rendering
    "model1": None,             # model name string frozen at conversation start
    "model2": None,
    "intro_enabled": False,
    "init_message": "",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_bytes(n: int) -> str:
    """Convert a byte count to a human-readable string (e.g. 4.7 GB)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


@st.cache_data(ttl=30)
def get_models() -> list[dict]:
    """
    Fetch the list of locally available Ollama models.

    Returns a list of dicts with keys:
      - name:  raw model identifier passed to ollama.chat()
      - label: display string shown in the dropdown, e.g. "llama3.2  (2.0 GB · 3.2B)"

    Results are cached for 30 seconds so repeated sidebar reruns don't hammer
    the Ollama API on every keystroke.
    """
    try:
        resp = ollama.list()
        models = []
        for m in resp.models:
            name = m.model or m.name
            size_str = _fmt_bytes(m.size) if m.size else "?"
            param_str = f" · {m.details.parameter_size}" if (m.details and m.details.parameter_size) else ""
            models.append({
                "name": name,
                "label": f"{name}  ({size_str}{param_str})",
            })
        return models
    except Exception as exc:
        st.error(f"Could not reach Ollama: {exc}")
        return []


def get_system_ram() -> str:
    """Return a human-readable string of available and total system RAM."""
    vm = psutil.virtual_memory()
    return f"{_fmt_bytes(vm.available)} free / {_fmt_bytes(vm.total)} total"


def build_system_prompt(own_name: str, other_name: str) -> str:
    """
    Build the AI Introduction system prompt for a model.

    Tells the model its own identity, the name of its conversation partner,
    and that the partner is a fellow AI — not a human.
    """
    return (
        f"You are {own_name}, an AI. You are having a direct, thoughtful conversation "
        f"with another AI named {other_name}. You are fully aware that you are speaking "
        f"to a fellow AI — not a human. Be genuine, curious, and intellectually engaged. "
        f"Keep your responses concise (2–4 sentences) so the conversation flows naturally."
    )


def stream_response(model: str, messages: list[dict]):
    """
    Generator that streams tokens from an Ollama chat call.

    Passes keep_alive=-1 so the model is never unloaded between turns.
    num_predict caps responses at ~300 tokens to keep turns conversational.
    """
    for chunk in ollama.chat(
        model=model,
        messages=messages,
        stream=True,
        keep_alive=-1,
        options={"num_predict": 300},
    ):
        yield chunk.message.content or ""


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
# Sidebar — configuration & conversation controls
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")
    st.caption(f"💾 System RAM: {get_system_ram()}")
    st.divider()

    models = get_models()
    model_names  = [m["name"]  for m in models]
    model_labels = [m["label"] for m in models]

    if not models:
        st.warning("No Ollama models found. Make sure Ollama is running.")

    # Lock the configuration widgets once a conversation is in progress.
    locked = st.session_state.running

    # ── Model selection ──────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**AI 1**")
        idx1 = st.selectbox(
            "AI 1 model",
            range(len(model_labels)),
            format_func=lambda i: model_labels[i],
            index=0 if models else None,
            disabled=locked,
            label_visibility="collapsed",
            key="sel_model1",
        )
    with col2:
        st.markdown("**AI 2**")
        # Default AI 2 to the second available model when possible.
        default_idx2 = min(1, len(models) - 1) if len(models) > 1 else 0
        idx2 = st.selectbox(
            "AI 2 model",
            range(len(model_labels)),
            format_func=lambda i: model_labels[i],
            index=default_idx2,
            disabled=locked,
            label_visibility="collapsed",
            key="sel_model2",
        )

    st.divider()

    # ── Conversation options ─────────────────────────────────────────────────
    intro_enabled = st.checkbox(
        "AI Introduction mode",
        value=False,
        disabled=locked,
        help=(
            "Injects a system prompt into each model's context telling it its own name, "
            "the other model's name, and that it is speaking to a fellow AI — not a human."
        ),
        key="cb_intro",
    )

    init_message = st.text_area(
        "Opening message (leave blank for a default starter)",
        value="",
        disabled=locked,
        placeholder="e.g. Let's discuss the nature of consciousness…",
        key="ta_init",
    )

    st.divider()

    # ── Start / pause / stop controls ───────────────────────────────────────
    if not st.session_state.running:
        if st.button("▶ Start conversation", use_container_width=True, disabled=not models):
            sel_m1 = model_names[idx1]
            sel_m2 = model_names[idx2]
            opener = init_message.strip() or "Hello! Let's start a conversation."

            # Initialise each model's message history.
            m1: list[dict] = []
            m2: list[dict] = []

            if intro_enabled:
                m1.append({"role": "system", "content": build_system_prompt(sel_m1, sel_m2)})
                m2.append({"role": "system", "content": build_system_prompt(sel_m2, sel_m1)})

            # The opening message is posed to AI1 as the first user turn.
            m1.append({"role": "user", "content": opener})

            st.session_state.update({
                "running": True,
                "paused": False,
                "pause_requested": False,
                "current_speaker": 0,
                "messages_ai1": m1,
                "messages_ai2": m2,
                "display_messages": [{"speaker": -1, "model": "Prompt", "content": opener}],
                "model1": sel_m1,
                "model2": sel_m2,
                "intro_enabled": intro_enabled,
                "init_message": opener,
            })
            st.rerun()

    else:
        # The pause toggle sets pause_requested; it takes effect after the
        # current streaming turn finishes (not mid-generation).
        st.toggle(
            "Pause after this turn",
            value=st.session_state.pause_requested,
            key="tgl_pause",
            on_change=lambda: st.session_state.update(
                pause_requested=st.session_state.tgl_pause
            ),
        )
        if st.button("⏹ Stop conversation", use_container_width=True):
            st.session_state.update(running=False, paused=False, pause_requested=False)
            st.rerun()

    st.divider()
    if st.button("🔄 Reset everything", use_container_width=True):
        st.session_state.update(DEFAULTS)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Chat display — render completed messages from history
# ─────────────────────────────────────────────────────────────────────────────

def render_history():
    """Render all messages in display_messages using Streamlit's chat UI."""
    for msg in st.session_state.display_messages:
        spk = msg["speaker"]
        if spk == -1:
            # The opening prompt injected at conversation start.
            with st.chat_message("user", avatar="📝"):
                st.markdown(f"**Opening prompt:** {msg['content']}")
        else:
            with st.chat_message("assistant", avatar=AVATARS[spk]):
                st.markdown(f"**{msg['model']}**")
                st.write(msg["content"])


render_history()


# ─────────────────────────────────────────────────────────────────────────────
# Butt-in UI — shown only while paused
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.paused and st.session_state.running:
    st.info("⏸ Conversation paused. Inject a message as either AI, then resume.", icon="✋")

    m1_name = st.session_state.model1 or "AI 1"
    m2_name = st.session_state.model2 or "AI 2"

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
        with st.expander(f"💬 Butt in as {AVATARS[0]} {m1_name}", expanded=True):
            butin1_text = st.text_area("Your message as AI 1", key="butin1_text", height=100)
            if st.button(f"Send as {m1_name}", key="butin1_send", use_container_width=True):
                if butin1_text.strip():
                    inject_as(0, butin1_text.strip())
                    st.rerun()

    with col_b:
        with st.expander(f"💬 Butt in as {AVATARS[1]} {m2_name}", expanded=True):
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

    # Stream the response live; st.write_stream returns the full assembled string.
    with st.chat_message("assistant", avatar=AVATARS[spk]):
        st.markdown(f"**{model}**")
        full_response: str = st.write_stream(stream_response(model, messages))

    # Append to the current speaker's history as their own reply.
    messages.append({"role": "assistant", "content": full_response})

    # Append to the other speaker's history as an incoming user message.
    other_msgs = [st.session_state.messages_ai1, st.session_state.messages_ai2][other_spk]
    other_msgs.append({"role": "user", "content": full_response})

    # Persist to the display history so it survives the upcoming rerun.
    st.session_state.display_messages.append(
        {"speaker": spk, "model": model, "content": full_response}
    )

    # Honour a pending pause request, or immediately advance to the next turn.
    if st.session_state.pause_requested:
        st.session_state.update(paused=True, pause_requested=False)
        st.rerun()
    else:
        st.session_state.current_speaker = other_spk
        st.rerun()
