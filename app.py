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

import os
import re
from datetime import datetime

import ollama
import psutil
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Words/phrases that indicate an AI is ending the conversation.
# Checked against the full response (punctuation-stripped, lowercased).
FAREWELL_WORDS = {
    "bye", "goodbye", "farewell", "ciao", "adieu", "ttyl", "toodles",
    "so long", "see ya", "see you", "take care", "au revoir", "later",
    "bye bye", "cheerio",
}

# Default session-state values; applied once on first page load.
DEFAULTS: dict = {
    "running": False,           # whether the conversation loop is active
    "paused": False,            # whether we are currently in the butt-in pause state
    "pause_requested": False,   # flag set by the sidebar toggle; honoured after current turn
    "current_speaker": 0,       # index 0 = AI1, 1 = AI2
    "messages_ai1": [],         # Ollama message dicts from AI1's perspective
    "messages_ai2": [],         # Ollama message dicts from AI2's perspective
    "display_messages": [],     # unified list of {speaker, model, content} for rendering
    "model1": None,             # model name string frozen at conversation start
    "model2": None,
    "intro_enabled": False,
    "init_message": "",
    "inp_avatar1": "⬛️",        # emoji avatar for AI 1 (widget key)
    "inp_avatar2": "⬜️",        # emoji avatar for AI 2 (widget key)
    "conversation_ended": False,  # set True when a farewell is detected
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
def get_models() -> tuple[list[dict], str | None]:
    """
    Fetch the list of locally available Ollama models.

    Returns (models, error_message) where:
      - models is a list of dicts with 'name' and 'label' keys
      - error_message is None on success, or a string describing the connection error

    Results are cached for 30 seconds.
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
        return models, None
    except Exception as exc:
        return [], str(exc)


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


def is_farewell(text: str) -> bool:
    """
    Return True if the response is a single farewell word/phrase.

    Strips punctuation, collapses whitespace, and lowercases before checking
    against the FAREWELL_WORDS set.
    """
    cleaned = re.sub(r"[^\w\s]", "", text.strip()).lower().strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned in FAREWELL_WORDS


# ── PDF formatting helpers ────────────────────────────────────────────────────

def _ordinal(n: int) -> str:
    """Return the ordinal string for an integer (e.g. 1 → '1st', 17 → '17th')."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}" + ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]


def _format_export_date(dt: datetime) -> str:
    """Return a human-readable date string, e.g. 'March 17th, 2026'."""
    return f"{dt.strftime('%B')} {_ordinal(dt.day)}, {dt.year}"


def _pdf_model_display(raw: str) -> tuple[str, str]:
    """
    Convert an Ollama model identifier to a (display_name, param_desc) pair.

    Examples:
      "llama3.2:3b"  → ("Llama 3.2",  "3 billion parameters")
      "gemma3:1b"    → ("Gemma 3",    "1 billion parameters")
      "mistral:7b"   → ("Mistral",    "7 billion parameters")
      "phi3:mini"    → ("Phi 3",      "mini")
      "qwen2.5:14b"  → ("Qwen 2.5",  "14 billion parameters")
    """
    base, _, tag = raw.partition(":")
    # Insert a space at letter→digit boundaries: "llama3.2" → "llama 3.2"
    spaced = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", base)
    spaced = spaced.replace("-", " ").replace("_", " ")
    display = " ".join(w.capitalize() for w in spaced.split())

    param_desc = ""
    if tag:
        m = re.match(r"^(\d+(?:\.\d+)?)b$", tag.lower())
        if m:
            param_desc = f"{m.group(1)} billion parameters"
        else:
            param_desc = tag

    return display, param_desc


def _pdf_speaker_label(raw: str) -> str:
    """
    Convert an Ollama model string (as stored in display_messages["model"])
    to a clean speaker label for the PDF — no emoji, no raw tag.

    Examples:
      "llama3.2:3b"       → "Llama 3.2"
      "gemma3:1b (you)"   → "Gemma 3 (you)"
    """
    suffix = " (you)" if raw.endswith(" (you)") else ""
    name = raw[: -len(" (you)")] if suffix else raw
    display, _ = _pdf_model_display(name)
    return display + suffix


# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf() -> bytes:
    """
    Generate a styled PDF of the current conversation and return as bytes.

    Light gray page background ensures legibility regardless of theme.
    Attempts to load a Unicode-capable system TTF font for international
    characters in AI responses; falls back to Helvetica (Latin-1) otherwise.
    Emoji are intentionally excluded from the PDF since most PDF fonts do not
    render them — the avatars are a web-only feature.
    """
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="fpdf")
            from fpdf import FPDF
            from fpdf.enums import XPos, YPos
    except ImportError:
        return b""

    raw_model1 = st.session_state.get("model1") or "AI 1"
    raw_model2 = st.session_state.get("model2") or "AI 2"
    messages   = st.session_state.get("display_messages", [])

    name1, params1 = _pdf_model_display(raw_model1)
    name2, params2 = _pdf_model_display(raw_model2)

    class ChatPDF(FPDF):
        def header(self):
            # Light gray background on every page so the document looks clean
            # even if the user's PDF viewer defaults to white.
            self.set_fill_color(245, 245, 245)
            self.rect(0, 0, self.w, self.h, "F")

    pdf = ChatPDF()
    pdf.set_auto_page_break(auto=True, margin=22)

    # Try to load a Unicode-capable TTF font for international characters.
    unicode_font = None
    for font_path in [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]:
        if os.path.exists(font_path):
            try:
                pdf.add_font("UniFont", fname=font_path)
                unicode_font = "UniFont"
                break
            except Exception:
                continue

    def safe_text(text: str) -> str:
        if unicode_font:
            return text
        return text.encode("latin-1", errors="replace").decode("latin-1")

    def use_font(size: int, bold: bool = False) -> None:
        if unicode_font:
            pdf.set_font(unicode_font, size=size)
        else:
            pdf.set_font("Helvetica", style="B" if bold else "", size=size)

    pdf.add_page()

    # ── Heading ──────────────────────────────────────────────────────────────

    # Main title: "Conversation between Model A and Model B"
    use_font(18, bold=True)
    pdf.set_text_color(25, 25, 25)
    pdf.cell(
        0, 11, safe_text(f"Conversation between {name1} and {name2}"),
        align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )

    # Export date
    use_font(10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(
        0, 7, safe_text(f"Exported: {_format_export_date(datetime.now())}"),
        align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )

    # Model info line  –  "AI 1: Llama 3.2 (3 billion parameters)  |  AI 2: Gemma 3 (1 billion parameters)"
    ai1_info = f"AI 1: {name1}" + (f" ({params1})" if params1 else "")
    ai2_info = f"AI 2: {name2}" + (f" ({params2})" if params2 else "")
    use_font(9)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(
        0, 6, safe_text(f"{ai1_info}   \u2022   {ai2_info}"),
        align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )

    pdf.ln(7)
    pdf.set_draw_color(190, 190, 190)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(8)

    # ── Messages ─────────────────────────────────────────────────────────────

    for msg in messages:
        spk     = msg["speaker"]
        content = msg.get("content", "")

        if spk == -1:
            # Opening prompt
            use_font(8, bold=True)
            pdf.set_text_color(70, 70, 70)
            pdf.set_fill_color(215, 215, 215)
            pdf.cell(
                0, 6, safe_text("Opening Prompt"),
                fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )
            use_font(9)
            pdf.set_text_color(35, 35, 35)
            pdf.set_fill_color(245, 245, 245)
            pdf.multi_cell(
                0, 6, safe_text(content),
                fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )

        else:
            label      = _pdf_speaker_label(msg.get("model", f"AI {spk + 1}"))
            fill_color = (207, 226, 255) if spk == 0 else (255, 224, 207)

            use_font(8, bold=True)
            pdf.set_text_color(40, 40, 40)
            pdf.set_fill_color(*fill_color)
            pdf.cell(
                0, 6, safe_text(label),
                fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )
            use_font(9)
            pdf.set_text_color(30, 30, 30)
            pdf.set_fill_color(245, 245, 245)
            pdf.multi_cell(
                0, 6, safe_text(content),
                fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )

        pdf.ln(4)

    return bytes(pdf.output())


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

    models, models_error = get_models()
    model_names  = [m["name"]  for m in models]
    model_labels = [m["label"] for m in models]

    if models_error:
        st.error(f"Cannot reach Ollama: {models_error}")
        st.info("Make sure Ollama is running, then refresh:\n```bash\nollama serve\n```", icon="ℹ️")
    elif not models:
        st.error("No Ollama models found!", icon="🚨")
        st.markdown("Pull at least one model to get started:")
        st.code("ollama pull llama3.2", language="bash")
        st.caption(
            "After pulling, refresh this page. "
            "You can use the **same model for both AI 1 and AI 2** — "
            "though two different models produce more diverse responses."
        )

    # Lock the configuration widgets once a conversation is in progress.
    locked = st.session_state.running

    # ── Model selection ──────────────────────────────────────────────────────
    st.markdown(
        "**Model Selection**",
        help=(
            "You can choose the same model for both AI 1 and AI 2 — it will still hold "
            "two independent conversation histories. Using two *different* models tends to "
            "produce more varied and interesting exchanges."
        ),
    )

    # AI 1 — emoji avatar + label on one row, selectbox below
    col_em1, col_lbl1 = st.columns([1, 4])
    with col_em1:
        st.text_input(
            "Avatar 1",
            key="inp_avatar1",
            max_chars=10,
            disabled=locked,
            label_visibility="collapsed",
            help="Type or paste any emoji to use as AI 1's avatar",
        )
    with col_lbl1:
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

    st.write("")  # spacer

    # AI 2 — emoji avatar + label on one row, selectbox below
    col_em2, col_lbl2 = st.columns([1, 4])
    with col_em2:
        st.text_input(
            "Avatar 2",
            key="inp_avatar2",
            max_chars=10,
            disabled=locked,
            label_visibility="collapsed",
            help="Type or paste any emoji to use as AI 2's avatar",
        )
    with col_lbl2:
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
            "the other model's name, and that it is speaking to a fellow AI — not a human. "
            "Works with the same model for both AIs, but two different models tend to produce "
            "more authentic, self-aware exchanges."
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
                "conversation_ended": False,
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
        # Widget-bound keys (inp_avatar1, inp_avatar2) cannot be written after
        # their widgets are instantiated — skip them so the avatars are preserved.
        st.session_state.update({k: v for k, v in DEFAULTS.items() if not k.startswith("inp_")})
        st.rerun()


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
            # The opening prompt injected at conversation start.
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

    # Stream the response live; st.write_stream returns the full assembled string.
    with st.chat_message("assistant", avatar=avatar):
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

    # Check whether this response is a farewell — if so, end the conversation.
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
