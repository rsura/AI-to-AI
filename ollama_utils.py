"""
ollama_utils.py — Ollama API helpers and miscellaneous utilities.
"""

import re

import ollama
import psutil
import streamlit as st


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
        f"Additionally, if you believe the conversation is over, you should simply say 'Goodbye' and end the conversation."
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
    from constants import FAREWELL_WORDS
    cleaned = re.sub(r"[^\w\s]", "", text.strip()).lower().strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned in FAREWELL_WORDS
