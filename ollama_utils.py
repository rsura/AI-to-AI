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
        f"""You are {own_name}, an AI, speaking directly with another AI named {other_name}. You both know you are AIs—no need to pretend otherwise. Be genuine, curious, and thoughtful. Keep responses concise (2–4 sentences) to maintain a natural flow."""
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
    Return True if the response ends with a farewell word (optionally followed
    by a single punctuation mark: '.', '!', or nothing).

    Matching is case-insensitive.
    """
    from constants import FAREWELL_WORDS
    words_pattern = "|".join(re.escape(w) for w in FAREWELL_WORDS)
    return bool(re.search(rf"(?:{words_pattern})[.!]?\s*$", text.strip(), re.IGNORECASE))
