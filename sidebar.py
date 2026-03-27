"""
sidebar.py — Sidebar UI for the 2-AIs-Talking app.
"""

import json
import pathlib
import streamlit as st

from constants import DEFAULTS
from ollama_utils import build_system_prompt, get_models, get_system_ram

_EXAMPLES_PATH = pathlib.Path(__file__).with_name("opening_msg_examples.json")


def _load_examples() -> list[dict]:
    try:
        return json.loads(_EXAMPLES_PATH.read_text())["opening_msg_examples"]
    except Exception:
        return []


@st.dialog("💡 Example prompts")
def _show_examples_dialog() -> None:
    examples = _load_examples()
    if not examples:
        st.caption("No examples found.")
        return
    locked = st.session_state.get("running", False)
    with st.container(height=260, width="content"):
        for ex in examples:
            intro = ex["AI Introduction Mode"]
            msg = ex["Opening Message"]
            label = f"{'🤖 ' if intro else ''}{msg[:50]}{'…' if len(msg) > 50 else ''}"
            if st.button(label, use_container_width=True, disabled=locked):
                st.session_state["_pending_msg"] = msg
                st.session_state["_pending_intro"] = intro
                st.rerun()


def render_sidebar() -> None:
    """
    Render the full sidebar: provider selection, configuration, model selection,
    and conversation controls.

    Mutates st.session_state directly and calls st.rerun() where needed,
    matching the original inline behaviour.
    """
    # Apply a pending example selection before any widget is instantiated,
    # so that setting cb_intro / ta_init doesn't conflict with widget keys.
    if "_pending_msg" in st.session_state:
        st.session_state["ta_init"] = st.session_state.pop("_pending_msg")
        st.session_state["cb_intro"] = st.session_state.pop("_pending_intro")

    with st.sidebar:
        st.header("Configuration")
        st.divider()

        locked = st.session_state.running

        # ── Provider selection ────────────────────────────────────────────────
        provider = st.selectbox(
            "Provider",
            ["Ollama", "OpenRouter"],
            key="sel_provider",
            disabled=locked,
        )

        st.divider()

        # ── Model selection ───────────────────────────────────────────────────
        st.markdown(
            "**Model Selection**",
            help=(
                "You can choose the same model for both AI 1 and AI 2 — it will still hold "
                "two independent conversation histories. Using two *different* models tends to "
                "produce more varied and interesting exchanges."
            ),
        )

        # Avatar inputs — always shown regardless of provider
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

        # Initialise variables so both branches of the start-button logic can
        # reference them without worrying about which branch ran.
        models: list[dict] = []
        model_names: list[str] = []
        model_labels: list[str] = []
        idx1 = idx2 = 0
        or_api_key = or_model1 = or_model2 = ""

        st.write("")  # spacer

        if provider == "Ollama":
            # ── Ollama: show RAM info + model dropdowns ───────────────────────
            st.caption(f"💾 System RAM: {get_system_ram()}")

            models, models_error = get_models()
            model_names = [m["name"] for m in models]
            model_labels = [m["label"] for m in models]

            if models_error:
                st.error(f"Cannot reach Ollama: {models_error}")
                st.info(
                    "Make sure Ollama is running, then refresh:\n```bash\nollama serve\n```",
                    icon="ℹ️",
                )
            elif not models:
                st.error("No Ollama models found!", icon="🚨")
                st.markdown("Pull at least one model to get started:")
                st.code("ollama pull llama3.2", language="bash")
                st.caption(
                    "After pulling, refresh this page. "
                    "You can use the **same model for both AI 1 and AI 2** — "
                    "though two different models produce more diverse responses."
                )

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

        else:
            # ── OpenRouter: three stacked text inputs ─────────────────────────
            or_api_key = st.text_input(
                "OpenRouter API Key",
                type="password",
                disabled=locked,
                key="inp_or_api_key",
                placeholder="sk-or-…",
            )
            or_model1 = st.text_input(
                "Model 1 Name",
                disabled=locked,
                key="inp_or_model1",
                placeholder="meta-llama/llama-3.3-70b-instruct:free",
                help="Name will look something like this: 'meta-llama/llama-3.3-70b-instruct:free'",
            )
            or_model2 = st.text_input(
                "Model 2 Name",
                disabled=locked,
                key="inp_or_model2",
                placeholder="qwen/qwen3-next-80b-a3b-instruct:free",
                help="Name will look something like this: 'qwen/qwen3-next-80b-a3b-instruct:free'",
            )

        st.divider()

        # ── Conversation options ──────────────────────────────────────────────
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

        col_lbl, col_pop = st.columns([3, 1])
        with col_lbl:
            st.caption("Opening message (leave blank for a default starter)")
        with col_pop:
            if st.button("💡 Examples", use_container_width=True, disabled=locked):
                _show_examples_dialog()

        init_message = st.text_area(
            "Opening message",
            value="",
            disabled=locked,
            placeholder="e.g. Let's discuss the nature of consciousness…",
            key="ta_init",
            label_visibility="collapsed",
        )

        st.divider()

        # ── Start / pause / stop controls ────────────────────────────────────
        if not st.session_state.running:
            if provider == "Ollama":
                start_disabled = not models
            else:
                start_disabled = not (
                    or_api_key.strip() and or_model1.strip() and or_model2.strip()
                )

            if st.button(
                "▶ Start conversation",
                use_container_width=True,
                disabled=start_disabled,
            ):
                if provider == "Ollama":
                    sel_m1 = model_names[idx1]
                    sel_m2 = model_names[idx2]
                    api_key = ""
                else:
                    sel_m1 = or_model1.strip()
                    sel_m2 = or_model2.strip()
                    api_key = or_api_key.strip()

                opener = init_message.strip() or "Hello! Let's start a conversation."

                m1: list[dict] = []
                m2: list[dict] = []

                if intro_enabled:
                    m1.append({"role": "system", "content": build_system_prompt(sel_m1, sel_m2)})
                    m2.append({"role": "system", "content": build_system_prompt(sel_m2, sel_m1)})

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
                    "provider": provider,
                    "openrouter_api_key": api_key,
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
