"""
sidebar.py — Sidebar UI for the 2-AIs-Talking app.
"""

import streamlit as st

from constants import DEFAULTS
from ollama_utils import build_system_prompt, get_models, get_system_ram


def render_sidebar() -> None:
    """
    Render the full sidebar: configuration, model selection, and conversation controls.

    Mutates st.session_state directly and calls st.rerun() where needed,
    matching the original inline behaviour.
    """
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

        locked = st.session_state.running

        # ── Model selection ───────────────────────────────────────────────────
        st.markdown(
            "**Model Selection**",
            help=(
                "You can choose the same model for both AI 1 and AI 2 — it will still hold "
                "two independent conversation histories. Using two *different* models tends to "
                "produce more varied and interesting exchanges."
            ),
        )

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

        init_message = st.text_area(
            "Opening message (leave blank for a default starter)",
            value="",
            disabled=locked,
            placeholder="e.g. Let's discuss the nature of consciousness…",
            key="ta_init",
        )

        st.divider()

        # ── Start / pause / stop controls ────────────────────────────────────
        if not st.session_state.running:
            if st.button("▶ Start conversation", use_container_width=True, disabled=not models):
                sel_m1 = model_names[idx1]
                sel_m2 = model_names[idx2]
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
