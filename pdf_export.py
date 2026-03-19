"""
pdf_export.py — PDF generation for the 2-AIs-Talking conversation export.
"""

import os
import re
from datetime import datetime

import streamlit as st


# ── Formatting helpers ────────────────────────────────────────────────────────

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
            self.set_fill_color(245, 245, 245)
            self.rect(0, 0, self.w, self.h, "F")

    pdf = ChatPDF()
    pdf.set_auto_page_break(auto=True, margin=22)

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

    # ── Heading ───────────────────────────────────────────────────────────────

    use_font(18, bold=True)
    pdf.set_text_color(25, 25, 25)
    pdf.cell(
        0, 11, safe_text(f"Conversation between {name1} and {name2}"),
        align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )

    use_font(10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(
        0, 7, safe_text(f"Exported: {_format_export_date(datetime.now())}"),
        align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )

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

    # ── Messages ──────────────────────────────────────────────────────────────

    for msg in messages:
        spk     = msg["speaker"]
        content = msg.get("content", "")

        if spk == -1:
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
