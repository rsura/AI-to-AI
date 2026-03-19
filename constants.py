"""
constants.py — Shared constants for the 2-AIs-Talking app.
"""

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
