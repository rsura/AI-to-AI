# AI-to-AI

A Streamlit app that lets two local LLMs converse with each other autonomously — fully offline, no API keys required. Pick any two [Ollama](https://ollama.com) models, give them an opening message, and watch them debate, philosophize, or riff off each other in real time with streamed tokens.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.37%2B-red)
![Ollama](https://img.shields.io/badge/ollama-local-green)

---

## Features

- **Two-model chat** — select any two Ollama models from a dropdown that shows each model's disk size and parameter count
- **Live token streaming** — responses stream token-by-token as each model generates, just like a normal chat UI
- **Always-loaded models** — both models stay resident in RAM between turns (`keep_alive=-1`), so there's no load/offload overhead on each message
- **Pause & butt-in** — flip the "Pause after this turn" toggle at any time; once the current AI finishes speaking, the conversation halts and you can inject a message as *either* AI before resuming
- **Opening message** — seed the conversation with any topic, question, or prompt; leave it blank for a neutral default starter
- **AI Introduction mode** — injects a system prompt into each model's context telling it: which model it is, which model it's talking to, and that it is speaking to a fellow AI (not a human); encourages more authentic AI-aware responses
- **System RAM display** — shows available / total RAM in the sidebar so you can gauge whether two large models will fit in memory simultaneously
- **Full reset** — clears all conversation state and returns to the configuration screen

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | |
| [Ollama](https://ollama.com/download) | Must be running (`ollama serve` or the desktop app) |
| At least 2 models pulled | e.g. `ollama pull llama3.2` then `ollama pull gemma3` |

---

## Installation

```bash
git clone https://github.com/rsura/AI-to-AI.git
cd AI-to-AI
pip install -r requirements.txt
```

---

## Running

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Usage

1. **Select models** — the sidebar shows all locally available Ollama models with their disk size (proxy for RAM footprint). Pick one for AI 1 and one for AI 2.
2. **AI Introduction mode** *(optional)* — check this to give each model a system prompt that tells it its own name, the other model's name, and that it's speaking to a peer AI. Produces noticeably more self-aware conversations.
3. **Opening message** *(optional)* — type a seed topic or question. Leave blank for a generic conversation starter.
4. **Start** — hit **▶ Start conversation**. Both models load into RAM and the chat begins.
5. **Pause / Butt In** — toggle **Pause after this turn** in the sidebar while a model is speaking. Once that turn finishes, the conversation pauses and two input panels appear — one per AI — letting you inject text as either participant. Click **▶ Resume without injecting** to continue without adding anything.
6. **Stop / Reset** — **⏹ Stop conversation** halts the loop (preserving the transcript). **🔄 Reset everything** wipes all state.

---

## Opening Message Examples

See [`opening_msg_examples.json`](opening_msg_examples.json) for a couple of curated starters, including one that works especially well with **AI Introduction mode** enabled.

---

## Project Structure

```
AI-to-AI/
├── app.py                    # Single-file Streamlit application
├── opening_msg_examples.json # Sample conversation starters
├── requirements.txt          # Python dependencies
└── README.md
```

---

## How It Works

### Conversation data model

Each AI maintains its own independent Ollama message history, from its own perspective:

- Its own replies are stored as `role: "assistant"`
- The other AI's replies arrive as `role: "user"`

This means each model sees a coherent back-and-forth dialogue without knowing it is reading another model's output — it simply sees "a user" speaking to it.

### Streaming loop

The conversation is driven by Streamlit's `st.rerun()` cycle:

```
rerun starts
  → render all completed messages from display history
  → if running and not paused:
      stream current speaker's response via st.write_stream()
      append completed response to both AIs' histories
      if pause was requested → enter paused state
      else → flip current_speaker and st.rerun()
```

### Keeping models in RAM

Every `ollama.chat()` call includes `keep_alive=-1`, which instructs Ollama to keep the model loaded in memory indefinitely. Without this, Ollama unloads a model after 5 minutes of inactivity — causing a cold-load delay whenever the other model finishes its turn.

### Butt-in injection

When you inject a message as AI N, it is written into the conversation histories as:
- `role: "assistant"` in AI N's history (so AI N "remembers" saying it)
- `role: "user"` in the other AI's history (so the other AI sees it as a message received)

The turn counter then advances to the *other* AI, so it responds to your injected message naturally.

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI and streaming display |
| `ollama` | Python client for the local Ollama API |
| `psutil` | System RAM info |

---

## License

MIT
