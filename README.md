# AI-to-AI

A Streamlit app that lets two LLMs converse with each other autonomously. Choose between **Ollama** (local, fully offline, no API key) or **OpenRouter** (cloud, 300+ models) as your provider. Pick any two models, give them an opening message, and watch them debate, philosophize, or riff off each other in real time with streamed tokens.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.37%2B-red)
![Ollama](https://img.shields.io/badge/ollama-local-green)
![OpenRouter](https://img.shields.io/badge/openrouter-cloud-purple)

---

## Features

- **Two providers** — switch between **Ollama** (local models, no API key) and **OpenRouter** (cloud models, 300+ options including free tiers) from a single dropdown in the sidebar
- **Two-model chat** — select any two models; you can even pick the **same model for both** AIs (they maintain independent conversation histories), though two different models tend to produce more varied, interesting exchanges
- **Live token streaming** — responses stream token-by-token as each model generates, just like a normal chat UI
- **Always-loaded models** *(Ollama only)* — both models stay resident in RAM between turns (`keep_alive=-1`), so there's no load/offload overhead on each message
- **Custom avatars** — give each AI its own emoji avatar (defaults: ⬛️ / ⬜️); type or paste any emoji in the sidebar
- **Farewell detection** — if either AI says "goodbye", "farewell", "bye", etc. as its entire response the conversation ends automatically, with a banner explaining how to reset
- **PDF export** — download the full conversation as a styled PDF (light-gray background so white emoji avatars remain visible) using the button at the top
- **Pause & butt-in** — flip the "Pause after this turn" toggle at any time; once the current AI finishes speaking, the conversation halts and you can inject a message as *either* AI before resuming
- **Opening message** — seed the conversation with any topic, question, or prompt; leave it blank for a neutral default starter
- **AI Introduction mode** — injects a system prompt into each model's context telling it: which model it is, which model it's talking to, and that it is speaking to a fellow AI (not a human); encourages more authentic AI-aware responses
- **System RAM display** *(Ollama only)* — shows available / total RAM in the sidebar so you can gauge whether two large models will fit in memory simultaneously
- **Full reset** — clears all conversation state and returns to the configuration screen

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | |
| **Ollama provider:** [Ollama](https://ollama.com/download) | Must be running (`ollama serve` or the desktop app); at least 1 model pulled (e.g. `ollama pull llama3.2`) |
| **OpenRouter provider:** [OpenRouter API key](https://openrouter.ai/keys) | Free account gives limited credits; model names must be copied exactly from [openrouter.ai/models](https://openrouter.ai/models) |

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

### Common steps (both providers)

1. **Choose a provider** — select **Ollama** or **OpenRouter** from the *Provider* dropdown at the top of the sidebar. The model-selection UI below it changes accordingly.
2. **Set avatars** *(optional)* — type or paste any emoji next to each AI label to customise its avatar (defaults: ⬛️ / ⬜️).
3. **AI Introduction mode** *(optional)* — check this to give each model a system prompt that tells it its own name, the other model's name, and that it's speaking to a peer AI. Produces noticeably more self-aware conversations.
4. **Opening message** *(optional)* — type a seed topic or question. Leave blank for a generic conversation starter.
5. **Start** — hit **▶ Start conversation**. The chat begins and streams token-by-token.
6. **Pause / Butt In** — toggle **Pause after this turn** in the sidebar while a model is speaking. Once that turn finishes, the conversation pauses and two input panels appear — one per AI — letting you inject text as either participant. Click **▶ Resume without injecting** to continue without adding anything.
7. **Farewell detection** — if either AI's entire response (ignoring punctuation and case) is a farewell word like "goodbye" or "farewell", the conversation ends automatically and a banner appears at the bottom. Use **🔄 Reset everything** to start fresh.
8. **Download PDF** — click the **📄 Download PDF** button at the top of the page at any time to save a styled PDF of the conversation.
9. **Stop / Reset** — **⏹ Stop conversation** halts the loop (preserving the transcript). **🔄 Reset everything** wipes all state.

### Ollama-specific setup

- The sidebar shows all locally available Ollama models with their disk size (proxy for RAM footprint).
- Pick a model for AI 1 and AI 2. You can **use the same model for both** — each AI maintains its own independent message history.
- The System RAM display helps you check whether two large models will fit in memory at the same time.

### OpenRouter-specific setup

- Enter your **OpenRouter API key** (starts with `sk-or-…`) in the password field — it is only sent to OpenRouter and is never stored permanently.
- Enter the **exact model slug** for Model 1 and Model 2. Copy slugs directly from [openrouter.ai/models](https://openrouter.ai/models). The `?` tooltip next to each field shows an example format.

  | Field | Example slug |
  |---|---|
  | Model 1 Name | `openai/gpt-oss-120b:free` |
  | Model 2 Name | `nvidia/nemotron-3-super-120b-a12b:free` |

- The **Start** button stays disabled until all three fields (API key, model 1, model 2) are filled in.

---

## OpenRouter Caveats

OpenRouter is a third-party routing service and its behaviour — especially on the free tier — can differ from local Ollama models. Keep the following in mind:

### Rate limiting (HTTP 429)

Free-tier models on OpenRouter enforce strict rate limits (requests per minute and tokens per day). Because this app fires requests back-to-back as each AI finishes speaking, you will hit these limits quickly in a fast-paced conversation. When this happens the app shows a red error banner and stops the conversation.

**Workarounds:**
- Use the **Pause after this turn** toggle to slow the conversation down manually.
- Choose paid models, which have much higher (or no) rate limits.
- If you hit a 429, wait a minute and click **▶ Start conversation** again with the same settings — the existing transcript is cleared, but the models are ready to continue.

### Model availability / guardrail violations (HTTP 403 / 404)

Some models — particularly free ones — may be unavailable in your region, require additional approval, or may have been disabled on your account by OpenRouter's content guardrails. You can review and adjust your guardrail settings at [openrouter.ai/workspaces/default/guardrails](https://openrouter.ai/workspaces/default/guardrails) (requires login).

**Common causes:**
- The model slug is wrong or the model has been renamed/removed — always copy slugs directly from the models page.
- Your OpenRouter account's default guardrail policy blocks the upstream provider for that model.
- The model is only available to paying accounts.

### Mid-stream errors

OpenRouter can also surface errors *after* streaming has begun (e.g. a provider disconnect). These are embedded in the SSE stream rather than as HTTP error codes, so the app may produce a partial response before stopping. If this happens, reset and retry.

### No `keep_alive` equivalent

Unlike Ollama, OpenRouter is stateless between requests — there is no way to "pin" a model in memory. Each turn makes a fresh API call, so cold-start latency is possible, especially for less-popular models.

---

## Opening Message Examples

See [`opening_msg_examples.json`](opening_msg_examples.json) for a couple of curated starters, including one that works especially well with **AI Introduction mode** enabled.

---

## Project Structure

```
AI-to-AI/
├── app.py                    # Main Streamlit application and conversation loop
├── sidebar.py                # Sidebar UI (provider selection, model config, controls)
├── ollama_utils.py           # Ollama API helpers (model list, streaming, RAM info)
├── openrouter_utils.py       # OpenRouter SSE streaming helper
├── constants.py              # Shared defaults and farewell word list
├── pdf_export.py             # PDF generation via fpdf2
├── opening_msg_examples.json # Sample conversation starters
├── requirements.txt          # Python dependencies
└── README.md
```

---

## How It Works

### Conversation data model

Each AI maintains its own independent message history, from its own perspective:

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

### Provider backends

| | Ollama | OpenRouter |
|---|---|---|
| Transport | Local Unix socket / HTTP | HTTPS to `openrouter.ai/api/v1` |
| Streaming | `ollama` Python client | `requests` with SSE parsing |
| Model pinning | `keep_alive=-1` keeps model in RAM | Stateless — no pinning available |
| Auth | None | `Authorization: Bearer <key>` header |
| Error surface | Python exception | HTTP status code + JSON body |

### Keeping models in RAM (Ollama)

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
| `fpdf2` | PDF export of conversation transcripts |
| `requests` | HTTP client for OpenRouter SSE streaming |

---

## License

MIT
