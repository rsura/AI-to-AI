"""
openrouter_utils.py — OpenRouter API helpers for the 2-AIs-Talking app.

OpenRouter exposes an OpenAI-compatible chat-completions endpoint with
server-sent-event (SSE) streaming, so we speak plain HTTP via `requests`.
"""

import json

import requests


_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def stream_response_openrouter(api_key: str, model: str, messages: list[dict]):
    """
    Generator that streams tokens from OpenRouter's chat-completions endpoint.

    Yields string chunks as they arrive via SSE.  Raises requests.HTTPError on
    a non-2xx response so the caller can surface the error to the user.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": 300,
    }

    with requests.post(
        _OPENROUTER_URL,
        headers=headers,
        json=payload,
        stream=True,
        timeout=60,
    ) as resp:
        if not resp.ok:
            # Read the body for a detailed error message before raising.
            try:
                body = resp.json()
                detail = (
                    body.get("error", {}).get("message")
                    or body.get("message")
                    or resp.text
                )
            except Exception:
                detail = resp.text or resp.reason
            raise requests.HTTPError(
                f"{resp.status_code} {resp.reason} — {detail}",
                response=resp,
            )
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
