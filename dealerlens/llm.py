"""Minimal client for any OpenAI-compatible chat endpoint.

Default: the local MLX server on your Mac (no API key, nothing leaves the machine):
    uv run --with mlx-lm mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8081
Or point DEALERLENS_LLM_URL / DEALERLENS_LLM_MODEL / DEALERLENS_LLM_KEY at Ollama, OpenAI, etc.
"""
import os

import requests

URL = os.getenv("DEALERLENS_LLM_URL", "http://localhost:8081/v1")
MODEL = os.getenv("DEALERLENS_LLM_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
KEY = os.getenv("DEALERLENS_LLM_KEY", "local")


class LLMUnavailable(RuntimeError):
    pass


def chat(messages, max_tokens=700, temperature=0.0, timeout=180):
    try:
        r = requests.post(f"{URL}/chat/completions", timeout=timeout,
                          headers={"Authorization": f"Bearer {KEY}"},
                          json={"model": MODEL, "messages": messages,
                                "max_tokens": max_tokens, "temperature": temperature})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""
    except (requests.RequestException, KeyError, ValueError) as e:
        raise LLMUnavailable(f"LLM endpoint {URL} not reachable or returned an error: {e}") from e


def available():
    try:
        requests.get(f"{URL}/models", timeout=3).raise_for_status()
        return True
    except requests.RequestException:
        return False
