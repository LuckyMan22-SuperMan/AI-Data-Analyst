"""Optional OpenAI client using the Responses API.

Key-optional: when OPENAI_API_KEY is unset the app still works (planning falls
back to heuristics and explanations fall back to templated text). This keeps
the app runnable offline / on restricted networks.

Env vars:
    OPENAI_API_KEY   enables the LLM
    OPENAI_BASE_URL  default https://api.openai.com/v1
    OPENAI_MODEL     default gpt-4o-mini
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests

_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def is_available() -> bool:
    return bool(_API_KEY)


def info() -> Dict[str, Any]:
    return {"available": is_available(), "model": _MODEL if is_available() else None}


def _call(instructions: str, user_input: str, timeout: int = 40) -> str:
    """Call the OpenAI Responses API and return the output text."""
    if not is_available():
        raise RuntimeError("LLM not configured (set OPENAI_API_KEY).")
    resp = requests.post(
        f"{_BASE_URL}/responses",
        headers={"Authorization": f"Bearer {_API_KEY}",
                 "Content-Type": "application/json"},
        json={"model": _MODEL, "instructions": instructions, "input": user_input},
        timeout=timeout,
    )
    resp.raise_for_status()
    return _extract_text(resp.json())


def _extract_text(payload: Dict[str, Any]) -> str:
    """Robustly pull text out of a Responses API payload."""
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"].strip()
    parts: List[str] = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") in ("output_text", "text") and content.get("text"):
                parts.append(content["text"])
    return "\n".join(parts).strip()
