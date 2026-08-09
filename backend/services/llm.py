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
