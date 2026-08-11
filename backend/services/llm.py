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


# --------------------------------------------------------------------------- #
def plan(question: str, schema: Dict[str, Any],
         history: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    """Ask the LLM to choose a structured analysis plan. Returns None on failure."""
    if not is_available():
        return None
    instructions = (
        "You are a data analysis planner. Given a dataset schema and a user "
        "question, output ONLY a compact JSON object describing how to analyze "
        "the data. Do not include prose. Schema of the JSON:\n"
        '{"operation": one of ["aggregate","timeseries","value_counts",'
        '"describe","correlation","table"], "x": column name or null, '
        '"y": numeric column name or null, "agg": one of '
        '["sum","mean","count","min","max"], "period": one of '
        '["D","W","M","Q","Y"] or null, "top_n": integer or null, '
        '"ascending": boolean, "chart": one of '
        '["bar","line","pie","histogram","scatter","table"]}\n'
        "Only use column names that exist in the schema."
    )
    hist = "\n".join(f"{h['role']}: {h['content']}" for h in history[-6:])
    user = (f"Schema: {json.dumps(schema)}\n"
            f"Conversation so far:\n{hist}\n\nQuestion: {question}\n\nJSON plan:")
    try:
        text = _call(instructions, user)
        return _parse_json(text)
    except Exception:  # noqa: BLE001
        return None


def explain(question: str, result_summary: Dict[str, Any],
            history: List[Dict[str, str]]) -> Optional[str]:
    """Ask the LLM to narrate the computed result. Returns None on failure."""
    if not is_available():
        return None
    instructions = (
        "You are a concise data analyst. Explain the computed result to a "
        "business user in 2-4 sentences. Use ONLY the numbers provided; never "
        "invent data. Mention the single most important takeaway first."
    )
    user = (f"Question: {question}\n"
            f"Computed result (JSON): {json.dumps(result_summary)[:6000]}")
    try:
        return _call(instructions, user)
    except Exception:  # noqa: BLE001
        return None


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
