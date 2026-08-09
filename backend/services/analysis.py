"""Safe analysis engine.

User questions are mapped to a *constrained* set of operations (aggregate,
timeseries, value_counts, describe, correlation, table). We NEVER execute
arbitrary code or SQL from user input — a plan only names an operation, columns
and parameters, which we run with Pandas. This keeps the AI useful while safe.

Planning is done by the LLM when available (services.llm.plan); otherwise a
heuristic planner inspects keywords + the schema. Execution is identical for
both, and every result carries a Chart.js-ready chart spec.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.services import llm
from backend.services.dataset import _json_safe

AGGS = {"sum", "mean", "count", "min", "max"}
CHARTS = {"bar", "line", "pie", "histogram", "scatter", "table"}


# --------------------------------------------------------------------------- #
# Schema helpers
# --------------------------------------------------------------------------- #
def classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric, datetime, categorical = [], [], []
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_numeric_dtype(col):
            numeric.append(str(c))
        elif pd.api.types.is_datetime64_any_dtype(col):
            datetime.append(str(c))
        else:
            categorical.append(str(c))
    return {"numeric": numeric, "datetime": datetime, "categorical": categorical}


def schema_summary(df: pd.DataFrame) -> Dict[str, Any]:
    cls = classify_columns(df)
    return {
        "columns": [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns],
        "numeric": cls["numeric"],
        "datetime": cls["datetime"],
        "categorical": cls["categorical"],
        "row_count": int(len(df)),
    }


def _match_column(question: str, columns: List[str]) -> Optional[str]:
    q = question.lower()
    # Prefer the longest column name that appears in the question.
    best = None
    for c in sorted(columns, key=len, reverse=True):
        if c.lower() in q:
            best = c
            break
    return best


_METRIC_HINTS = ("revenue", "sales", "amount", "profit", "price", "total",
                 "value", "units", "quantity", "count", "cost")


def _pick_metric(question: str, numeric: List[str]) -> Optional[str]:
    if not numeric:
        return None
    explicit = _match_column(question, numeric)
    if explicit:
        return explicit
    for hint in _METRIC_HINTS:
        for c in numeric:
            if hint in c.lower():
                return c
    return numeric[0]
