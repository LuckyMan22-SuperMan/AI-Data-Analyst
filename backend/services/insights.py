"""Automatic business-insight generation.

Scans the dataset (categoricals x a chosen metric, time trends, data quality,
outliers) and surfaces the most important, human-readable findings. These are
computed deterministically; the LLM (if configured) is only used to polish
wording, never to invent numbers.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backend.services import llm
from backend.services.analysis import _pick_metric, classify_columns


def generate(df: pd.DataFrame) -> Dict[str, Any]:
    cls = classify_columns(df)
    numeric, datetime, categorical = cls["numeric"], cls["datetime"], cls["categorical"]
    metric = _pick_metric("", numeric)

    findings: List[Dict[str, str]] = []

    if metric:
        findings += _category_extremes(df, categorical, metric)
        findings += _concentration(df, categorical, metric)
    if metric and datetime:
        findings += _trend(df, datetime[0], metric)
        findings += _seasonality(df, datetime[0], metric)
    findings += _data_quality(df)
    findings += _outliers(df, numeric)

    if not findings:
        findings.append({"title": "Dataset loaded",
                         "detail": f"{len(df):,} rows and {df.shape[1]} columns are ready to explore."})

    # Optional LLM polish of the phrasing (kept faithful to the numbers).
    polished = _polish(findings)
    return {"metric_used": metric, "insights": polished, "count": len(polished)}
