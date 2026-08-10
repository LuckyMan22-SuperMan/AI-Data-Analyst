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


def _category_extremes(df: pd.DataFrame, cats: List[str], metric: str) -> List[Dict[str, str]]:
    out = []
    for c in cats[:3]:
        if df[c].nunique() < 2 or df[c].nunique() > 50:
            continue
        grouped = df.groupby(c, dropna=True)[metric].sum().sort_values(ascending=False)
        if grouped.empty:
            continue
        top, bottom = grouped.index[0], grouped.index[-1]
        out.append({
            "title": f"Best {c} by {metric}",
            "detail": f"'{top}' leads with {grouped.iloc[0]:,.2f}, while '{bottom}' is lowest at {grouped.iloc[-1]:,.2f}.",
        })
    return out


def _concentration(df: pd.DataFrame, cats: List[str], metric: str) -> List[Dict[str, str]]:
    out = []
    for c in cats[:2]:
        n = df[c].nunique()
        if n < 3 or n > 50:
            continue
        grouped = df.groupby(c, dropna=True)[metric].sum().sort_values(ascending=False)
        total = grouped.sum()
        if total <= 0:
            continue
        top3_share = grouped.head(3).sum() / total * 100
        if top3_share >= 60:
            out.append({
                "title": f"High concentration in {c}",
                "detail": f"The top 3 {c} values account for {top3_share:.1f}% of total {metric}.",
            })
    return out


def _trend(df: pd.DataFrame, date_col: str, metric: str) -> List[Dict[str, str]]:
    tmp = df[[date_col, metric]].dropna()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col]).set_index(date_col).sort_index()
    if tmp.empty:
        return []
    monthly = tmp[metric].resample("ME").sum()
    monthly = monthly[monthly != 0]
    if len(monthly) < 2:
        return []
    change = (monthly.iloc[-1] - monthly.iloc[0]) / abs(monthly.iloc[0]) * 100 if monthly.iloc[0] else 0
    direction = "grew" if change >= 0 else "declined"
    return [{
        "title": f"{metric} trend",
        "detail": f"{metric} {direction} {abs(change):.1f}% overall, from {monthly.iloc[0]:,.0f} to {monthly.iloc[-1]:,.0f} across {len(monthly)} months.",
    }]


def _seasonality(df: pd.DataFrame, date_col: str, metric: str) -> List[Dict[str, str]]:
    tmp = df[[date_col, metric]].dropna()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])
    if tmp.empty:
        return []
    by_month = tmp.groupby(tmp[date_col].dt.month)[metric].mean()
    if by_month.empty or len(by_month) < 3:
        return []
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    best, worst = by_month.idxmax(), by_month.idxmin()
    return [{
        "title": "Seasonality",
        "detail": f"On average, {names[best-1]} is the strongest month for {metric} and {names[worst-1]} the weakest.",
    }]
