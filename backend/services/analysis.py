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


# --------------------------------------------------------------------------- #
# Heuristic planner (offline fallback)
# --------------------------------------------------------------------------- #
def heuristic_plan(question: str, df: pd.DataFrame) -> Dict[str, Any]:
    q = question.lower()
    cls = classify_columns(df)
    numeric, datetime, categorical = cls["numeric"], cls["datetime"], cls["categorical"]

    plan: Dict[str, Any] = {
        "operation": "table", "x": None, "y": None, "agg": "sum",
        "period": None, "top_n": None, "ascending": False, "chart": "table",
    }

    metric = _pick_metric(question, numeric)

    # Time-series / trend
    if datetime and any(k in q for k in ("trend", "over time", "monthly", "month",
                                         "daily", "weekly", "yearly", "by year",
                                         "time series", "timeseries")):
        plan.update(operation="timeseries", x=datetime[0], y=metric,
                    agg="sum", period=_pick_period(q), chart="line")
        return plan

    # Correlation / relationship
    if any(k in q for k in ("correlation", "correlate", "relationship", "related")):
        plan.update(operation="correlation", chart="table")
        return plan

    # Distribution / counts
    if any(k in q for k in ("distribution", "count of", "how many", "frequency",
                            "value counts", "breakdown")):
        cat = _match_column(question, categorical) or (categorical[0] if categorical else None)
        if cat:
            plan.update(operation="value_counts", x=cat, chart="bar")
            return plan

    # Describe / summary stats
    if any(k in q for k in ("describe", "summary", "statistics", "stats", "overview")):
        plan.update(operation="describe", chart="table")
        return plan

    # Aggregate by category ("by region", "per product", "top products")
    cat = _match_column(question, categorical)
    wants_top = any(k in q for k in ("top", "most", "highest", "best", "largest"))
    wants_bottom = any(k in q for k in ("least", "lowest", "worst", "smallest", "bottom"))
    if cat or wants_top or wants_bottom:
        cat = cat or (categorical[0] if categorical else None)
        if cat and metric:
            top_n = _extract_int(q) or (10 if (wants_top or wants_bottom) else None)
            share = any(k in q for k in ("share", "proportion", "percentage", "percent", "split"))
            n_cats = df[cat].nunique()
            chart = "pie" if (share and n_cats <= 8) else "bar"
            plan.update(operation="aggregate", x=cat, y=metric, agg=_pick_agg(q),
                        top_n=top_n, ascending=wants_bottom, chart=chart)
            return plan

    # Fallbacks
    if datetime and metric:
        plan.update(operation="timeseries", x=datetime[0], y=metric, agg="sum",
                    period=_pick_period(q), chart="line")
    elif categorical and metric:
        plan.update(operation="aggregate", x=categorical[0], y=metric,
                    agg=_pick_agg(q), top_n=10, chart="bar")
    elif numeric:
        plan.update(operation="describe", chart="table")
    return plan


def _pick_agg(q: str) -> str:
    if any(k in q for k in ("average", "avg", "mean")):
        return "mean"
    if any(k in q for k in ("count", "number of", "how many")):
        return "count"
    if "max" in q or "maximum" in q or "highest" in q:
        return "max"
    if "min" in q or "minimum" in q or "lowest" in q:
        return "min"
    return "sum"


def _pick_period(q: str) -> str:
    if "year" in q:
        return "Y"
    if "quarter" in q:
        return "Q"
    if "week" in q:
        return "W"
    if "day" in q or "daily" in q:
        return "D"
    return "M"


def _extract_int(q: str) -> Optional[int]:
    m = re.search(r"\btop\s+(\d+)\b", q) or re.search(r"\b(\d+)\b", q)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Plan validation
# --------------------------------------------------------------------------- #
def sanitize_plan(plan: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    cols = {str(c) for c in df.columns}
    op = plan.get("operation")
    if op not in {"aggregate", "timeseries", "value_counts", "describe", "correlation", "table"}:
        op = "table"
    x = plan.get("x") if plan.get("x") in cols else None
    y = plan.get("y") if plan.get("y") in cols else None
    agg = plan.get("agg") if plan.get("agg") in AGGS else "sum"
    chart = plan.get("chart") if plan.get("chart") in CHARTS else "table"
    top_n = plan.get("top_n")
    top_n = int(top_n) if isinstance(top_n, (int, float)) and top_n else None
    period = plan.get("period") if plan.get("period") in {"D", "W", "M", "Q", "Y"} else None
    return {"operation": op, "x": x, "y": y, "agg": agg, "period": period,
            "top_n": top_n, "ascending": bool(plan.get("ascending", False)), "chart": chart}


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def execute(plan: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    op = plan["operation"]
    if op == "aggregate":
        return _exec_aggregate(plan, df)
    if op == "timeseries":
        return _exec_timeseries(plan, df)
    if op == "value_counts":
        return _exec_value_counts(plan, df)
    if op == "correlation":
        return _exec_correlation(df)
    if op == "describe":
        return _exec_describe(df)
    return _exec_table(df)


def _table(columns: List[str], rows: List[List[Any]]) -> Dict[str, Any]:
    return {"columns": columns,
            "rows": [[_json_safe(v) for v in r] for r in rows]}


def _exec_aggregate(plan: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    x, y, agg = plan["x"], plan["y"], plan["agg"]
    if not x:
        return _exec_describe(df)
    if agg == "count" or not y:
        grouped = df.groupby(x, dropna=True).size().rename("count")
        y_label = "count"
    else:
        grouped = df.groupby(x, dropna=True)[y].agg(agg)
        y_label = f"{agg}({y})"
    grouped = grouped.sort_values(ascending=plan["ascending"])
    if plan["top_n"]:
        grouped = grouped.head(plan["top_n"])
    labels = [str(i) for i in grouped.index.tolist()]
    values = [float(v) for v in grouped.values]
    return {
        "table": _table([str(x), y_label], list(zip(labels, values))),
        "chart": _chart(plan["chart"], labels, values, y_label),
        "summary": {"operation": "aggregate", "group_by": x, "metric": y_label,
                    "items": [{"label": l, "value": round(v, 4)} for l, v in zip(labels, values)]},
    }
