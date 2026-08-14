"""Data-cleaning issue detection.

Detects missing values, duplicate rows, invalid dates, likely-wrong data types,
and numeric outliers, and returns *suggestions only* — no changes are applied
automatically (the user reviews first).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from backend.services.analysis import classify_columns


def detect(df: pd.DataFrame) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    issues += _missing(df)
    issues += _duplicates(df)
    issues += _invalid_dates(df)
    issues += _wrong_types(df)
    issues += _outliers(df)

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda i: severity_rank.get(i["severity"], 3))
    return {
        "total_issues": len(issues),
        "clean": len(issues) == 0,
        "issues": issues,
    }


def _missing(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    total = len(df)
    for c in df.columns:
        n = int(df[c].isna().sum())
        if n == 0:
            continue
        pct = n / total * 100
        is_numeric = pd.api.types.is_numeric_dtype(df[c])
        fix = "impute with median" if is_numeric else "impute with mode (most frequent)"
        out.append({
            "type": "missing_values", "column": str(c), "count": n,
            "detail": f"{n} missing ({pct:.1f}%).",
            "suggestion": f"Drop those rows, or {fix}.",
            "severity": "high" if pct > 30 else "medium" if pct > 5 else "low",
        })
    return out


def _duplicates(df: pd.DataFrame) -> List[Dict[str, Any]]:
    n = int(df.duplicated().sum())
    if not n:
        return []
    return [{
        "type": "duplicate_rows", "column": None, "count": n,
        "detail": f"{n} fully duplicated row(s).",
        "suggestion": "Remove duplicates (keep the first occurrence).",
        "severity": "medium",
    }]


def _invalid_dates(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    for c in df.columns:
        name = str(c).lower()
        is_texty = pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
        if is_texty and any(k in name for k in ("date", "time", "day")):
            parsed = pd.to_datetime(df[c], errors="coerce")
            n_bad = int(parsed.isna().sum() - df[c].isna().sum())
            if 0 < n_bad:
                out.append({
                    "type": "invalid_dates", "column": str(c), "count": n_bad,
                    "detail": f"{n_bad} value(s) could not be parsed as dates.",
                    "suggestion": "Standardize the date format or set unparseable values to null.",
                    "severity": "medium",
                })
    return out


def _wrong_types(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    for c in df.columns:
        is_texty = pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
        if not is_texty:
            continue
        non_null = df[c].dropna()
        if non_null.empty:
            continue
        coerced = pd.to_numeric(non_null, errors="coerce")
        frac_numeric = coerced.notna().mean()
