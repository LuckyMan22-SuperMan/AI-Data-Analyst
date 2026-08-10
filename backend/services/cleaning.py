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
