"""Dataset loading, in-memory registry, and preview computation.

Datasets are held in memory keyed by a dataset_id (returned on upload) and the
raw file is persisted under uploads/. Other routers look datasets up by id, so
each request operates on the correct data without re-uploading.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backend.utils.files import validate_extension


@dataclass
class Dataset:
    dataset_id: str
    filename: str
    path: str
    df: pd.DataFrame
    # Per-dataset chat history for follow-up context (used from Phase 2).
    history: List[Dict[str, str]] = field(default_factory=list)


def load_dataframe(path: Path, filename: str) -> pd.DataFrame:
    """Read a CSV/Excel file into a DataFrame with light type inference."""
    ext = validate_extension(filename)
    if ext == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    if df.empty:
        raise ValueError("The file was read but contains no rows.")
    # Attempt to parse object columns that look like dates.
    df = _infer_datetimes(df)
    return df


def _infer_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        # pandas >= 3.0 reads text as the native "str" dtype (not "object"),
        # so accept both object and string dtypes here.
        is_texty = pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
        if is_texty:
            name = str(col).lower()
            if any(k in name for k in ("date", "time", "day", "month", "year")):
                parsed = pd.to_datetime(df[col], errors="coerce")
                # Only adopt if most values parsed successfully.
                if parsed.notna().mean() >= 0.8:
                    df[col] = parsed
    return df


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalars and NaN/NaT into JSON-serializable values."""
    if value is None:
        return None
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """DataFrame -> list of JSON-safe row dicts."""
    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        out.append({str(k): _json_safe(v) for k, v in row.items()})
    return out
