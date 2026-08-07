from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    missing: int
    missing_pct: float
    unique: int


class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    rows: int
    columns: int
    column_names: List[str]
    sample: List[Dict[str, Any]]

