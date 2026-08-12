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


class PreviewResponse(BaseModel):
    dataset_id: str
    filename: str
    rows: int
    columns: int
    column_names: List[str]
    column_info: List[ColumnInfo]
    duplicate_rows: int
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    head: List[Dict[str, Any]]


class AnalyzeRequest(BaseModel):
    dataset_id: str
    question: str
    answer: str
