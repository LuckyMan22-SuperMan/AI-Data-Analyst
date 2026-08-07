"""File validation and safe I/O helpers."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_BYTES = 50 * 1024 * 1024  # 50 MB


def validate_extension(filename: str) -> str:
    """Return the lowercased extension if allowed, else raise ValueError."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext or '?'}'. Allowed: CSV, XLSX, XLS."
        )
    return ext

