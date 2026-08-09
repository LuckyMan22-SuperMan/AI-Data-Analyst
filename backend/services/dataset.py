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
