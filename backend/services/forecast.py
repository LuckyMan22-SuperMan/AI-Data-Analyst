from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.services.analysis import _pick_metric, classify_columns

_FREQ = {"D": "D", "W": "W", "M": "ME", "Q": "QE", "Y": "YE"}
_SEASON_PERIOD = {"D": 7, "W": 4, "M": 12, "Q": 4, "Y": 1}


def _resolve_columns(df: pd.DataFrame, date_col: Optional[str],
                     metric: Optional[str]) -> tuple[str, str]:
    cls = classify_columns(df)
    if not cls["datetime"]:
        raise ValueError("No date/time column found — forecasting needs a time series.")
    dcol = date_col if date_col in df.columns else cls["datetime"][0]
    mcol = metric if metric in df.columns else _pick_metric("", cls["numeric"])
    if not mcol:
        raise ValueError("No numeric column found to forecast.")
    return dcol, mcol
