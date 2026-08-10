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


def forecast(df: pd.DataFrame, periods: int = 6, period: str = "M",
             date_col: Optional[str] = None,
             metric: Optional[str] = None) -> Dict[str, Any]:
    periods = max(1, min(int(periods), 60))
    dcol, mcol = _resolve_columns(df, date_col, metric)
    freq = _FREQ.get(period, "ME")

    tmp = df[[dcol, mcol]].dropna().copy()
    tmp[dcol] = pd.to_datetime(tmp[dcol], errors="coerce")
    tmp = tmp.dropna(subset=[dcol]).set_index(dcol).sort_index()
    series = tmp[mcol].resample(freq).sum().dropna()
    if len(series) < 4:
        raise ValueError("Need at least 4 time periods of data to forecast.")

    y = series.values.astype(float)
    n = len(y)
    t = np.arange(n)

    # Linear trend.
    slope, intercept = np.polyfit(t, y, 1)
    trend = intercept + slope * t

    # Additive seasonality (average detrended value per season slot).
    m = _SEASON_PERIOD.get(period, 1)
    seasonal = np.zeros(n)
    season_means = None
    if m > 1 and n >= 2 * m:
        resid = y - trend
        slots = t % m
        season_means = np.array([resid[slots == s].mean() if np.any(slots == s) else 0.0
                                 for s in range(m)])
        season_means -= season_means.mean()  # keep it zero-centered
        seasonal = season_means[slots]

    fitted = trend + seasonal
    residuals = y - fitted
    sigma = float(np.std(residuals, ddof=1)) if n > 2 else float(np.std(residuals))
    ci = 1.96 * sigma
