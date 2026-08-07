"""Business insights + data-cleaning endpoints (Phase 3)."""

from __future__ import annotations

import html
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from backend.services import cleaning, forecast, insights
from backend.services.dataset import build_preview, service

router = APIRouter(tags=["reports"])


class ForecastRequest(BaseModel):
    dataset_id: str
    periods: int = 6
    period: str = "M"
    date_column: str | None = None
    metric: str | None = None


def _get_df(dataset_id: str):
    try:
        return service.get(dataset_id).df
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload again.")


def _get_ds(dataset_id: str):
    try:
        return service.get(dataset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload again.")
