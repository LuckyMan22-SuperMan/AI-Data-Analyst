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


@router.get("/insights")
def get_insights(dataset_id: str) -> dict:
    df = _get_df(dataset_id)
    try:
        return insights.generate(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {exc}")


@router.post("/clean")
def clean(dataset_id: str) -> dict:
    df = _get_df(dataset_id)
    try:
        return cleaning.detect(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cleaning analysis failed: {exc}")


@router.post("/forecast")
def make_forecast(req: ForecastRequest) -> dict:
    df = _get_df(req.dataset_id)
    try:
        return forecast.forecast(df, periods=req.periods, period=req.period,
                                 date_col=req.date_column, metric=req.metric)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {exc}")


@router.get("/export", response_class=HTMLResponse)
def export_report(dataset_id: str) -> HTMLResponse:
    """Generate a standalone HTML report (preview + insights + cleaning)."""
    ds = _get_ds(dataset_id)
    preview = build_preview(ds, head_n=10)
    ins = insights.generate(ds.df)
    clean_res = cleaning.detect(ds.df)
    html_doc = _render_report(ds.filename, preview, ins, clean_res)
    return HTMLResponse(
        content=html_doc,
        headers={"Content-Disposition": f'attachment; filename="report_{dataset_id}.html"'},
    )


def _render_report(filename: str, preview: dict, ins: dict, clean_res: dict) -> str:
    esc = html.escape
    insight_items = "".join(
        f"<li><strong>{esc(i['title'])}:</strong> {esc(i['detail'])}</li>" for i in ins["insights"]
    )
