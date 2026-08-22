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
    issue_items = "".join(
        f"<li><span class='sev {esc(i['severity'])}'>{esc(i['severity'])}</span> "
        f"<strong>{esc(i['type'])}</strong>"
        f"{(' · ' + esc(str(i['column']))) if i.get('column') else ''}: "
        f"{esc(i['detail'])} <em>{esc(i['suggestion'])}</em></li>"
        for i in clean_res["issues"]
    ) or "<li>No data-quality issues detected.</li>"
    cols = "".join(f"<th>{esc(c)}</th>" for c in preview["column_names"])
    body_rows = "".join(
        "<tr>" + "".join(f"<td>{esc('' if row.get(c) is None else str(row.get(c)))}</td>"
                         for c in preview["column_names"]) + "</tr>"
        for row in preview["head"]
    )
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Data Report - {esc(filename)}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#1e293b}}
h1{{color:#2563eb}}h2{{border-bottom:2px solid #e2e8f0;padding-bottom:6px;margin-top:32px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #e2e8f0;padding:6px 10px;text-align:left}}
th{{background:#f1f5f9}}ul{{line-height:1.7}}
.sev{{font-size:11px;font-weight:700;text-transform:uppercase;padding:2px 8px;border-radius:8px}}
.high{{background:#fee2e2;color:#b91c1c}}.medium{{background:#fef3c7;color:#b45309}}.low{{background:#e0f2fe;color:#0369a1}}
.meta{{color:#64748b;font-size:13px}}
</style></head><body>
<h1>AI Data Analyst — Report</h1>
<p class="meta">File: <strong>{esc(filename)}</strong> · {preview['rows']:,} rows × {preview['columns']} columns ·
Duplicates: {preview['duplicate_rows']}</p>
<h2>Business Insights</h2><ul>{insight_items}</ul>
<h2>Data Cleaning Suggestions</h2><ul>{issue_items}</ul>
<h2>Data Sample (first 10 rows)</h2>
<table><thead><tr>{cols}</tr></thead><tbody>{body_rows}</tbody></table>
</body></html>"""
