from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
app = FastAPI(title="AI Data Analyst", version="0.1.0")
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}

