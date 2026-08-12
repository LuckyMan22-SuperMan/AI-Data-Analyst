from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
app = FastAPI(title="AI Data Analyst", version="0.1.0")
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}

