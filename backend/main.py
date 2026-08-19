from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.routers import analysis, reports, upload

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="AI Data Analyst", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api")
app.include_router(analysis.router, prefix="/api")
app.include_router(reports.router, prefix="/api")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)
