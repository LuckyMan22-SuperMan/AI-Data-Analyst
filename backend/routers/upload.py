"""Upload and preview endpoints (Phase 1)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.models.schemas import PreviewResponse, UploadResponse
from backend.services import dataset as ds_service
from backend.services.dataset import build_preview, load_dataframe, records, service
from backend.utils.files import make_stored_path, validate_extension, validate_size

router = APIRouter(tags=["data"])

UPLOADS_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)
