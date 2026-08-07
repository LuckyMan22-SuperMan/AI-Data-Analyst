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


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    """Validate, store, and load a CSV/Excel file; return a quick summary."""
    filename = file.filename or "upload"
    try:
        validate_extension(filename)
        data = await file.read()
        validate_size(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    dataset_id, stored = make_stored_path(UPLOADS_DIR, filename)
    stored.write_bytes(data)

    try:
        df = load_dataframe(stored, filename)
    except ValueError as exc:
        stored.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        stored.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not parse file: {exc}")

    ds = service.register(dataset_id, filename, stored, df)
    return UploadResponse(
        dataset_id=ds.dataset_id,
        filename=ds.filename,
        rows=int(len(df)),
        columns=int(df.shape[1]),
        column_names=[str(c) for c in df.columns],
        sample=records(df.head(5)),
    )
