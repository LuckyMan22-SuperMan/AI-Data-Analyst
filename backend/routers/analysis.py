"""AI analysis endpoint (Phase 2).

Maps a natural-language question to a constrained analysis plan, executes it
safely with Pandas, and returns a natural-language explanation + a chart spec.
Conversation context is kept per dataset to support follow-up questions.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.models.schemas import AnalyzeRequest, AnalyzeResponse
from backend.services import analysis, llm
from backend.services.dataset import service

router = APIRouter(tags=["ai"])

# Keep chat history bounded per dataset.
_MAX_HISTORY = 12


@router.get("/ai-status")
def ai_status() -> dict:
    return {"llm": llm.info()}


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")
    try:
        ds = service.get(req.dataset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload again.")

    try:
        result = analysis.analyze(question, ds.df, ds.history)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")
