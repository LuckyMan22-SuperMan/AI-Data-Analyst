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
