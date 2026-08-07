from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backend.services import llm
from backend.services.analysis import _pick_metric, classify_columns
