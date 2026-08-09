"""Safe analysis engine.

User questions are mapped to a *constrained* set of operations (aggregate,
timeseries, value_counts, describe, correlation, table). We NEVER execute
arbitrary code or SQL from user input — a plan only names an operation, columns
and parameters, which we run with Pandas. This keeps the AI useful while safe.

Planning is done by the LLM when available (services.llm.plan); otherwise a
heuristic planner inspects keywords + the schema. Execution is identical for
both, and every result carries a Chart.js-ready chart spec.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.services import llm
from backend.services.dataset import _json_safe

AGGS = {"sum", "mean", "count", "min", "max"}
CHARTS = {"bar", "line", "pie", "histogram", "scatter", "table"}


