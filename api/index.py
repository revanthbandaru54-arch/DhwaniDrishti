"""
Vercel Serverless Function Entrypoint (api/index.py)
====================================================
Exposes the FastAPI `app` instance under `/api` for Vercel Python serverless routing.
Ensures repository root is on `sys.path` inside Vercel's `/var/task/api` runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.main import app  # noqa: E402

__all__ = ["app"]
