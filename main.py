"""
Root ASGI Entrypoint (main.py)
==============================
Allows Vercel's FastAPI preset (`uvicorn main:app`) and standard ASGI servers
to discover the DhwaniDrishti FastAPI `app` directly at the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.main import app  # noqa: E402

__all__ = ["app"]
