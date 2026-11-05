"""
Vercel FastAPI Entrypoint (main.py)
===================================
Exposes the FastAPI `app` instance at the repository root for Vercel's FastAPI preset.
"""

from backend.main import app

__all__ = ["app"]
