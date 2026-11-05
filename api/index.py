"""
Vercel Serverless Function Entrypoint (api/index.py)
====================================================
Exposes the FastAPI `app` instance under `/api` for Vercel Python serverless routing.
"""

from backend.main import app

__all__ = ["app"]

