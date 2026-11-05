#!/usr/bin/env bash
# DhwaniDrishti - Phase 1 Environment & Scaffolding Script (POSIX / Bash)
set -e

echo "======================================================"
echo "  DhwaniDrishti: ISL Translation & STEM Tutor Setup   "
echo "======================================================"

# 1. Create isolated Python virtual environment (.venv)
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating Python virtual environment in .venv..."
    python3 -m venv .venv
else
    echo "[1/4] Virtual environment .venv already exists."
fi

# 2. Upgrade pip and install requirements inside .venv
echo "[2/4] Installing backend dependencies from requirements.txt..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# 3. Download spaCy language model
echo "[3/4] Downloading spaCy NLP model for ISL S-O-V Dependency Parsing..."
.venv/bin/python -m spacy download en_core_web_sm || echo "Notice: Using built-in ISL dependency grammar fallback."

# 4. Launch FastAPI Server
echo "[4/4] Starting DhwaniDrishti Local Server on http://localhost:8000 ..."
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

