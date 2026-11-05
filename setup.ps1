# DhwaniDrishti - Phase 1 Environment & Scaffolding Script (Windows PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  DhwaniDrishti: ISL Translation & STEM Tutor Setup   " -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

# 1. Create isolated Python virtual environment (.venv)
if (-Not (Test-Path ".venv")) {
    Write-Host "[1/3] Creating Python virtual environment in .venv..." -ForegroundColor Yellow
    $pyExe = "C:\Program Files\Blender Foundation\Blender 5.2\5.2\python\bin\python.exe"
    if (Test-Path $pyExe) {
        & $pyExe -m venv .venv
    } else {
        python -m venv .venv
    }
} else {
    Write-Host "[1/3] Virtual environment .venv already exists." -ForegroundColor Green
}

# 2. Install requirements inside .venv using signed python.exe
Write-Host "[2/3] Installing backend dependencies from requirements.txt..." -ForegroundColor Yellow
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. Launch FastAPI Server
Write-Host "[3/3] Starting DhwaniDrishti Local Server on http://127.0.0.1:8000 ..." -ForegroundColor Green
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
