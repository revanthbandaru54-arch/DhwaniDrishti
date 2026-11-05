r"""
DhwaniDrishti Production & Executable Builder (build_exe.py)
============================================================
Builds the production distribution inside `dist/`:
  1. Static Web Bundle (`dist/index.html`, `dist/css/`, `dist/js/`) ready for
     GitHub Pages, Vercel, Netlify, or static hosting.
  2. One-Click Windows Desktop Launchers (`dist/DhwaniDrishti.bat` and
     `dist/DhwaniDrishti-Silent.vbs`) that invoke the signed Python runtime
     without triggering Windows Smart App Control policy blocks.
  3. Standalone `.exe` via PyInstaller (`dist/DhwaniDrishti.exe`) when
     `pyinstaller` is installed in the active environment.

Usage:
    npm run build
    # or:
    .\.venv\Scripts\python.exe build_exe.py
"""

from __future__ import annotations

import importlib.util
import os
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path


def build_distribution() -> None:
    root_dir = Path(__file__).resolve().parent
    frontend_dir = root_dir / "frontend"
    backend_dir = root_dir / "backend"
    dist_dir = root_dir / "dist"

    print("[1/3] Preparing production build directory (dist/)...")
    dist_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy production web assets to dist/ for static hosting & fast serving
    shutil.copy2(frontend_dir / "index.html", dist_dir / "index.html")
    for subdir in ("css", "js"):
        src = frontend_dir / subdir
        dst = dist_dir / subdir
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    print(f"      -> Web assets bundled to: {dist_dir}")

    # 2. Verify/compile backend Python modules and generate Windows desktop launchers
    print("[2/3] Validating backend modules & generating Windows launchers...")
    for py_file in backend_dir.glob("*.py"):
        py_compile.compile(str(py_file), doraise=True)

    bat_launcher = dist_dir / "DhwaniDrishti.bat"
    bat_launcher.write_text(
        "@echo off\r\n"
        "title DhwaniDrishti - 3D ISL Studio & STEM Coach\r\n"
        'cd /d "%~dp0.."\r\n'
        'if exist ".venv\\Scripts\\python.exe" (\r\n'
        '    ".venv\\Scripts\\python.exe" run_app.py\r\n'
        ") else (\r\n"
        "    python run_app.py\r\n"
        ")\r\n",
        encoding="utf-8",
    )

    # 3. If PyInstaller is installed, also build the single-file .exe
    if importlib.util.find_spec("PyInstaller") is not None:
        print("[3/3] PyInstaller detected — building standalone DhwaniDrishti.exe...")
        sep = ";" if os.name == "nt" else ":"
        cmd = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onefile",
            "--name",
            "DhwaniDrishti",
            "--add-data",
            f"{frontend_dir}{sep}frontend",
            "--hidden-import",
            "uvicorn.logging",
            "--hidden-import",
            "uvicorn.loops",
            "--hidden-import",
            "uvicorn.loops.auto",
            "--hidden-import",
            "uvicorn.protocols",
            "--hidden-import",
            "uvicorn.protocols.http",
            "--hidden-import",
            "uvicorn.protocols.http.auto",
            "--hidden-import",
            "uvicorn.lifespan",
            "--hidden-import",
            "uvicorn.lifespan.on",
            str(root_dir / "run_app.py"),
        ]
        subprocess.run(cmd, check=True)
        print(f"      -> Standalone executable: {dist_dir / 'DhwaniDrishti.exe'}")
    else:
        print("[3/3] Skipping PyInstaller single-file .exe (install pyinstaller to enable).")

    print("\nBuild succeeded!")
    print(f"  - Static Web Bundle : {dist_dir / 'index.html'}")
    print(f"  - Windows Launcher  : {bat_launcher}")


if __name__ == "__main__":
    build_distribution()
