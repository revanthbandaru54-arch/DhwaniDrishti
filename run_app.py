"""
DhwaniDrishti Standalone Application Launcher (run_app.py)
==========================================================
Starts the local FastAPI/Uvicorn server and automatically opens the
DhwaniDrishti web interface in the default browser. Compatible with both
standard Python execution and PyInstaller .exe single-file bundles.
"""

from __future__ import annotations

import threading
import time
import webbrowser
import uvicorn

from backend.main import app


def _open_browser_when_ready(url: str, delay_sec: float = 1.2) -> None:
    time.sleep(delay_sec)
    webbrowser.open(url)


def main() -> None:
    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"

    print("=" * 60)
    print("  DhwaniDrishti — Bi-Directional ISL & STEM Tutor Engine")
    print(f"  Launching local server at {url}")
    print("=" * 60)

    threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()

