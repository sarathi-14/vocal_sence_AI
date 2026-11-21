"""
VocalSense AI - Unified Application Launcher
Starts FastAPI backend server hosting the full React frontend and Deep Learning inference engine.
"""

import os
import sys
import webbrowser
from pathlib import Path

# Ensure localized Keras cache & Python paths
PROJECT_ROOT = Path(__file__).resolve().parent
os.environ["KERAS_HOME"] = str(PROJECT_ROOT / ".keras")
os.environ["PYTHONPATH"] = str(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn


def main():
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")

    print("=" * 60)
    print("   VOCALSENSE AI — SPEECH EMOTION RECOGNITION SYSTEM")
    print("=" * 60)
    print(f"Backend API URL:    http://{host}:{port}/api")
    print(f"Web Dashboard URL:  http://{host}:{port}/")
    print(f"Interactive Docs:   http://{host}:{port}/docs")
    print("=" * 60)

    uvicorn.run("backend.app.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
