"""
VocalSense AI - Production FastAPI Application
Speech Emotion Recognition Backend Service
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.routes.api import router as api_router
from backend.app.services.emotion_service import EmotionInferenceService

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML models
    print("[Server] Starting VocalSense AI backend...")
    service = EmotionInferenceService.get_instance()
    print(f"[Server] Active Emotion Model: {service.model_name}")
    yield
    print("[Server] Shutting down VocalSense AI backend...")


app = FastAPI(
    title="VocalSense AI - Speech Emotion Recognition API",
    description="Deep Learning Based Human Emotion Recognition from Speech (CNN, LSTM, CNN-LSTM Hybrid)",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Mount Frontend static build if available
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend_spa(full_path: str):
        # Don't intercept API paths
        if full_path.startswith("api"):
            return None
        index_file = FRONTEND_DIST / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "VocalSense AI API is running. Build frontend with 'npm run build'."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
