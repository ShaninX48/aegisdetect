"""AegisDetect backend skeleton - Phase 1 Week 1.
Run: pip install -r requirements.txt; uvicorn app.main:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path

from app.api import image, audio, analysis, health

app = FastAPI(title="AegisDetect API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(image.router, prefix="/api/v1", tags=["image"])
app.include_router(audio.router, prefix="/api/v1", tags=["audio"])
app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "index.html"


@app.get("/")
def root():
    # Serve the UI from the same origin (no CORS/file:// issues)
    if _FRONTEND.exists():
        return FileResponse(str(_FRONTEND), media_type="text/html")
    return {"service": "AegisDetect", "docs": "/docs"}
