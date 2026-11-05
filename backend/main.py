"""
DhwaniDrishti FastAPI Backend Server (Phase 3)
==============================================
Provides REST endpoints for the Three Core Pillars:
  1. LectureCast (Teacher -> Student): S-V-O to ISL S-O-V Dependency Parser & Avatar Keyframe Generator
  2. DoubtDesk (Student -> Teacher): 543-Landmark Temporal 1D-CNN/LSTM Sequence Classifier
  3. ISL-STEM Coach (Interactive Tutor): Dynamic Time Warping (DTW) + Vector Cosine Distance Evaluator
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.grammar_engine import ISLGrammarEngine
from backend.dtw_evaluator import evaluate_stem_sign, list_stem_benchmarks
from backend.sequence_classifier import TemporalSignSequenceClassifier


app = FastAPI(
    title="DhwaniDrishti API",
    description=(
        "Bi-directional Indian Sign Language (ISL) Translation & STEM Tutoring Engine "
        "aligned with NEP 2020 Accessibility Mandates."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize singleton AI engines
grammar_engine = ISLGrammarEngine()
sequence_classifier = TemporalSignSequenceClassifier()


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class LectureCastParseRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Spoken classroom transcript in English, Telugu, or Hindi")
    language: str = Field(default="en-IN", description="BCP-47 language tag (en-IN, te-IN, hi-IN)")


class DoubtDeskClassifyRequest(BaseModel):
    sequence: List[List[float]] = Field(
        ...,
        description="Time-series sequence of normalized kinematic landmark vectors extracted from 543 MediaPipe Holistic landmarks",
    )
    target_language: str = Field(default="en-IN", description="Output speech synthesis language (en-IN, te-IN, hi-IN)")
    hint_index: int = Field(default=-1, description="Optional deterministic demo class index (-1 for live inference)")


class StemCoachEvaluateRequest(BaseModel):
    term_id: str = Field(default="Algorithm", description="Target STEM concept (Algorithm, Derivative, Photosynthesis, Momentum, Recursion)")
    student_sequence: List[List[float]] = Field(
        ...,
        description="Live time-series trajectory of 12 key 3D anatomical joints (shape: T x 36)",
    )
    measured_palm_angle_deg: Optional[float] = Field(
        default=None,
        description="Live right-palm inclination angle in degrees computed on the client",
    )
    measured_eyebrow_raise: Optional[float] = Field(
        default=None,
        description="Normalized non-manual eyebrow elevation ratio [0.0 - 1.0]",
    )


# ---------------------------------------------------------------------------
# Phase 3 Core API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
@app.get("/health")
async def health_check() -> dict:
    """Returns backend health and active NLP parser model status."""
    return {
        "status": "ok",
        "app": "DhwaniDrishti",
        "nlp_backend": grammar_engine.parser_name,
        "holistic_landmarks": 543,
        "pillars": ["LectureCast", "DoubtDesk", "ISL-STEM Coach"],
    }


@app.post("/api/lecturecast/parse")
@app.post("/lecturecast/parse")
async def parse_lecture_speech(payload: LectureCastParseRequest) -> dict:
    """
    Pillar 1 (LectureCast): Converts spoken S-V-O transcript (English/Telugu/Hindi)
    into ISL S-O-V gloss structure with spatial loci (IX-LEFT/IX-RIGHT), Non-Manual
    Markers (NMM), and 3D Avatar Bone Rotation Keyframes.
    """
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Transcript text cannot be empty.")
    return grammar_engine.parse_to_isl(text=payload.text, language=payload.language)


@app.post("/api/doubtdesk/classify")
@app.post("/doubtdesk/classify")
async def classify_student_doubt(payload: DoubtDeskClassifyRequest) -> dict:
    """
    Pillar 2 (DoubtDesk): Evaluates a multi-frame time-series sequence of MediaPipe
    Holistic landmarks via a lightweight 1D-CNN + LSTM sequence classifier and returns
    English, Telugu, and Hindi translations for classroom audio synthesis.
    """
    if not payload.sequence:
        raise HTTPException(status_code=400, detail="Landmark sequence cannot be empty.")
    return sequence_classifier.classify_sequence(
        landmark_sequence=payload.sequence,
        target_language=payload.target_language,
        hint_index=payload.hint_index,
    )


@app.get("/api/stem-coach/terms")
@app.get("/stem-coach/terms")
async def get_stem_terms() -> dict:
    """
    Pillar 3 (ISL-STEM Coach): Lists available STEM concepts, canonical ISL signing
    descriptions, target angles, and 30-frame 3D benchmark trajectories.
    """
    return {"terms": list_stem_benchmarks()}


@app.post("/api/stem-coach/evaluate")
@app.post("/stem-coach/evaluate")
async def evaluate_stem_practice(payload: StemCoachEvaluateRequest) -> dict:
    """
    Pillar 3 (ISL-STEM Coach): Runs Dynamic Time Warping (DTW) and Vector Cosine Distance
    between the student's live trajectory and the ISL benchmark vector, returning
    per-joint Green/Red overlay statuses and actionable biomechanical correction hints.
    """
    if not payload.student_sequence:
        raise HTTPException(status_code=400, detail="Student trajectory sequence cannot be empty.")
    return evaluate_stem_sign(
        term_id=payload.term_id,
        student_sequence=payload.student_sequence,
        measured_palm_angle_deg=payload.measured_palm_angle_deg,
        measured_eyebrow_raise=payload.measured_eyebrow_raise,
    )


# ---------------------------------------------------------------------------
# Static Frontend Hosting
# ---------------------------------------------------------------------------

def _resolve_frontend_dir() -> Path:
    """Resolves the frontend static assets directory for both local dev and PyInstaller .exe bundles."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")) / "frontend"
    return Path(__file__).resolve().parent.parent / "frontend"


FRONTEND_DIR = _resolve_frontend_dir()


@app.get("/")
async def serve_index():
    """Serves the DhwaniDrishti Single-Page Application."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "DhwaniDrishti API is running. Visit /docs for OpenAPI documentation."}


if FRONTEND_DIR.exists():
    css_dir = FRONTEND_DIR / "css"
    js_dir = FRONTEND_DIR / "js"
    if css_dir.exists():
        app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")
    if js_dir.exists():
        app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

