# FastAPI layer — same inference code as streamlit app

from __future__ import annotations

import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference import load_metrics, run_triage, symptom_catalog_for_ui

app = FastAPI(title="Symptom Triage API")

origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TriageRequest(BaseModel):
    age: int = Field(ge=1, le=110)
    sex: Literal["M", "F"]
    symptoms: list[str] = Field(default_factory=list)
    text: str = Field(min_length=8)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "symptom-triage"}


@app.get("/api/meta")
def meta():
    metrics = load_metrics()
    if not metrics:
        raise HTTPException(503, "Run python main.py first")
    return {
        "pathologies": metrics["dataset"]["pathologies"],
        "symptom_catalog": symptom_catalog_for_ui(24),
        "eval_notes": metrics.get("eval_notes", {}),
    }


@app.get("/api/metrics")
def metrics_endpoint():
    metrics = load_metrics()
    if not metrics:
        raise HTTPException(404, "No metrics file yet")
    return metrics


@app.post("/api/triage")
def triage(body: TriageRequest):
    if not body.symptoms and len(body.text.strip()) < 12:
        raise HTTPException(400, "Need symptoms or a longer description")
    try:
        return run_triage(body.age, body.sex, body.symptoms, body.text.strip())
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc