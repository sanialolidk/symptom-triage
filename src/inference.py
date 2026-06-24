"""Inference helpers for API + Streamlit."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DistilBertTokenizerFast

from .data import EvidenceTextBuilder, load_evidence_lexicon
from .evaluation import modality_disagreement
from .models import build_fusion_model
from .paths import project_path

MODEL_NAME = "distilbert-base-uncased"


@lru_cache(maxsize=1)
def load_metrics() -> dict | None:
    path = project_path("models", "metrics.json")
    if not path.exists():
        return None
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def load_meta() -> dict:
    return joblib.load(project_path("models", "project_meta.pkl"))


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_structured_row(age: int, sex: str, evidence_tokens: list[str], meta: dict) -> pd.DataFrame:
    row = {"age": float(age), "sex_m": 1.0 if sex == "M" else 0.0, "sex_f": 1.0 if sex == "F" else 0.0}
    evidence_matrix = meta["mlb"].transform([evidence_tokens])[0]
    ev_cols = [c for c in meta["structured_cols"] if c.startswith("ev_")]
    for i, col in enumerate(ev_cols):
        row[col] = float(evidence_matrix[i])
    return pd.DataFrame([row])[meta["structured_cols"]]


def _top_differential(probs: np.ndarray, pathologies: list[str], k: int = 3) -> list[dict[str, Any]]:
    idx = np.argsort(probs)[::-1][:k]
    return [{"pathology": pathologies[i], "probability": float(probs[i])} for i in idx]


def _format_branch(
    probs: np.ndarray,
    pathologies: list[str],
    threshold: float,
    model_name: str,
    extra: dict | None = None,
) -> dict:
    max_prob = float(probs.max())
    top3 = _top_differential(probs, pathologies, k=3)
    abstain = max_prob < threshold
    payload = {
        "model": model_name,
        "top_prediction": top3[0]["pathology"],
        "confidence": max_prob,
        "differential": top3,
        "abstain": abstain,
        "abstain_message": (
            "Not confident enough to call it — would flag for review."
            if abstain
            else None
        ),
    }
    if extra:
        payload.update(extra)
    return payload


def narrative_from_inputs(age: int, sex: str, evidence_tokens: list[str]) -> str:
    builder = EvidenceTextBuilder(load_evidence_lexicon())
    return builder.narrative(age, sex, evidence_tokens)


def explain_symptoms(evidence_tokens: list[str]) -> list[dict[str, str]]:
    builder = EvidenceTextBuilder(load_evidence_lexicon())
    out = []
    for token in evidence_tokens:
        phrase = builder.phrase(token)
        if phrase:
            out.append({"token": token, "description": phrase})
    return out


def symptom_catalog(limit: int = 50) -> list[dict[str, str]]:
    lexicon = load_evidence_lexicon()
    rows = []
    for name, meta in lexicon.items():
        if meta.get("is_antecedent"):
            continue
        question = meta.get("question_en", name)
        if question and question != "NA":
            rows.append({"id": name, "label": question})
    rows.sort(key=lambda r: r["label"])
    return rows[:limit]


# Symptoms I actually demo with — pinned to the top of the React checklist.
_UI_PINNED = ("E_91", "E_201", "E_97", "E_66", "E_148")


def symptom_catalog_for_ui(limit: int = 24) -> list[dict[str, str]]:
    rows = symptom_catalog(300)
    by_id = {row["id"]: row for row in rows}
    pinned = [by_id[sid] for sid in _UI_PINNED if sid in by_id]
    pinned_ids = {row["id"] for row in pinned}
    rest = [row for row in rows if row["id"] not in pinned_ids]
    return (pinned + rest)[:limit]


def evidence_label(token_or_name: str) -> str:
    name = token_or_name.split("_@_")[0]
    meta = load_evidence_lexicon().get(name, {})
    return meta.get("question_en", name)


def common_evidence_options(limit: int = 40) -> list[str]:
    return [row["id"] for row in symptom_catalog(limit)]


def _struct_probs(age: int, sex: str, evidence_tokens: list[str], meta: dict, bundle: dict) -> np.ndarray:
    row = build_structured_row(age, sex, evidence_tokens, meta)
    return bundle["model"].predict_proba(bundle["scaler"].transform(row))[0]


def _text_probs(text: str, device: torch.device) -> np.ndarray:
    model_dir = project_path("models", "distilbert_text")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device).eval()
    enc = tokenizer(text, truncation=True, padding=True, max_length=160, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        return torch.softmax(model(**enc).logits, dim=1).cpu().numpy()[0]


def predict_structured(age: int, sex: str, evidence_tokens: list[str]) -> dict:
    bundle = joblib.load(project_path("models", "structured_bundle.pkl"))
    meta = load_meta()
    probs = _struct_probs(age, sex, evidence_tokens, meta, bundle)
    return _format_result_branch(probs, bundle, "Structured")


def predict_text_only(text: str) -> dict:
    meta = load_meta()
    probs = _text_probs(text, _device())
    return _format_branch(probs, meta["pathologies"], meta["abstain_threshold"], "Text")


def predict_multimodal(age: int, sex: str, evidence_tokens: list[str], text: str) -> dict:
    meta = load_meta()
    bundle = joblib.load(project_path("models", "multimodal_bundle.pkl"))
    device = _device()
    model_dir = project_path("models", "multimodal")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = build_fusion_model(bundle, len(bundle["structured_cols"]), bundle["n_classes"])
    model.load_state_dict(torch.load(model_dir / "fusion_model.pt", map_location=device, weights_only=True))
    model.to(device).eval()

    row = build_structured_row(age, sex, evidence_tokens, meta)
    struct_vec = bundle["scaler"].transform(row)
    enc = tokenizer(text, truncation=True, padding=True, max_length=160, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    struct_t = torch.tensor(struct_vec, dtype=torch.float32, device=device)

    extra = {}
    with torch.no_grad():
        if bundle.get("architecture") == "gated_v2":
            logits, gate = model(
                enc["input_ids"],
                enc["attention_mask"],
                struct_t,
                return_gate=True,
            )
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            extra["modality_weights"] = {
                "text": round(float(gate[0, 0].cpu()), 3),
                "structured": round(float(gate[0, 1].cpu()), 3),
            }
        else:
            logits = model(enc["input_ids"], enc["attention_mask"], struct_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return _format_branch(
        probs,
        meta["pathologies"],
        bundle["abstain_threshold"],
        "Fusion" if bundle.get("architecture") == "gated_v2" else "Fusion (concat)",
        extra=extra or None,
    )


def _format_result_branch(probs: np.ndarray, bundle: dict, model_name: str) -> dict:
    return _format_branch(probs, bundle["pathologies"], bundle["abstain_threshold"], model_name)


def run_triage(age: int, sex: str, evidence_tokens: list[str], text: str) -> dict:
    meta = load_meta()
    device = _device()

    # Load each model once and reuse the raw probability vectors for both
    # the formatted branch output AND the disagreement calculation.
    struct_bundle = joblib.load(project_path("models", "structured_bundle.pkl"))
    struct_full = _struct_probs(age, sex, evidence_tokens, meta, struct_bundle)
    struct = _format_result_branch(struct_full, struct_bundle, "Structured")

    text_full = _text_probs(text, device)
    text_res = _format_branch(text_full, meta["pathologies"], meta["abstain_threshold"], "Text")

    fusion = predict_multimodal(age, sex, evidence_tokens, text)

    disagree = modality_disagreement(struct_full, text_full)
    disagree_flag = disagree > 0.35

    return {
        "input": {
            "age": age,
            "sex": sex,
            "symptom_count": len(evidence_tokens),
            "narrative_preview": text[:240],
        },
        "structured": struct,
        "text_only": text_res,
        "multimodal": fusion,
        "modality_analysis": {
            "disagreement_score": round(disagree, 4),
            "modalities_conflict": disagree_flag,
            "interpretation": (
                "Symptoms and narrative don't really match."
                if disagree_flag
                else "Symptoms and narrative mostly agree."
            ),
        },
        "explainability": {
            "symptoms": explain_symptoms(evidence_tokens),
        },
        "safety": {
            "abstain_recommended": fusion["abstain"] or disagree_flag,
            "rationale": (
                "Low confidence and symptoms/text disagree."
                if fusion["abstain"] and disagree_flag
                else fusion["abstain_message"]
                if fusion["abstain"]
                else "Symptoms and text disagree — I'd double-check this one."
                if disagree_flag
                else "Confidence is fine for a ranked suggestion (not a diagnosis)."
            ),
        },
    }