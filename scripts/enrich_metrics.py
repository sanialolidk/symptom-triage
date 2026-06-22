#!/usr/bin/env python3
# Re-run metrics (ECE, abstention threshold) without training again.
# I use this when I already have models/ from a long main.py run.

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DistilBertTokenizerFast

from src.config import DEFAULT_CONFIG
from src.data import EvidenceTextBuilder, _parse_evidence_list, load_evidence_lexicon
from src.evaluation import modality_disagreement, summarize_predictions, tune_abstention_threshold
from src.models import build_fusion_model
from src.paths import project_path
from src.pipeline import _device


def _load_eval_frames(meta, max_val=600, max_test=800):
    builder = EvidenceTextBuilder(load_evidence_lexicon())
    keep = set(meta["pathologies"])
    frames = {}
    for split, limit in (("validate", max_val), ("test", max_test)):
        ds = load_dataset("aai530-group6/ddxplus", split=split)
        df = pd.DataFrame(ds)
        df["EVIDENCES"] = df["EVIDENCES"].map(_parse_evidence_list)
        df = df[df["PATHOLOGY"].isin(keep)].head(limit)
        evidence_matrix = meta["mlb"].transform(df["EVIDENCES"])
        evidence_df = pd.DataFrame(
            evidence_matrix.astype(float),
            columns=[c for c in meta["structured_cols"] if c.startswith("ev_")],
        )
        base = pd.DataFrame(
            {
                "text": [builder.narrative(r["AGE"], r["SEX"], r["EVIDENCES"]) for _, r in df.iterrows()],
                "age": df["AGE"].astype(float).values,
                "sex_m": (df["SEX"].astype(str) == "M").astype(float).values,
                "sex_f": (df["SEX"].astype(str) == "F").astype(float).values,
                "label": meta["label_encoder"].transform(df["PATHOLOGY"]).astype(int),
            }
        )
        frames[split] = pd.concat([base, evidence_df], axis=1)
    return frames["validate"], frames["test"]


def main():
    cfg = DEFAULT_CONFIG
    meta = joblib.load(project_path("models", "project_meta.pkl"))
    val_df, test_df = _load_eval_frames(meta)
    structured_cols = meta["structured_cols"]
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    struct = joblib.load(project_path("models", "structured_bundle.pkl"))
    s_prob = struct["model"].predict_proba(struct["scaler"].transform(test_df[structured_cols]))

    device = _device()
    text_tok = AutoTokenizer.from_pretrained(project_path("models", "distilbert_text"))
    text_model = AutoModelForSequenceClassification.from_pretrained(project_path("models", "distilbert_text"))
    text_model.to(device).eval()
    enc = text_tok(list(test_df["text"]), truncation=True, padding=True, max_length=160, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        t_prob = torch.softmax(text_model(**enc).logits, dim=1).cpu().numpy()

    fusion_bundle = joblib.load(project_path("models", "multimodal_bundle.pkl"))
    if "architecture" not in fusion_bundle:
        fusion_bundle["architecture"] = "concat_v1"
    fusion_tok = DistilBertTokenizerFast.from_pretrained(project_path("models", "multimodal"))
    fusion_model = build_fusion_model(fusion_bundle, len(fusion_bundle["structured_cols"]), fusion_bundle["n_classes"])
    fusion_model.load_state_dict(torch.load(project_path("models", "multimodal/fusion_model.pt"), map_location=device))
    fusion_model.to(device).eval()

    def fusion_probs(texts, X):
        chunks = []
        for i in range(0, len(texts), 32):
            sl = slice(i, i + 32)
            enc = fusion_tok(list(texts[sl]), truncation=True, padding=True, max_length=160, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            st = torch.tensor(X[sl], dtype=torch.float32, device=device)
            with torch.no_grad():
                chunks.append(torch.softmax(fusion_model(enc["input_ids"], enc["attention_mask"], st), dim=1).cpu().numpy())
        return np.vstack(chunks)

    X_val = fusion_bundle["scaler"].transform(val_df[structured_cols])
    X_test = fusion_bundle["scaler"].transform(test_df[structured_cols])
    m_prob_val = fusion_probs(val_df["text"].values, X_val)
    m_prob = fusion_probs(test_df["text"].values, X_test)
    threshold, tune_stats = tune_abstention_threshold(y_val, m_prob_val, target_abstain_rate=cfg.abstain_target_rate)
    disagree = float(np.mean([modality_disagreement(a, b) for a, b in zip(s_prob, t_prob)]))

    metrics_path = project_path("models", "metrics.json")
    base = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    base.update({
        "eval_notes": {
            "split": "DDXPlus official splits",
            "abstention": "threshold from validate set",
            "fusion": fusion_bundle.get("architecture", "concat_v1"),
        },
        "structured": summarize_predictions(y_test, s_prob, threshold),
        "text_only": summarize_predictions(y_test, t_prob, threshold),
        "multimodal": summarize_predictions(y_test, m_prob, threshold),
        "abstention_policy": tune_stats,
        "modality_disagreement_mean": round(disagree, 4),
    })

    meta["abstain_threshold"] = threshold
    joblib.dump(meta, project_path("models", "project_meta.pkl"))
    struct["abstain_threshold"] = threshold
    joblib.dump(struct, project_path("models", "structured_bundle.pkl"))
    fusion_bundle["abstain_threshold"] = threshold
    joblib.dump(fusion_bundle, project_path("models", "multimodal_bundle.pkl"))
    metrics_path.write_text(json.dumps(base, indent=2))
    print(f"Updated {metrics_path}")


if __name__ == "__main__":
    main()