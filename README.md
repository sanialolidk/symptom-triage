# Symptom Triage (Multimodal)

Sania Thankan — Penn State, Computational Data Science

Ranks likely conditions from a symptom checklist + a short patient description. Uses [DDXPlus](https://arxiv.org/abs/2205.09148) (general medicine — not psychiatry). Follow-up to my ADE detector: same healthcare-ish NLP space, but multi-class and two input types instead of binary text-only.

## Overview

The app takes age, sex, checked symptoms, and free text. Three models run in parallel:

- **Structured** — gradient boosting on symptom codes + demographics
- **Text** — fine-tuned DistilBERT on the narrative
- **Fusion** — combines both (concat or gated, depending on which `main.py` run you used)

It returns a top-3 list per model and flags when the checklist and the story don't line up. There's also an abstain threshold tuned on the validation split so it doesn't always force a single answer.

**Note on scope:** I call this multimodal because the inputs are genuinely different (checkbox symptoms vs prose). There are no images or audio. The patient text is generated from symptom codes for training; in the UI you can edit that text to simulate someone describing things differently than the form — that's the case I actually care about.

## Compared to my other projects

| ADE detector | This project |
|--------------|--------------|
| Binary (ADE yes/no) | 15-class top-3 ranking |
| Text only | Symptoms + text + fusion |
| DistilBERT vs TF-IDF | DistilBERT + HistGradientBoosting + fusion |
| Streamlit only | FastAPI + React (same layout idea as climate-signal) + Streamlit demo |

## Dataset

[DDXPlus English on HuggingFace](https://huggingface.co/datasets/aai530-group6/ddxplus). I subsample to the 15 most common pathologies so training finishes on my laptop without an all-night run.

## How to run

**Train**
```bash
cd ~/symptom_triage_project
source venv/bin/activate
pip install -r requirements-train.txt
python main.py
python scripts/enrich_metrics.py   # optional — refreshes ECE / abstention stats
```

**App (what I'd demo)**
```bash
# terminal 1
uvicorn api.main:app --reload --port 8001

# terminal 2
cd frontend && npm install && npm run dev
```

http://localhost:5174 — port 8001 so it doesn't fight with climate-signal on 8000.

**Streamlit** (hosted demo): `streamlit run app.py`

Weights aren't in git (~510 MB). Streamlit Cloud pulls them from [sanialolidk/symptom-triage-models](https://huggingface.co/sanialolidk/symptom-triage-models) on first load. One-time upload from a machine that already trained:

```bash
hf login
python scripts/upload_models_hf.py
```

Then deploy at [share.streamlit.io](https://share.streamlit.io) — repo `sanialolidk/symptom-triage`, branch `main`, entrypoint `app.py`, `environment.yml` for deps.

## Results (test split, last run)

| Model | Top-3 acc | Macro F1 | Notes |
|-------|-----------|----------|-------|
| Structured | 1.00 | 0.997 | symptom codes carry most of the signal |
| Text (DistilBERT) | 1.00 | 0.997 | narrative is built from those same codes |
| Fusion | 1.00 | 0.997 | matches text when inputs agree |

Scores look suspiciously perfect because DDXPlus ties the label tightly to the evidence list. The noisy-text ablation in `main.py` is there to show text-only falling off when you mangle the narrative. Real value is the UI path where you **uncheck symptoms or rewrite the story** and watch structured vs text diverge.

## Design decisions

**Why DDXPlus:** public, English, symptom-level labels, and it fits the “intake form + patient words” idea without touching mental health data.

**Why HistGradientBoosting for structured:** sparse binary symptom vector — trees handle it well and train in seconds. I tried keeping a simple logistic baseline early on; GBM was consistently better on top-3.

**Why DistilBERT not full BERT:** ADE project already fine-tuned DistilBERT; same stack, faster on MPS.

**Evidence column bug:** HuggingFace serves `EVIDENCES` as a string that looks like a Python list, not an actual list. First training run had garbage features until I added `ast.literal_eval` in `data.py`. Worth knowing if you reload the dataset.

**React frontend:** copied the climate-signal pattern (FastAPI + Vite proxy). I'm not a frontend person — kept it to two tabs, no routing library. Pinned fever/cough/sore throat to the top of the symptom list and added a "Mismatch example" button because that's the only interesting demo.

**Mismatch demo:** form says shortness of breath + nausea, text says fever/cough/sore throat. Structured and text models disagree — that's the whole point of running both.

## Known limitations

- Synthetic patients, not real EHR data. Don't treat outputs as medical advice.
- 2 epochs on DistilBERT is enough for a class project, not enough for production.
- Abstention threshold is tuned on validate; I didn't do a full nested CV.
- Fusion checkpoint might be concat or gated depending on when you trained — `models/multimodal_bundle.pkl` has `architecture` if you need to check.
- Symptom catalog in the UI is a subset of DDXPlus codes (first ~24 by label sort), not the full 223.

## Tests

```bash
pytest tests/ -q
```

Small suite — mostly parsing and metric helpers. I added it after breaking the evidence loader once.

## Stack

Python, PyTorch, DistilBERT, scikit-learn, FastAPI, React, Vite