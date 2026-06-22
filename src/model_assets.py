"""Fetch trained weights from Hugging Face when models/ isn't on disk."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from huggingface_hub import hf_hub_download

from .paths import project_path

HF_REPO = "saniathankan5/symptom-triage-models"

# Artifacts gitignored locally — metrics.json stays in the repo.
ARTIFACTS = (
    "structured_bundle.pkl",
    "multimodal_bundle.pkl",
    "project_meta.pkl",
    "distilbert_text/config.json",
    "distilbert_text/model.safetensors",
    "distilbert_text/special_tokens_map.json",
    "distilbert_text/tokenizer_config.json",
    "distilbert_text/tokenizer.json",
    "distilbert_text/vocab.txt",
    "multimodal/fusion_model.pt",
    "multimodal/special_tokens_map.json",
    "multimodal/tokenizer_config.json",
    "multimodal/tokenizer.json",
    "multimodal/vocab.txt",
)


def models_ready(models_dir: Path | None = None) -> bool:
    root = models_dir or project_path("models")
    return all((root / rel).is_file() for rel in ARTIFACTS)


@lru_cache(maxsize=1)
def ensure_models() -> Path:
    """Download missing artifacts once per process."""
    models_dir = project_path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    if models_ready(models_dir):
        return models_dir

    missing = [rel for rel in ARTIFACTS if not (models_dir / rel).is_file()]
    for rel in missing:
        dest = models_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(repo_id=HF_REPO, filename=rel, repo_type="model")
        dest.write_bytes(Path(cached).read_bytes())

    if not models_ready(models_dir):
        raise RuntimeError(f"Model download incomplete — still missing files under {models_dir}")

    return models_dir