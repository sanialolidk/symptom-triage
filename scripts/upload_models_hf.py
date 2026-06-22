#!/usr/bin/env python3
"""One-time upload of gitignored model artifacts to Hugging Face."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi, create_repo

from src.model_assets import ARTIFACTS, HF_REPO, models_ready

MODELS_DIR = ROOT / "models"


def main() -> None:
    if not models_ready(MODELS_DIR):
        missing = [rel for rel in ARTIFACTS if not (MODELS_DIR / rel).is_file()]
        raise SystemExit(f"Missing local artifacts:\n  " + "\n  ".join(missing))

    api = HfApi()
    who = api.whoami()
    username = who["name"]
    repo_id = f"{username}/symptom-triage-models"
    if repo_id != HF_REPO:
        raise SystemExit(
            f"HF account is {username} but src/model_assets.py expects {HF_REPO}.\n"
            f"Update HF_REPO to {repo_id!r} and push before uploading."
        )
    print(f"logged in as {username}")

    create_repo(repo_id, repo_type="model", exist_ok=True, private=False)

    for rel in ARTIFACTS:
        path = MODELS_DIR / rel
        print(f"uploading {rel} ({path.stat().st_size / 1e6:.1f} MB)")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=rel,
            repo_id=repo_id,
            repo_type="model",
        )

    print(f"done — https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()