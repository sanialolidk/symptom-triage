"""Load DDXPlus and build features."""

from __future__ import annotations

import ast
import json
from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from .paths import project_path

DATASET_ID = "aai530-group6/ddxplus"
SEG = "_@_"
CHOICES = {"0": "no", "1": "yes", "N": "no", "Y": "yes", "F": "female", "M": "male"}


def _download_asset(filename: str) -> str:
    cached = project_path("data", filename)
    if cached.exists():
        return str(cached)
    path = hf_hub_download(repo_id=DATASET_ID, filename=filename, repo_type="dataset")
    cached.parent.mkdir(parents=True, exist_ok=True)
    from pathlib import Path as _Path
    cached.write_bytes(_Path(path).read_bytes())
    return str(cached)


@lru_cache(maxsize=1)
def load_evidence_lexicon() -> dict:
    from pathlib import Path as _Path
    return json.loads(_Path(_download_asset("release_evidences.json")).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_condition_lexicon() -> dict:
    from pathlib import Path as _Path
    return json.loads(_Path(_download_asset("release_conditions.json")).read_text(encoding="utf-8"))


class EvidenceTextBuilder:
    """Convert DDXPlus evidence tokens into patient-facing narrative text."""

    def __init__(self, lexicon: dict | None = None):
        self.lexicon = lexicon or load_evidence_lexicon()

    def _parse_token(self, token: str) -> tuple[str, str]:
        if SEG in token:
            name, value = token.split(SEG, 1)
            return name, value
        return token, "1"

    def _answer_text(self, evidence_name: str, value: str) -> str:
        meta = self.lexicon.get(evidence_name)
        if not meta:
            return value.replace("_", " ")
        if meta.get("data_type") == "B":
            return CHOICES.get(value, value)
        meanings = meta.get("value_meaning", {})
        if value in meanings:
            return meanings[value].get("en", value)
        return value.replace("_", " ")

    def phrase(self, token: str) -> str:
        name, value = self._parse_token(token)
        meta = self.lexicon.get(name, {})
        question = meta.get("question_en", name.replace("_", " "))
        answer = self._answer_text(name, value)
        question = question.strip().rstrip("?")
        if str(answer).lower() in {"yes", "no", "na"}:
            if str(answer).lower() == "yes":
                return question
            return ""
        return f"{question}: {answer}"

    def narrative(self, age: int, sex: str, evidences: Iterable[str]) -> str:
        sex_label = CHOICES.get(str(sex), str(sex))
        phrases = []
        for token in evidences:
            line = self.phrase(token)
            if line:
                phrases.append(line)
        symptom_block = "; ".join(phrases[:18]) if phrases else "no specific symptoms recorded"
        return (
            f"Patient is a {int(age)}-year-old {sex_label}. "
            f"Presenting concerns include {symptom_block}."
        )


def _parse_evidence_list(value) -> list[str]:
    # HF gives EVIDENCES as a string like "['E_91', 'E_48']" — not a real list.
    # Took me one bad training run to notice; features were single characters before this.
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except (SyntaxError, ValueError):
            pass
    return []


def _load_split(split: str, max_rows: int | None) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=split)
    df = pd.DataFrame(ds)
    df["EVIDENCES"] = df["EVIDENCES"].map(_parse_evidence_list)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
    return df.reset_index(drop=True)


def prepare_multimodal_frames(
    max_train_samples: int = 6000,
    max_test_samples: int = 1500,
    top_n_pathologies: int = 15,
    random_state: int = 42,
):
    """Build train/test frames with text, structured evidence, and labels."""
    train_raw = _load_split("train", max_train_samples * 3)
    test_raw = _load_split("validate", max_test_samples * 3)

    counts = train_raw["PATHOLOGY"].value_counts()
    keep_labels = counts.head(top_n_pathologies).index.tolist()

    train_raw = train_raw[train_raw["PATHOLOGY"].isin(keep_labels)]
    test_raw = test_raw[test_raw["PATHOLOGY"].isin(keep_labels)]

    if len(train_raw) > max_train_samples:
        train_raw, _ = train_test_split(
            train_raw,
            train_size=max_train_samples,
            stratify=train_raw["PATHOLOGY"],
            random_state=random_state,
        )
    if len(test_raw) > max_test_samples:
        test_raw, _ = train_test_split(
            test_raw,
            train_size=max_test_samples,
            stratify=test_raw["PATHOLOGY"],
            random_state=random_state,
        )

    builder = EvidenceTextBuilder()
    label_encoder = LabelEncoder()
    label_encoder.fit(keep_labels)

    mlb = MultiLabelBinarizer()
    all_evidence_tokens = pd.concat([train_raw["EVIDENCES"], test_raw["EVIDENCES"]])
    mlb.fit(all_evidence_tokens)

    def transform(df: pd.DataFrame) -> pd.DataFrame:
        evidence_matrix = mlb.transform(df["EVIDENCES"])
        evidence_df = pd.DataFrame(
            evidence_matrix.astype(float),
            columns=[f"ev_{i}" for i in range(evidence_matrix.shape[1])],
        )
        base = pd.DataFrame(
            {
                "text": [
                    builder.narrative(row["AGE"], row["SEX"], row["EVIDENCES"])
                    for _, row in df.iterrows()
                ],
                "age": df["AGE"].astype(float).values,
                "sex_m": (df["SEX"].astype(str) == "M").astype(float).values,
                "sex_f": (df["SEX"].astype(str) == "F").astype(float).values,
                "label": label_encoder.transform(df["PATHOLOGY"]).astype(int),
                "pathology": df["PATHOLOGY"].values,
                "evidence_tokens": df["EVIDENCES"].values,
            }
        )
        return pd.concat([base, evidence_df], axis=1).reset_index(drop=True)

    train_df = transform(train_raw)
    test_df = transform(test_raw)

    structured_cols = ["age", "sex_m", "sex_f"] + [c for c in train_df.columns if c.startswith("ev_")]
    meta = {
        "label_encoder": label_encoder,
        "mlb": mlb,
        "structured_cols": structured_cols,
        "pathologies": keep_labels,
        "n_classes": len(keep_labels),
        "evidence_builder": builder,
    }
    return train_df, test_df, meta


def augment_narrative(text: str, noise_prob: float = 0.12, rng: np.random.Generator | None = None) -> str:
    """Drop a random clause — training noise."""
    if rng is None:
        rng = np.random.default_rng(42)
    if rng.random() > noise_prob:
        return text
    clauses = [c.strip() for c in text.replace(";", ".").split(".") if c.strip()]
    if len(clauses) <= 1:
        return text
    drop = rng.integers(0, len(clauses))
    noisy = [c for i, c in enumerate(clauses) if i != drop]
    return ". ".join(noisy) + "."


def prepare_with_validation(
    max_train_samples: int = 6000,
    max_val_samples: int = 800,
    max_test_samples: int = 1500,
    top_n_pathologies: int = 15,
    text_noise_prob: float = 0.12,
    random_state: int = 42,
):
    """Train/validation/test split for threshold tuning and honest evaluation."""
    train_raw = _load_split("train", max_train_samples * 3)
    val_raw = _load_split("validate", max_val_samples * 3)
    test_raw = _load_split("test", max_test_samples * 3)

    counts = train_raw["PATHOLOGY"].value_counts()
    keep_labels = counts.head(top_n_pathologies).index.tolist()

    train_raw = train_raw[train_raw["PATHOLOGY"].isin(keep_labels)].copy()
    val_raw = val_raw[val_raw["PATHOLOGY"].isin(keep_labels)].copy()
    test_raw = test_raw[test_raw["PATHOLOGY"].isin(keep_labels)].copy()

    def subsample(df, n):
        if len(df) > n:
            df, _ = train_test_split(df, train_size=n, stratify=df["PATHOLOGY"], random_state=random_state)
        return df

    train_raw = subsample(train_raw, max_train_samples)
    val_raw = subsample(val_raw, max_val_samples)
    test_raw = subsample(test_raw, max_test_samples)

    builder = EvidenceTextBuilder()
    label_encoder = LabelEncoder()
    label_encoder.fit(keep_labels)
    mlb = MultiLabelBinarizer()
    mlb.fit(pd.concat([train_raw["EVIDENCES"], val_raw["EVIDENCES"], test_raw["EVIDENCES"]]))

    rng = np.random.default_rng(random_state)

    def transform(df: pd.DataFrame, augment: bool = False) -> pd.DataFrame:
        texts = []
        for _, row in df.iterrows():
            narrative = builder.narrative(row["AGE"], row["SEX"], row["EVIDENCES"])
            if augment:
                narrative = augment_narrative(narrative, noise_prob=text_noise_prob, rng=rng)
            texts.append(narrative)
        evidence_matrix = mlb.transform(df["EVIDENCES"])
        evidence_df = pd.DataFrame(
            evidence_matrix.astype(float),
            columns=[f"ev_{i}" for i in range(evidence_matrix.shape[1])],
        )
        base = pd.DataFrame(
            {
                "text": texts,
                "age": df["AGE"].astype(float).values,
                "sex_m": (df["SEX"].astype(str) == "M").astype(float).values,
                "sex_f": (df["SEX"].astype(str) == "F").astype(float).values,
                "label": label_encoder.transform(df["PATHOLOGY"]).astype(int),
                "pathology": df["PATHOLOGY"].values,
                "evidence_tokens": df["EVIDENCES"].values,
            }
        )
        return pd.concat([base, evidence_df], axis=1).reset_index(drop=True)

    train_df = transform(train_raw, augment=True)
    val_df = transform(val_raw, augment=False)
    test_df = transform(test_raw, augment=False)

    structured_cols = ["age", "sex_m", "sex_f"] + [c for c in train_df.columns if c.startswith("ev_")]
    meta = {
        "label_encoder": label_encoder,
        "mlb": mlb,
        "structured_cols": structured_cols,
        "pathologies": keep_labels,
        "n_classes": len(keep_labels),
    }
    return train_df, val_df, test_df, meta