"""Train structured, text, and fusion models."""

from __future__ import annotations

import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)

from .config import DEFAULT_CONFIG, ExperimentConfig
from .data import augment_narrative, prepare_with_validation
from .evaluation import (
    cross_validate_structured,
    modality_disagreement,
    summarize_predictions,
    tune_abstention_threshold,
)
from .models import GatedMultimodalFusionModel
from .paths import project_path

MODEL_NAME = "distilbert-base-uncased"


class TextOnlyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=160):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class FusionDataset(Dataset):
    def __init__(self, texts, structured, labels, tokenizer, max_length=160):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        self.structured = torch.tensor(structured, dtype=torch.float32)
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["structured"] = self.structured[idx]
        item["labels"] = self.labels[idx]
        return item


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _predict_text_probs(model, tokenizer, texts, device, batch_size=32, max_length=160):
    model.eval()
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start : start + batch_size])
        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            chunks.append(torch.softmax(model(**enc).logits, dim=1).cpu().numpy())
    return np.vstack(chunks)


def _predict_fusion_probs(model, tokenizer, texts, structured, device, batch_size=32, max_length=160):
    model.eval()
    chunks = []
    for start in range(0, len(texts), batch_size):
        sl = slice(start, start + batch_size)
        enc = tokenizer(
            list(texts[sl]),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        struct_t = torch.tensor(structured[sl], dtype=torch.float32, device=device)
        with torch.no_grad():
            chunks.append(torch.softmax(model(enc["input_ids"], enc["attention_mask"], struct_t), dim=1).cpu().numpy())
    return np.vstack(chunks)


def _train_structured(train_df, test_df, structured_cols, cfg: ExperimentConfig):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[structured_cols])
    X_test = scaler.transform(test_df[structured_cols])
    clf = HistGradientBoostingClassifier(max_depth=8, learning_rate=0.08, max_iter=300, random_state=cfg.random_state)
    cv = cross_validate_structured(X_train, train_df["label"].values, clf, folds=cfg.cv_folds)
    clf.fit(X_train, train_df["label"])
    return scaler, clf, clf.predict_proba(X_test), cv


def _train_text_only(train_df, cfg: ExperimentConfig):
    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=cfg.top_n_pathologies)
    model.to(device)
    train_loader = DataLoader(
        TextOnlyDataset(train_df["text"], train_df["label"], tokenizer, cfg.max_length),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(cfg.epochs):
        running = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = loss_fn(model(**batch).logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
        print(f"Text-only epoch {epoch + 1}/{cfg.epochs} — loss {running / len(train_loader):.4f}")

    save_dir = project_path("models", "distilbert_text")
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return model, tokenizer, device


def _train_gated_fusion(train_df, structured_cols, cfg: ExperimentConfig):
    device = _device()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[structured_cols])
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = GatedMultimodalFusionModel(
        n_structured=X_train.shape[1],
        n_classes=cfg.top_n_pathologies,
        struct_hidden=cfg.struct_hidden,
        fusion_hidden=cfg.fusion_hidden,
        dropout=cfg.dropout,
    )
    model.to(device)
    train_loader = DataLoader(
        FusionDataset(train_df["text"], X_train, train_df["label"], tokenizer, cfg.max_length),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(cfg.epochs):
        running = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["structured"],
                modality_dropout=cfg.modality_dropout,
                training=True,
            )
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
        print(f"Gated fusion epoch {epoch + 1}/{cfg.epochs} — loss {running / len(train_loader):.4f}")

    bundle = {
        "scaler": scaler,
        "structured_cols": structured_cols,
        "model_name": MODEL_NAME,
        "n_classes": cfg.top_n_pathologies,
        "architecture": "gated_v2",
        "struct_hidden": cfg.struct_hidden,
        "fusion_hidden": cfg.fusion_hidden,
        "dropout": cfg.dropout,
    }
    save_dir = project_path("models", "multimodal")
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "fusion_model.pt")
    tokenizer.save_pretrained(save_dir)
    joblib.dump(bundle, project_path("models", "multimodal_bundle.pkl"))
    return model, tokenizer, scaler, device


def _ablation_noisy_text(test_df, structured_cols, cfg: ExperimentConfig):
    rng = np.random.default_rng(cfg.random_state)
    noisy_texts = [augment_narrative(t, noise_prob=0.65, rng=rng) for t in test_df["text"]]
    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(project_path("models", "distilbert_text"))
    model = AutoModelForSequenceClassification.from_pretrained(project_path("models", "distilbert_text"))
    model.to(device)
    text_probs = _predict_text_probs(model, tokenizer, noisy_texts, device, max_length=cfg.max_length)
    struct_bundle = joblib.load(project_path("models", "structured_bundle.pkl"))
    struct_probs = struct_bundle["model"].predict_proba(struct_bundle["scaler"].transform(test_df[structured_cols]))
    disagree = float(np.mean([modality_disagreement(a, b) for a, b in zip(struct_probs, text_probs)]))
    y = test_df["label"].values
    return {
        "mean_modality_disagreement": round(disagree, 4),
        "text_top3_under_noise": round(float(np.mean([int(yi in np.argsort(p)[-3:]) for yi, p in zip(y, text_probs)])), 4),
        "structured_top3_stable": round(float(np.mean([int(yi in np.argsort(p)[-3:]) for yi, p in zip(y, struct_probs)])), 4),
    }


def _plot_comparison(metrics: dict):
    keys = ["structured", "text_only", "multimodal"]
    labels = ["Structured", "Text-only", "Gated fusion"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, [metrics[k]["top3_accuracy"] for k in keys], color="#1a4480")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Top-3 accuracy")
    axes[1].bar(labels, [metrics[k]["ece"] for k in keys], color="#b35c00")
    axes[1].set_title("Expected calibration error")
    fig.tight_layout()
    out = project_path("plots", "model_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def run(config: ExperimentConfig | None = None, **overrides):
    cfg = config or DEFAULT_CONFIG
    if overrides:
        cfg = ExperimentConfig(**{**cfg.to_dict(), **overrides})
    print("Loading DDXPlus with train/validation/test splits...")
    train_df, val_df, test_df, meta = prepare_with_validation(
        max_train_samples=cfg.max_train_samples,
        max_val_samples=800,
        max_test_samples=cfg.max_test_samples,
        top_n_pathologies=cfg.top_n_pathologies,
        text_noise_prob=cfg.text_noise_prob,
        random_state=cfg.random_state,
    )
    y_test = test_df["label"].values
    y_val = val_df["label"].values
    structured_cols = meta["structured_cols"]
    print(f"Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)} | Classes {meta['n_classes']}")

    print("\n=== Structured (HistGradientBoosting) ===")
    scaler, struct_clf, s_prob, struct_cv = _train_structured(train_df, test_df, structured_cols, cfg)
    print("CV macro F1:", struct_cv)

    print("\n=== Text (DistilBERT) ===")
    t0 = time.time()
    text_model, text_tok, device = _train_text_only(train_df, cfg)
    t_prob = _predict_text_probs(text_model, text_tok, test_df["text"], device, max_length=cfg.max_length)
    print(f"Text train time: {time.time() - t0:.1f}s")

    print("\n=== Fusion ===")
    t0 = time.time()
    fusion_model, fusion_tok, fusion_scaler, device = _train_gated_fusion(train_df, structured_cols, cfg)
    X_val = fusion_scaler.transform(val_df[structured_cols])
    X_test = fusion_scaler.transform(test_df[structured_cols])
    m_prob_val = _predict_fusion_probs(
        fusion_model, fusion_tok, val_df["text"].values, X_val, device, max_length=cfg.max_length
    )
    m_prob = _predict_fusion_probs(
        fusion_model, fusion_tok, test_df["text"].values, X_test, device, max_length=cfg.max_length
    )
    print(f"Fusion train time: {time.time() - t0:.1f}s")

    tuned_threshold, tune_stats = tune_abstention_threshold(
        y_val, m_prob_val, target_abstain_rate=cfg.abstain_target_rate
    )
    print("Tuned abstention:", tune_stats)

    structured_metrics = summarize_predictions(y_test, s_prob, tuned_threshold)
    text_metrics = summarize_predictions(y_test, t_prob, tuned_threshold)
    multimodal_metrics = summarize_predictions(y_test, m_prob, tuned_threshold)
    multimodal_metrics["confusion_matrix"] = confusion_matrix(y_test, m_prob.argmax(axis=1)).tolist()

    fusion_bundle = joblib.load(project_path("models", "multimodal_bundle.pkl"))
    fusion_bundle["abstain_threshold"] = tuned_threshold
    joblib.dump(fusion_bundle, project_path("models", "multimodal_bundle.pkl"))

    struct_bundle = {
        "scaler": scaler,
        "model": struct_clf,
        "structured_cols": structured_cols,
        "label_encoder": meta["label_encoder"],
        "mlb": meta["mlb"],
        "pathologies": meta["pathologies"],
        "abstain_threshold": tuned_threshold,
        "modality": "structured",
        "cv": struct_cv,
    }
    joblib.dump(struct_bundle, project_path("models", "structured_bundle.pkl"))

    meta_bundle = {
        "label_encoder": meta["label_encoder"],
        "mlb": meta["mlb"],
        "structured_cols": structured_cols,
        "pathologies": meta["pathologies"],
        "abstain_threshold": tuned_threshold,
        "device_hint": str(device),
        "architecture": "gated_v2",
    }
    joblib.dump(meta_bundle, project_path("models", "project_meta.pkl"))

    ablation = _ablation_noisy_text(test_df, structured_cols, cfg)
    print("\nAblation (noisy narrative):", ablation)
    print(classification_report(y_test, m_prob.argmax(axis=1), zero_division=0))

    metrics = {
        "config": cfg.to_dict(),
        "dataset": {
            "name": "DDXPlus (English)",
            "task": "symptom triage, top-3 conditions",
            "train_samples": int(len(train_df)),
            "val_samples": int(len(val_df)),
            "test_samples": int(len(test_df)),
            "n_classes": int(meta["n_classes"]),
            "pathologies": meta["pathologies"],
        },
        "eval_notes": {
            "split": "DDXPlus train/validate/test, stratified subsample",
            "abstention": "threshold tuned on validate (~12% abstain target)",
            "text_aug": "random clause drop on training narratives",
            "fusion": "gated model, modality dropout during training",
        },
        "structured": {**structured_metrics, "cv": struct_cv},
        "text_only": text_metrics,
        "multimodal": multimodal_metrics,
        "abstention_policy": tune_stats,
        "ablation_noisy_text": ablation,
        "improvement": {
            "top3_delta_vs_text": round(multimodal_metrics["top3_accuracy"] - text_metrics["top3_accuracy"], 4),
            "ece_delta_vs_text": round(multimodal_metrics["ece"] - text_metrics["ece"], 4),
        },
    }

    metrics_path = project_path("models", "metrics.json")
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    plot_path = _plot_comparison(metrics)
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved comparison plot to {plot_path}")


if __name__ == "__main__":
    run()