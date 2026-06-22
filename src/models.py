"""Fusion models."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import DistilBertModel


class MultimodalFusionModel(nn.Module):
    """Concat fusion — old checkpoints."""

    def __init__(
        self,
        n_structured: int,
        n_classes: int,
        model_name: str = "distilbert-base-uncased",
        struct_hidden: int = 128,
        fusion_hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.struct_encoder = nn.Sequential(
            nn.Linear(n_structured, struct_hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(struct_hidden * 2, struct_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + struct_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, n_classes),
        )

    def forward(self, input_ids, attention_mask, structured, **_kwargs):
        text_hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        struct_hidden = self.struct_encoder(structured)
        fused = torch.cat([text_hidden, struct_hidden], dim=1)
        return self.classifier(fused)


class GatedMultimodalFusionModel(nn.Module):
    """Gated fusion; can drop a modality during training."""

    def __init__(
        self,
        n_structured: int,
        n_classes: int,
        model_name: str = "distilbert-base-uncased",
        struct_hidden: int = 128,
        fusion_hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.text_proj = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.struct_encoder = nn.Sequential(
            nn.Linear(n_structured, struct_hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(struct_hidden * 2, struct_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.struct_proj = nn.Sequential(
            nn.Linear(struct_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(fusion_hidden * 2, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, n_classes),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        structured,
        modality_dropout: float = 0.0,
        training: bool = False,
        return_gate: bool = False,
    ):
        text_hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        struct_hidden = self.struct_encoder(structured)
        text_vec = self.text_proj(text_hidden)
        struct_vec = self.struct_proj(struct_hidden)

        if training and modality_dropout > 0:
            batch_size = text_vec.size(0)
            for i in range(batch_size):
                if torch.rand(1).item() < modality_dropout:
                    if torch.rand(1).item() < 0.5:
                        text_vec[i] = 0.0
                    else:
                        struct_vec[i] = 0.0

        gate_logits = self.gate(torch.cat([text_vec, struct_vec], dim=1))
        weights = torch.softmax(gate_logits, dim=-1)
        fused = weights[:, 0:1] * text_vec + weights[:, 1:2] * struct_vec
        logits = self.classifier(fused)
        if return_gate:
            return logits, weights
        return logits


def build_fusion_model(bundle: dict, n_structured: int, n_classes: int):
    version = bundle.get("architecture", "gated_v2")
    if version == "concat_v1":
        return MultimodalFusionModel(
            n_structured=n_structured,
            n_classes=n_classes,
            struct_hidden=bundle.get("struct_hidden", 128),
            fusion_hidden=bundle.get("fusion_hidden", 256),
            dropout=bundle.get("dropout", 0.2),
        )
    return GatedMultimodalFusionModel(
        n_structured=n_structured,
        n_classes=n_classes,
        struct_hidden=bundle.get("struct_hidden", 128),
        fusion_hidden=bundle.get("fusion_hidden", 256),
        dropout=bundle.get("dropout", 0.2),
    )