"""Knobs for training — edit here or pass kwargs into pipeline.run()."""

from dataclasses import asdict, dataclass


@dataclass
class ExperimentConfig:
    dataset_id: str = "aai530-group6/ddxplus"
    model_name: str = "distilbert-base-uncased"
    max_train_samples: int = 6000
    max_test_samples: int = 1500
    top_n_pathologies: int = 15
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_length: int = 160
    struct_hidden: int = 128
    fusion_hidden: int = 256
    dropout: float = 0.2
    modality_dropout: float = 0.15
    text_noise_prob: float = 0.12
    abstain_target_rate: float = 0.12
    random_state: int = 42
    cv_folds: int = 5

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = ExperimentConfig()