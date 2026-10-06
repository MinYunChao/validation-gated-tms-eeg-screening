from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    patch_size: int = 50
    embedding_dim: int = 128
    graph_hidden_dim: int = 128
    latent_dim: int = 128
    window_sizes: list[int] = field(default_factory=lambda: [6, 2])
    window_strides: list[int] = field(default_factory=lambda: [2, 1])
    temporal_decay: float = 0.2
    dropout: float = 0.6
    num_classes: int = 2


@dataclass
class TrainConfig:
    seed: int = 20260611
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    patience: int = 10
    num_workers: int = 0
    device: str = "auto"
    test_fraction: float = 0.2
    validation_fraction: float = 0.2


@dataclass
class ResidualConfig:
    enabled: bool = True
    covariance_shrinkage: float = 0.2
    lambda_candidates: list[float] = field(default_factory=lambda: [0.0, 0.02, 0.05, 0.08, 0.10])


@dataclass
class MetadataConfig:
    enabled: bool = True
    c_values: list[float] = field(default_factory=lambda: [0.01, 0.1, 1.0])
    confidence_values: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    policies: list[str] = field(
        default_factory=lambda: [
            "clean",
            "specificity_guard",
            "sensitivity_guard",
            "bidirectional_guard",
        ]
    )


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type, values: dict[str, Any] | None):
    return cls(**(values or {}))


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown = set(raw) - {"model", "train", "residual", "metadata"}
    if unknown:
        raise ValueError(f"Unknown top-level configuration keys: {sorted(unknown)}")
    return ExperimentConfig(
        model=_build(ModelConfig, raw.get("model")),
        train=_build(TrainConfig, raw.get("train")),
        residual=_build(ResidualConfig, raw.get("residual")),
        metadata=_build(MetadataConfig, raw.get("metadata")),
    )
