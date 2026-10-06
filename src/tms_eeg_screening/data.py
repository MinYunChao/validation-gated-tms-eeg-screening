from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


@dataclass
class TMSData:
    waveforms: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    aligned_waveforms: np.ndarray | None = None
    metadata: np.ndarray | None = None
    metadata_subject_ids: np.ndarray | None = None
    metadata_names: np.ndarray | None = None

    def validate(self) -> TMSData:
        if self.waveforms.ndim != 3:
            raise ValueError("waveforms must have shape [epochs, channels, samples]")
        n = len(self.waveforms)
        if len(self.labels) != n or len(self.subject_ids) != n:
            raise ValueError("waveforms, labels, and subject_ids must have equal length")
        if not np.isfinite(self.waveforms).all():
            raise ValueError("waveforms contains NaN or infinite values")
        labels = np.unique(self.labels)
        if not set(labels.tolist()).issubset({0, 1}):
            raise ValueError(f"labels must be binary 0/1; found {labels.tolist()}")
        for sid in np.unique(self.subject_ids):
            subject_labels = np.unique(self.labels[self.subject_ids == sid])
            if len(subject_labels) != 1:
                raise ValueError(f"subject {sid!r} has inconsistent labels")
        if (
            self.aligned_waveforms is not None
            and self.aligned_waveforms.shape != self.waveforms.shape
        ):
            raise ValueError("aligned_waveforms must match waveforms exactly")
        if self.metadata is not None:
            if self.metadata.ndim != 2 or self.metadata_subject_ids is None:
                raise ValueError("metadata requires a 2D array and metadata_subject_ids")
            if len(self.metadata) != len(self.metadata_subject_ids):
                raise ValueError("metadata rows must match metadata_subject_ids")
            if len(np.unique(self.metadata_subject_ids)) != len(self.metadata_subject_ids):
                raise ValueError("metadata_subject_ids must contain one row per subject")
            if not np.isfinite(self.metadata).all():
                raise ValueError("metadata contains NaN or infinite values")
        return self

    @property
    def subject_labels(self) -> dict[str, int]:
        return {
            str(sid): int(self.labels[np.flatnonzero(self.subject_ids == sid)[0]])
            for sid in np.unique(self.subject_ids)
        }


class EpochDataset(Dataset):
    def __init__(self, data: TMSData, indices: np.ndarray):
        self.x = torch.as_tensor(data.waveforms[indices], dtype=torch.float32)
        self.y = torch.as_tensor(data.labels[indices], dtype=torch.long)
        subjects = [str(x) for x in data.subject_ids[indices]]
        unique = sorted(set(subjects))
        mapping = {sid: idx for idx, sid in enumerate(unique)}
        self.subject_keys = subjects
        self.subject_index = torch.tensor([mapping[sid] for sid in subjects], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index], self.subject_index[index], self.subject_keys[index]


def load_npz(path: str | Path) -> TMSData:
    with np.load(path, allow_pickle=False) as arrays:
        required = {"waveforms", "labels", "subject_ids"}
        missing = required - set(arrays.files)
        if missing:
            raise ValueError(f"Dataset is missing arrays: {sorted(missing)}")
        values = {name: arrays[name] for name in arrays.files}
    return TMSData(
        waveforms=np.asarray(values["waveforms"], dtype=np.float32),
        labels=np.asarray(values["labels"], dtype=np.int64),
        subject_ids=np.asarray(values["subject_ids"]),
        aligned_waveforms=(
            np.asarray(values["aligned_waveforms"], dtype=np.float32)
            if "aligned_waveforms" in values
            else None
        ),
        metadata=np.asarray(values["metadata"], dtype=np.float32) if "metadata" in values else None,
        metadata_subject_ids=np.asarray(values["metadata_subject_ids"])
        if "metadata_subject_ids" in values
        else None,
        metadata_names=np.asarray(values["metadata_names"]) if "metadata_names" in values else None,
    ).validate()


def make_subject_split(
    data: TMSData,
    seed: int,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
) -> dict[str, list[str]]:
    mapping = data.subject_labels
    subjects = np.asarray(sorted(mapping), dtype=str)
    labels = np.asarray([mapping[sid] for sid in subjects], dtype=int)
    train_val, test = train_test_split(
        subjects, test_size=test_fraction, stratify=labels, random_state=seed
    )
    train_val_labels = np.asarray([mapping[sid] for sid in train_val], dtype=int)
    relative_val = validation_fraction / (1.0 - test_fraction)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val_labels, random_state=seed + 1
    )
    split = {
        "train": sorted(train.tolist()),
        "validation": sorted(val.tolist()),
        "test": sorted(test.tolist()),
    }
    validate_split(data, split)
    return split


def validate_split(data: TMSData, split: dict[str, list[str]]) -> None:
    required = {"train", "validation", "test"}
    if set(split) != required:
        raise ValueError(f"split keys must be {sorted(required)}")
    sets = {name: set(map(str, split[name])) for name in required}
    if (
        sets["train"] & sets["validation"]
        or sets["train"] & sets["test"]
        or sets["validation"] & sets["test"]
    ):
        raise ValueError("subject split is not mutually exclusive")
    available = set(data.subject_labels)
    listed = set().union(*sets.values())
    if listed != available:
        raise ValueError(
            f"split subject mismatch: missing={sorted(available - listed)}, extra={sorted(listed - available)}"
        )
    for name, values in sets.items():
        classes = {data.subject_labels[sid] for sid in values}
        if classes != {0, 1}:
            raise ValueError(f"{name} split must contain both classes")


def load_split(path: str | Path, data: TMSData) -> dict[str, list[str]]:
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    split = {key: [str(item) for item in value] for key, value in split.items()}
    validate_split(data, split)
    return split


def indices_for_subjects(data: TMSData, subjects: list[str]) -> np.ndarray:
    wanted = set(map(str, subjects))
    return np.flatnonzero(np.asarray([str(x) in wanted for x in data.subject_ids]))


def write_synthetic_dataset(path: str | Path, seed: int = 7, n_subjects: int = 30) -> Path:
    if n_subjects < 15 or n_subjects % 2:
        raise ValueError("n_subjects must be an even integer of at least 16")
    rng = np.random.default_rng(seed)
    channels, samples, epochs_per_subject = 8, 128, 4
    time = np.linspace(0, 1, samples, endpoint=False)
    waveforms, aligned, labels, subject_ids = [], [], [], []
    metadata, metadata_ids = [], []
    for subject in range(n_subjects):
        label = subject % 2
        sid = f"synthetic-{subject:03d}"
        subject_bias = rng.normal(0, 0.12, size=(channels, 1))
        for _ in range(epochs_per_subject):
            noise = rng.normal(0, 0.45, size=(channels, samples))
            effect = (0.22 if label else -0.08) * np.sin(2 * np.pi * 6 * time)[None, :]
            signal = noise + subject_bias + effect
            waveforms.append(signal)
            aligned.append(signal - signal.mean(axis=1, keepdims=True))
            labels.append(label)
            subject_ids.append(sid)
        metadata.append(
            [40 + rng.normal(0, 8), subject % 2, 0.6 + rng.normal(0, 0.05), 55 + rng.normal(0, 5)]
        )
        metadata_ids.append(sid)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        waveforms=np.asarray(waveforms, dtype=np.float32),
        aligned_waveforms=np.asarray(aligned, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        subject_ids=np.asarray(subject_ids),
        metadata=np.asarray(metadata, dtype=np.float32),
        metadata_subject_ids=np.asarray(metadata_ids),
        metadata_names=np.asarray(["age", "sex", "motor_threshold", "stimulation_intensity"]),
    )
    return path
