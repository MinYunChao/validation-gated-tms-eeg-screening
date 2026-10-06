from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import EpochDataset, TMSData, indices_for_subjects, make_subject_split
from .decision import fit_metadata_gate, fit_residual_branch
from .metrics import aggregate_subject_probabilities, classification_metrics, paired_bootstrap
from .model import CleanSTGNN


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def fit_normalization(data: TMSData, train_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = data.waveforms[train_indices].astype(np.float64)
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True)
    return mean.astype(np.float32), np.clip(std, 1e-6, None).astype(np.float32)


def normalize_data(data: TMSData, mean: np.ndarray, std: np.ndarray) -> TMSData:
    data.waveforms = ((data.waveforms - mean) / std).astype(np.float32)
    if data.aligned_waveforms is not None:
        data.aligned_waveforms = ((data.aligned_waveforms - mean) / std).astype(np.float32)
    return data


@torch.no_grad()
def predict_epochs(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, np.ndarray]:
    model.eval()
    labels, probabilities, subject_ids = [], [], []
    for waveforms, target, _, subject_keys in loader:
        logits = model(waveforms.to(device))
        probability = torch.softmax(logits, dim=1)[:, 1]
        labels.extend(target.numpy().tolist())
        probabilities.extend(probability.cpu().numpy().tolist())
        subject_ids.extend(subject_keys)
    return {
        "labels": np.asarray(labels, dtype=int),
        "probabilities": np.asarray(probabilities, dtype=float),
        "subject_ids": np.asarray(subject_ids, dtype=str),
    }


def subject_metrics(prediction: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
    subject = aggregate_subject_probabilities(
        prediction["labels"], prediction["probabilities"], prediction["subject_ids"]
    )
    return subject, classification_metrics(subject["labels"], subject["probabilities"])


def _metadata_for_subjects(data: TMSData, subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if data.metadata is None or data.metadata_subject_ids is None:
        raise ValueError("metadata branch requested, but metadata is absent")
    lookup = {str(sid): row for sid, row in zip(data.metadata_subject_ids, data.metadata)}
    missing = [str(subject) for subject in subjects if str(subject) not in lookup]
    if missing:
        raise ValueError(f"metadata is missing subjects: {missing}")
    values = np.asarray([lookup[str(subject)] for subject in subjects], dtype=np.float32)
    labels = np.asarray([data.subject_labels[str(subject)] for subject in subjects], dtype=int)
    return values, labels


def _loaders(data: TMSData, split: dict[str, list[str]], config: ExperimentConfig):
    loaders = {}
    indices = {}
    for name in ("train", "validation", "test"):
        indices[name] = indices_for_subjects(data, split[name])
        loaders[name] = DataLoader(
            EpochDataset(data, indices[name]),
            batch_size=config.train.batch_size,
            shuffle=name == "train",
            num_workers=config.train.num_workers,
            drop_last=False,
        )
    return indices, loaders


def run_training(
    data: TMSData,
    config: ExperimentConfig,
    output_dir: str | Path,
    split: dict[str, list[str]] | None = None,
) -> dict:
    seed_everything(config.train.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = split or make_subject_split(
        data,
        config.train.seed,
        config.train.test_fraction,
        config.train.validation_fraction,
    )
    train_indices = indices_for_subjects(data, split["train"])
    mean, std = fit_normalization(data, train_indices)
    data = normalize_data(data, mean, std)
    indices, loaders = _loaders(data, split, config)
    device = resolve_device(config.train.device)
    model = CleanSTGNN(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    best_key, best_epoch, patience = (-1.0, -1.0), 0, 0
    history = []
    checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(1, config.train.epochs + 1):
        model.train()
        losses = []
        for waveforms, target, _, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(waveforms.to(device))
            loss = criterion(logits, target.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_prediction = predict_epochs(model, loaders["validation"], device)
        _, validation_metrics = subject_metrics(validation_prediction)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_subject_weighted_f1": validation_metrics["weighted_f1"],
            "validation_subject_balanced_accuracy": validation_metrics["balanced_accuracy"],
        }
        history.append(row)
        key = (validation_metrics["weighted_f1"], validation_metrics["balanced_accuracy"])
        if key > best_key:
            best_key = key
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "format_version": 1,
                    "model_state": model.state_dict(),
                    "config": config.to_dict(),
                    "normalization_mean": mean,
                    "normalization_std": std,
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )
        else:
            patience += 1
            if patience >= config.train.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    clean_predictions = {
        name: predict_epochs(model, loaders[name], device) for name in ("validation", "test")
    }
    clean_subject = {}
    metrics = {}
    for name in ("validation", "test"):
        clean_subject[name], metrics[f"clean_{name}"] = subject_metrics(clean_predictions[name])

    auxiliary = {}
    final_test_probabilities = clean_subject["test"]["probabilities"].copy()
    if config.residual.enabled and data.aligned_waveforms is not None:
        residual, candidates = fit_residual_branch(
            data.aligned_waveforms[indices["train"]],
            data.labels[indices["train"]],
            data.aligned_waveforms[indices["validation"]],
            data.labels[indices["validation"]],
            data.subject_ids[indices["validation"]],
            clean_predictions["validation"]["probabilities"],
            config.residual.lambda_candidates,
            config.residual.covariance_shrinkage,
            config.train.seed,
        )
        residual_epoch_test = residual.epoch_probabilities(data.aligned_waveforms[indices["test"]])
        combined_epoch_test = residual.combine(
            clean_predictions["test"]["probabilities"], residual_epoch_test
        )
        residual_subject_test = aggregate_subject_probabilities(
            data.labels[indices["test"]], combined_epoch_test, data.subject_ids[indices["test"]]
        )
        metrics["residual_test"] = classification_metrics(
            residual_subject_test["labels"], residual_subject_test["probabilities"]
        )
        metrics["residual_paired_bootstrap"] = paired_bootstrap(
            residual_subject_test["labels"],
            clean_subject["test"]["probabilities"],
            residual_subject_test["probabilities"],
            n_bootstrap=2000,
            seed=config.train.seed,
        )
        auxiliary["residual"] = residual
        final_test_probabilities = residual_subject_test["probabilities"]
        pd.DataFrame(candidates).to_csv(
            output_dir / "residual_validation_candidates.csv", index=False
        )

    if config.metadata.enabled and data.metadata is not None:
        train_subjects = np.asarray(split["train"])
        train_meta, train_labels = _metadata_for_subjects(data, train_subjects)
        val_meta, val_labels = _metadata_for_subjects(
            data, clean_subject["validation"]["subject_ids"]
        )
        test_meta, test_labels = _metadata_for_subjects(data, clean_subject["test"]["subject_ids"])
        names = (
            [str(value) for value in data.metadata_names]
            if data.metadata_names is not None
            else [f"metadata_{index}" for index in range(train_meta.shape[1])]
        )
        gate, candidates = fit_metadata_gate(
            train_meta,
            train_labels,
            val_meta,
            val_labels,
            clean_subject["validation"]["probabilities"],
            names,
            config.metadata.c_values,
            config.metadata.confidence_values,
            config.metadata.policies,
            config.train.seed,
        )
        gated_test, active = gate.apply(clean_subject["test"]["probabilities"], test_meta)
        metrics["metadata_gated_test"] = {
            **classification_metrics(test_labels, gated_test),
            "activation_rate": float(active.mean()),
            "selected_policy": gate.policy,
            "selected_confidence": gate.confidence,
        }
        final_test_probabilities = gated_test
        auxiliary["metadata_gate"] = gate
        pd.DataFrame(candidates).to_csv(
            output_dir / "metadata_validation_candidates.csv", index=False
        )

    predictions = pd.DataFrame(
        {
            "subject_id": clean_subject["test"]["subject_ids"],
            "target": clean_subject["test"]["labels"],
            "clean_probability": clean_subject["test"]["probabilities"],
            "final_probability": final_test_probabilities,
        }
    )
    predictions["clean_prediction"] = (predictions["clean_probability"] >= 0.5).astype(int)
    predictions["final_prediction"] = (predictions["final_probability"] >= 0.5).astype(int)
    predictions.to_csv(output_dir / "subject_predictions.csv", index=False)
    metrics["final_test"] = classification_metrics(
        predictions["target"].to_numpy(), predictions["final_probability"].to_numpy()
    )
    summary = {
        "best_epoch": best_epoch,
        "trainable_parameters": model.trainable_parameters,
        "device": str(device),
        "n_subjects": {name: len(split[name]) for name in split},
        "metrics": metrics,
        "notice": "Synthetic results are installation checks only."
        if str(data.subject_ids[0]).startswith("synthetic-")
        else "Research use only; not a clinical diagnostic output.",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    if auxiliary:
        joblib.dump(auxiliary, output_dir / "auxiliary_models.joblib")
    return summary
