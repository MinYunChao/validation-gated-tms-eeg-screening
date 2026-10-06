import json

import numpy as np
import pytest

from tms_eeg_screening.data import (
    load_npz,
    load_split,
    make_subject_split,
    write_synthetic_dataset,
)


def test_synthetic_contract_and_subject_split(tmp_path):
    path = write_synthetic_dataset(tmp_path / "synthetic.npz", n_subjects=30)
    data = load_npz(path)
    assert data.waveforms.shape == (120, 8, 128)
    assert len(data.subject_labels) == 30
    split = make_subject_split(data, seed=11)
    all_subjects = [subject for values in split.values() for subject in values]
    assert len(all_subjects) == len(set(all_subjects)) == 30
    assert all({data.subject_labels[sid] for sid in split[name]} == {0, 1} for name in split)


def test_overlapping_split_is_rejected(tmp_path):
    path = write_synthetic_dataset(tmp_path / "synthetic.npz", n_subjects=30)
    data = load_npz(path)
    split = make_subject_split(data, seed=11)
    split["test"][0] = split["train"][0]
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(ValueError, match="mutually exclusive|mismatch"):
        load_split(split_path, data)


def test_inconsistent_subject_label_is_rejected(tmp_path):
    path = write_synthetic_dataset(tmp_path / "synthetic.npz", n_subjects=30)
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    arrays["labels"] = arrays["labels"].copy()
    arrays["labels"][0] = 1 - arrays["labels"][0]
    broken = tmp_path / "broken.npz"
    np.savez_compressed(broken, **arrays)
    with pytest.raises(ValueError, match="inconsistent labels"):
        load_npz(broken)
