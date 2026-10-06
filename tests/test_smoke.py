import json

from tms_eeg_screening.config import load_config
from tms_eeg_screening.data import load_npz, write_synthetic_dataset
from tms_eeg_screening.train import run_training


def test_end_to_end_smoke(tmp_path):
    data_path = write_synthetic_dataset(tmp_path / "synthetic.npz", n_subjects=30)
    config = load_config("configs/smoke.yaml")
    config.train.epochs = 1
    config.train.patience = 1
    output = tmp_path / "output"
    summary = run_training(load_npz(data_path), config, output)
    assert summary["n_subjects"] == {"train": 18, "validation": 6, "test": 6}
    assert "final_test" in summary["metrics"]
    persisted = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert persisted["notice"] == "Synthetic results are installation checks only."
