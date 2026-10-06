from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import load_npz, load_split, write_synthetic_dataset
from .train import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tms-eeg", description="Public TMS-EEG research pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("make-synthetic", help="Create non-clinical smoke-test data")
    synthetic.add_argument("--output", default="data/synthetic_demo.npz")
    synthetic.add_argument("--subjects", type=int, default=30)
    synthetic.add_argument("--seed", type=int, default=7)

    inspect = subparsers.add_parser("inspect-data", help="Validate and summarize an NPZ dataset")
    inspect.add_argument("--data", required=True)

    train = subparsers.add_parser("train", help="Train and evaluate the public pipeline")
    train.add_argument("--data", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--output", default="outputs/run")
    train.add_argument("--split", help="Optional subject split JSON; generated otherwise")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "make-synthetic":
        path = write_synthetic_dataset(args.output, seed=args.seed, n_subjects=args.subjects)
        print(path)
        return
    data = load_npz(args.data)
    if args.command == "inspect-data":
        payload = {
            "epochs": len(data.waveforms),
            "subjects": len(data.subject_labels),
            "class_counts": {
                str(label): sum(value == label for value in data.subject_labels.values())
                for label in (0, 1)
            },
            "waveform_shape": list(data.waveforms.shape),
            "has_aligned_waveforms": data.aligned_waveforms is not None,
            "has_metadata": data.metadata is not None,
        }
        print(json.dumps(payload, indent=2))
        return
    config = load_config(args.config)
    split = load_split(args.split, data) if args.split else None
    summary = run_training(data, config, Path(args.output), split)
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
