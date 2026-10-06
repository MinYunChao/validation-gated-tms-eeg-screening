#!/usr/bin/env bash
set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$release_root"

python -m tms_eeg_screening make-synthetic \
  --output data/synthetic_demo.npz \
  --subjects 30 \
  --seed 7
python -m tms_eeg_screening inspect-data --data data/synthetic_demo.npz
python -m tms_eeg_screening train \
  --data data/synthetic_demo.npz \
  --config configs/smoke.yaml \
  --output outputs/smoke

test -s outputs/smoke/metrics.json
echo "Smoke test completed: outputs/smoke/metrics.json"
