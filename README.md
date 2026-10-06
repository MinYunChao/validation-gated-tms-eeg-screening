# Validation-Gated TMS-EEG Screening

[中文说明](README_CN.md)

[Project page](https://minyunchao.github.io/validation-gated-tms-eeg-screening/) |
[Anonymized manuscript](paper/validation-gated-tms-eeg-screening-anonymized.pdf)

Research code for subject-exclusive TMS-EEG depression screening with a clean
spatio-temporal graph backbone and conservative decision-level auxiliary
evidence.

> **Research use only.** This repository does not provide a medical device,
> diagnosis, treatment recommendation, or evidence of clinical deployment
> readiness. Positive model outputs require standard clinical assessment.

## What is included

- An anonymized six-page manuscript without author, affiliation, funding, or
  participant-level information.
- A self-contained Clean STGNN implementation without subject-ID normalization.
- Subject-exclusive train/validation/test splitting with leakage checks.
- A covariance log-feature branch that contributes only a validation-selected
  decision-level residual.
- Validation-gated use of non-diagnostic baseline metadata.
- Epoch-to-subject probability aggregation and clinical-style metrics.
- Paired subject bootstrap utilities.
- A synthetic, non-clinical smoke-test generator.
- Aggregate manuscript results without subject-level records.

## What is not included

- Human-participant TMS-EEG recordings or preprocessed tensors.
- Clinical spreadsheets, subject metadata, split identifiers, or individual predictions.
- Manuscript checkpoints, optimizer states, or W&B artifacts.
- Internal exploratory scripts, logs, failed-run caches, or machine-specific paths.

See [Data statement](docs/DATA.md), [Weights statement](docs/WEIGHTS.md), and
[Manuscript-to-code mapping](docs/MANUSCRIPT_MAPPING.md).

## Repository layout

```text
.
|-- configs/                 # Paper-scale and smoke-test configurations
|-- docs/                    # Data, weights, and reproducibility statements
|-- reports/                 # Aggregate, non-identifying paper results
|-- scripts/                 # Reproducible local commands and release audit
|-- src/tms_eeg_screening/   # Installable implementation
|-- tests/                   # Unit and integration tests
|-- LICENSE
|-- MODEL_CARD.md
`-- pyproject.toml
```

## Installation

Python 3.10 or newer is required. GPU execution is optional.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For exact environment capture after installation:

```bash
python -m pip freeze > local-environment.txt
```

Do not commit `local-environment.txt`; platform-specific lock files are not
presented as universally reproducible environments.

## Five-minute smoke test

```bash
tms-eeg make-synthetic --output data/synthetic_demo.npz --subjects 30
tms-eeg inspect-data --data data/synthetic_demo.npz
tms-eeg train \
  --data data/synthetic_demo.npz \
  --config configs/smoke.yaml \
  --output outputs/smoke
```

The generated data are artificial. Metrics in `outputs/smoke` only show that
the software runs; they do not validate the scientific or clinical claim.

The same check is available as:

```bash
bash scripts/run_smoke_test.sh
```

## Using an authorized dataset

Prepare a de-identified `.npz` file following [the input contract](docs/DATA.md),
then run:

```bash
tms-eeg inspect-data --data /path/to/authorized_export.npz
tms-eeg train \
  --data /path/to/authorized_export.npz \
  --config configs/paper.yaml \
  --output outputs/paper_protocol
```

To use a frozen split, pass a JSON file:

```json
{
  "train": ["S001", "S002"],
  "validation": ["S003", "S004"],
  "test": ["S005", "S006"]
}
```

```bash
tms-eeg train --data export.npz --config configs/paper.yaml \
  --split split.json --output outputs/fold01
```

Every subject in the dataset must appear once and only once. Each split must
contain both classes. The loader rejects overlap and inconsistent labels.

## Outputs

Each run writes:

| File | Purpose | Commit to Git? |
| --- | --- | --- |
| `metrics.json` | Aggregate epoch/subject metrics | Only after privacy review |
| `history.json` | Training and validation history | Optional |
| `resolved_config.json` | Fully resolved configuration | Yes, after path review |
| `split.json` | Subject identifiers by split | **No** for restricted data |
| `subject_predictions.csv` | Individual predictions | **No** |
| `best_model.pt` | Local checkpoint | **No** |
| `auxiliary_models.joblib` | Residual and metadata models | **No** |

The repository `.gitignore` excludes `data/`, `outputs/`, weights, spreadsheet
formats, and common experiment stores. Run `bash scripts/release_audit.sh`
before creating a public commit.

## Method summary

1. The waveform backbone embeds non-overlapping temporal patches.
2. Dynamic similarity graphs operate over channel-time nodes at two scales.
3. Clean epoch probabilities are averaged per subject.
4. Stabilized log-covariance evidence is fitted separately and can add a small
   validation-selected logit residual; it never rewrites the waveform tensor.
5. A low-capacity metadata model may change a subject decision only when a
   policy selected on validation subjects permits it.
6. The held-out test is evaluated after all model and policy choices are frozen.

The public implementation follows these reported design principles but is a
clean-room release refactor, not a byte-identical copy of the exploratory
training tree. See [Reproducibility scope](docs/REPRODUCIBILITY.md) and the
[manuscript-to-code mapping](docs/MANUSCRIPT_MAPPING.md).

## Aggregate reference results

The cohort-level results used in the manuscript are stored in
`reports/paper_aggregate_results.json`. The main boundary is important: the
fixed 18-subject residual result increased subject F1 from 0.6625 to 0.7662,
but its paired bootstrap interval crossed zero and the gain did not reproduce
consistently in five-fold nested evaluation. These are internal research
results, not clinical performance guarantees.

## Data and weight access

Data and historical checkpoints are not publicly downloadable. No automatic
access process is claimed. If institutional sharing is later approved, this
section should be updated with the official data-controller contact and review
procedure. GitHub issues must not be used to post participant information or
request informal file transfers.

## Tests

```bash
pytest
ruff check src tests
```

GitHub Actions runs unit tests and a CPU smoke test on supported Python versions.

## Citation

The public manuscript is intentionally anonymous. Until a stable public
bibliographic record is available, cite the repository URL and immutable
release tag. A `CITATION.cff` is intentionally omitted because it would expose
author identity or present incomplete publication metadata.

## License and contributions

Code in this release directory is provided under the MIT License. The license
does not grant access to participant data, manuscript checkpoints, or
third-party datasets. See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a
pull request.

## Limitations

- The cohort is small and from one governed dataset.
- Fixed-split results have wide uncertainty.
- Nested folds remain internal validation, not external validation.
- Model capacity is high relative to the number of independent subjects.
- Specificity at the reported fixed operating point was limited.
- Recent pretrained EEG encoders were not evaluated in the frozen study.
- Calibration, multi-center transportability, and clinical utility are unknown.
- Synthetic smoke tests cannot assess scientific validity.
