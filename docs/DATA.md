# Data statement

## What is not in this repository

The clinical TMS-EEG recordings, preprocessed epoch tensors, subject-level
metadata, treatment variables, randomization fields, clinical scores, subject
identifiers, split manifests containing identifiers, and subject-level
predictions are not distributed here. They are human-participant research data
and may also be subject to consent, ethics, institutional, and data-use limits.

The repository does not provide a hidden download URL or a script that fetches
the restricted cohort. Opening the source code does not change the governance
status of the data.

## Data availability

The analyzed cohort contained 89 participants (44 controls and 45 participants
with MDD) and 8,307 retained epochs. Only aggregate, non-identifying statistics
are included in `reports/paper_aggregate_results.json`.

Access to participant data is not automatic. A future request process, if
approved by the responsible institution, should require at minimum:

1. a named principal investigator and institutional affiliation;
2. a research proposal and intended analyses;
3. ethics or IRB documentation where applicable;
4. a signed data-use agreement prohibiting re-identification and onward sharing;
5. approval by the original data controller.

The public repository must not promise access before the institution confirms
that the consent and ethics terms allow it. Replace the contact placeholder in
the main README only after that process exists.

## Public input contract

The CLI accepts a compressed NumPy `.npz` file. Required arrays:

| Array | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `waveforms` | `[epochs, channels, samples]` | float32 | Preprocessed TMS-EEG epochs |
| `labels` | `[epochs]` | int64 | Binary labels: 0=control, 1=MDD |
| `subject_ids` | `[epochs]` | string or integer | Stable IDs used only for grouping |

Optional arrays:

| Array | Shape | Meaning |
| --- | --- | --- |
| `aligned_waveforms` | same as `waveforms` | Independently prepared covariance view |
| `metadata` | `[subjects, features]` | Non-diagnostic baseline variables |
| `metadata_subject_ids` | `[subjects]` | One-to-one metadata row mapping |
| `metadata_names` | `[features]` | Feature names |

The paper metadata branch used only age, sex, motor threshold, and stimulation
intensity. Do not include HAMD, diagnosis-derived variables, treatment arm,
response, post-treatment outcomes, names, dates, medical record numbers, or
other direct identifiers.

## Required preprocessing principles

- Keep all epochs from one participant in exactly one split.
- Fit normalization and every learned preprocessing step on training subjects.
- Select models, thresholds, residual strength, and gating policies on validation subjects.
- Use test subjects once for final reporting.
- Verify channel order, sample rate, epoch timing, units, missing values, and label consistency.
- Create a fresh de-identified ID namespace before producing an export.

The public loader validates shapes and subject-label consistency but cannot
certify that an export is ethically de-identified.

## Synthetic data

Run `tms-eeg make-synthetic`. The generated signals contain no real waveform,
participant, clinical, or demographic information. Their metrics are smoke-test
outputs and must not be compared with the manuscript results.
