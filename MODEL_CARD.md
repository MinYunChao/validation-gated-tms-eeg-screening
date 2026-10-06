# Model card

## Model details

The public model is a binary TMS-EEG research classifier composed of a Clean
STGNN waveform backbone, an optional covariance log-feature residual, and an
optional validation-gated metadata branch. The public release version is 0.1.0.

The public architecture is configurable. Parameter counts therefore depend on
channel count and configuration and are recorded for every run. The historical
manuscript implementation contained 26,495,204 trainable parameters for the
Clean backbone and 27,562,598 for the full residual system. Those historical
weights are not distributed.

## Intended use

- Reproduction and extension of subject-exclusive TMS-EEG research workflows.
- Evaluation of where auxiliary evidence enters a screening pipeline.
- Methodological studies of leakage, validation gating, and uncertainty.

## Out-of-scope use

- Standalone diagnosis, triage, treatment selection, prognosis, or patient care.
- Use without local ethics, privacy, and governance review.
- Re-identification, surveillance, employment, insurance, or forensic decisions.
- Claims of efficacy based on the synthetic demo.

## Inputs and outputs

Inputs are preprocessed epochs shaped `[epochs, channels, samples]`, a binary
label and subject grouping key per epoch, and optional aligned waveforms and
non-diagnostic metadata. Output probabilities are aggregated to the subject
level. Threshold 0.5 is the default; it must not be tuned on the held-out test.

## Evaluation

The aggregate manuscript results are in `reports/paper_aggregate_results.json`.
The fixed test contained 18 subjects. Its residual branch achieved sensitivity
1.0000 and specificity 0.5556, producing four false positives among nine
controls. The paired subject bootstrap interval for weighted-F1 improvement
crossed zero. Five-fold internal nested evaluation did not show a stable
residual advantage.

## Risks and mitigations

- **Small-sample overfitting:** use subject-exclusive nested evaluation and external cohorts.
- **False positives:** report specificity and workload, not sensitivity alone.
- **Dataset shift:** recalibrate only on authorized local validation data.
- **Privacy:** do not publish subject rows, checkpoints, or raw signals without approval.
- **Automation bias:** do not present outputs as clinical conclusions.
- **Metadata leakage:** restrict inputs to baseline non-diagnostic variables and audit provenance.

## Training data

The clinical training data are governed human-participant data and are not
included. The synthetic generator is entirely artificial and not a substitute.
See `docs/DATA.md`.

## Weights

No trained manuscript weights are released. See `docs/WEIGHTS.md`.
