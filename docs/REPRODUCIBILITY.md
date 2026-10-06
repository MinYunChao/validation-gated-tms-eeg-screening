# Reproducibility scope

This repository supports three levels of reproduction:

1. **Installation reproduction.** Synthetic data exercises loading, subject
   splitting, training, validation selection, residual evidence, metadata
   gating, and subject-level metrics.
2. **Method reproduction.** An authorized dataset export following `DATA.md`
   can be used with the public implementation and fixed configuration.
3. **Numerical manuscript reproduction.** Exact historical values additionally
   require the governed cohort export, frozen original subject splits, and
   historical checkpoints. Those restricted artifacts are not public.

The public code is a cleaned, self-contained implementation of the reported
workflow. It is not byte-for-byte identical to the internal exploratory code.
Consequently, the included aggregate paper values are references, not expected
outputs of the synthetic demo or of arbitrary third-party data.

## Protocol

1. Validate the data contract with `tms-eeg inspect-data`.
2. Supply a subject-exclusive split JSON or let the CLI create a stratified split.
3. Train the Clean STGNN and select its checkpoint on validation subject F1.
4. Select residual strength using validation subjects only.
5. Fit and select metadata gating using training and validation subjects only.
6. Freeze the selected rules before evaluating test subjects.
7. Report epoch metrics as secondary and subject-level metrics as primary.
8. Repeat with subject-level nested folds for internal robustness analysis.

The current CLI implements one subject-exclusive train/validation/test run.
External orchestration can invoke it on each nested fold by passing a frozen
split file with `--split`. Never merge validation or test subjects into training
to reproduce a favorable result.
