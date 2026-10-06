# Manuscript-to-code mapping

This document maps the revised manuscript concepts to the public implementation
and states the limits of numerical reproduction.

| Revised manuscript component | Public implementation | Scope |
| --- | --- | --- |
| Subject-exclusive data contract and split validation | `src/tms_eeg_screening/data.py` | Rejects overlapping, incomplete, single-class, or label-inconsistent splits. |
| EA-Clean-STGNN design principle | `src/tms_eeg_screening/model.py` | Subject-agnostic patch and dynamic-graph backbone without subject-ID normalization. |
| Decision-level covariance evidence | `src/tms_eeg_screening/decision.py` | Stabilized log-covariance features contribute only through a validation-selected logit residual. |
| Validation-gated metadata | `src/tms_eeg_screening/decision.py` | Low-capacity metadata evidence is activated only by a validation-selected policy. |
| Validation-only model and policy selection | `src/tms_eeg_screening/train.py` | Freezes checkpoint, residual strength, and metadata rule before held-out evaluation. |
| Subject aggregation, clinical metrics, and paired bootstrap | `src/tms_eeg_screening/metrics.py` | Treats the subject as the independent reporting unit. |
| Reported protocol defaults | `configs/paper.yaml` | Reference configuration for the public refactor, not a historical checkpoint manifest. |
| Aggregate revised-manuscript values | `reports/paper_aggregate_results.json` | Cohort-level, non-identifying reference results only. |

## Reproduction boundary

The public package is a cleaned and independently organized implementation of
the revised method. It does not contain the internal exploratory training tree,
the governed cohort, frozen subject identifiers, or historical checkpoints.
Consequently, `configs/paper.yaml` documents the public protocol and scale but
does not guarantee byte-identical architecture, optimization trajectories, or
the numerical values reported in the manuscript.

Exact numerical reproduction requires the governed cohort export, the original
subject splits, preprocessing state, and historical model artifacts. Those
items are intentionally not public. The synthetic smoke test verifies software
execution only and must not be used to support scientific or clinical claims.

## Terminology alignment

- **Clean** means the waveform backbone without subject-conditioned feature
  normalization or metadata input.
- **Decision-level residual** means auxiliary evidence modifies the final logit
  and never rewrites the waveform tensor or hidden graph representation.
- **Validation-gated metadata** means the policy, confidence rule, and model are
  chosen using source validation subjects and then frozen.
- **Nested evaluation** remains internal validation when all folds originate
  from the same cohort; it is not external or multi-center validation.
- **Screening** is a research framing. The package is not a diagnostic system or
  a medical device.
