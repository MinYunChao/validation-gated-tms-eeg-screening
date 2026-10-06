# Model weights statement

## Release status

No manuscript checkpoint is bundled in this repository or Git history. The
historical weights were trained on restricted human-participant data. Releasing
them requires a separate assessment of consent, membership-inference and model
inversion risk, institutional policy, and third-party code or model licenses.

The absence of weights is deliberate, not a missing Git LFS object. The public
pipeline can be trained on data that follows `DATA.md`.

## If weights are approved later

Publish them as a versioned release asset, not in the Git repository. Each file
must have:

- an immutable release tag;
- SHA-256 checksum and byte size;
- architecture and configuration identifier;
- training-data scope stated at cohort level only;
- intended-use and prohibited-use terms;
- model card, evaluation metrics, and known limitations;
- compatibility version for this package;
- explicit approval record from the responsible institution.

Recommended manifest:

```json
{
  "format_version": 1,
  "artifact": "clean_stgnn_v1.pt",
  "sha256": "REPLACE_AFTER_APPROVAL",
  "bytes": 0,
  "package_version": "0.1.0",
  "config": "configs/paper.yaml",
  "status": "not-released"
}
```

Do not upload checkpoints from local `Results/`, `wandb/`, `outputs/`, or
experiment cache directories. Those files may retain training configuration,
paths, split identifiers, optimizer state, or other project metadata.

## Loading policy

The public trainer writes a state dictionary plus the resolved configuration
and training-only normalization statistics. Load only weights obtained from a
trusted release and verify the checksum. PyTorch checkpoints are serialized
artifacts and should never be accepted from unknown sources.
