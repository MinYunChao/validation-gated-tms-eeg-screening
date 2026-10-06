# Public release checklist

Complete this checklist before creating a public GitHub release.

- [ ] Confirm the repository name, owner, public URL, and release tag.
- [ ] Confirm author order and add `CITATION.cff` only when publication metadata is final.
- [ ] Run `python -m pytest`.
- [ ] Run `python -m ruff check src tests`.
- [ ] Run `bash scripts/run_smoke_test.sh` and confirm the synthetic-data notice.
- [ ] Run `bash scripts/release_audit.sh` after removing generated caches and outputs.
- [ ] Confirm that no real subject IDs, split manifests, predictions, clinical tables, or absolute paths are present.
- [ ] Confirm that no checkpoints, optimizer states, W&B artifacts, or restricted data are present.
- [ ] Confirm that `docs/DATA.md` matches the institutionally approved data-access position.
- [ ] Confirm that `docs/WEIGHTS.md` matches the approved checkpoint-release position.
- [ ] Confirm that aggregate values in `reports/paper_aggregate_results.json` match the accepted manuscript.
- [ ] Create an immutable tagged release and record checksums for any separately approved assets.

The repository is suitable for source publication without clinical data or
historical weights. Data and weight access must not be promised until the
responsible institution approves a formal process.
