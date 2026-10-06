# Contributing

Contributions should be reproducible, narrowly scoped, and free of participant
data. Before submitting a pull request:

1. create an issue describing the scientific or software problem;
2. add or update tests;
3. run `pytest` and `ruff check src tests`;
4. run `bash scripts/release_audit.sh`;
5. document behavioral and configuration changes;
6. confirm that no data, checkpoint, subject identifier, prediction row,
   institutional secret, API key, or absolute local path is included.

Do not attach clinical records or model weights to issues or pull requests.
Security and privacy concerns should be reported privately to the repository
maintainer after an official contact is configured.

All contributions must be compatible with the project license. Contributors
retain responsibility for having the right to submit their code.
