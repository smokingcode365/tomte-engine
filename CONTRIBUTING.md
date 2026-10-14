# Contributing

Use Python 3.12 for the verified development path. Create `.env` from the
placeholder template only for local services/live calls; never submit secrets.
Run `uv sync --locked` then `uv run pytest tests` for offline tests. Integration
checks are opt-in; follow README and bootstrap a dedicated local database first.
No test should call paid APIs without explicit authorization.

Keep changes focused, use four-space Python indentation and `test_*.py` tests.
Include a regression test for confirmed defects, preserve tenant/user/session
scoping, and explain transaction and retry behavior for persistence changes.
Describe what changed, why, validation, configuration impact and limitations in
pull requests. Prefer `fix:` or `feat:` commit subjects. Do not tune frozen
benchmark labels to improve results. See [evaluation evidence](docs/evaluation.md).

By contributing, you confirm you can license your contribution under the MIT
license in [LICENSE](LICENSE). Dependencies retain their own licenses.


The planned public repository is `smokingcode365/tomte-engine` (not yet created).
Ordinary project inquiries: [smokingcode365@gmail.com](mailto:smokingcode365@gmail.com).
Use the private route in [SECURITY.md](SECURITY.md) for vulnerability details.
V1 starts with fresh public Git history; do not import private development history.
