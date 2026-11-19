# Contributing

Tomte is a Python memory-engine library and engineering project. Keep changes small,
explain the actual failure, preserve tenant/user/session scoping and add regression
coverage for confirmed bugs. Features beyond the current specification belong in a
separate proposal.

Follow [local setup](README.md#local-development) and [testing](docs/testing.md).
Use the committed lockfile; do not mix dependency upgrades into unrelated fixes.
Never use a private database or live model credentials in routine tests.

Before submitting, run the focused case, related group and full regression in that
order. Explain behavior before/after, design trade-offs, tests actually run,
configuration or schema effects and known limits. Do not claim live model accuracy
from mocked output. Keep credentials, personal conversations and local diagnostics
out of commits and reports.

Bug reports should include the package/Python version, a synthetic reproduction,
expected/actual behavior and sanitized failure metadata. Report security issues
using [the security policy](SECURITY.md).
