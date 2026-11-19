# v1.0 release checklist

Scope is frozen to [the existing engine specification](specification.md). This
checklist distinguishes implemented work from evidence still required. Do not
publish a final-version claim while release-blocking items remain unchecked.

Local engineering checks completed on 19 November 2026. See the
[verification record](release-verification.md) for results and scope. Remote CI
and publication are separate from these local checks.

## Behavior and code

- [x] Define existing features, public boundaries and known operational limits.
- [x] Review current implementation and callers for correctness and unfinished paths.
- [x] Fix reproduced release-blocking defects with focused regression coverage.
- [x] Review unused code; remove only code proven unnecessary, preserving public compatibility.
- [x] Pass deterministic regression on the final source state.
- [x] Pass isolated PostgreSQL/Redis integration, including ownership and replay checks.
- [x] Complete a budgeted synthetic live ingestion/change/retrieval demonstration.

## Distribution

- [x] Build wheel and source distribution from the final source state.
- [x] Inspect package contents for secrets, private records, caches and prototypes.
- [x] Install/import/test the wheel in a clean environment outside the checkout.
- [x] Build a wheel from the source distribution.
- [x] Add credential-free CI for unit/package and isolated persistence checks.
- [x] Record which declared Python versions were actually verified.
- [x] Complete license, release notes and consistent version metadata.

## Documentation and handoff

- [x] Written scope/specification, architecture, operational runbook and testing guide.
- [x] Verify README setup and runnable SDK/demo instructions.
- [x] Check technical links and mark superseded design/evaluation descriptions.
- [x] Document supported providers, data migration limits and security reporting.
- [x] Record commands, results, API spend and remaining limitations.

## Publication

Commit, push, tagging and package publication require separate explicit instructions.
Completion of local engineering work does not imply any of those actions occurred.
Before a public release, inspect the actual Git/distribution content and verify
the approved public author identity. Keep private workflow records outside the repository.
