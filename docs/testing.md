# Testing and release validation

Run commands from the repository root. The default suite blocks real HTTP through
the model transport. It uses deterministic model doubles; local HTTP server tests
need loopback sockets. It does not measure live semantic accuracy.

```powershell
uv sync --locked
uv run pytest tests -q
```

Tests requiring PostgreSQL/Redis skip unless explicitly enabled. Skips are not
passing integration evidence. Check the summary and report both executed and
skipped checks.

## Progressive debugging

Reproduce the single failing case. After each correction, rerun that case until it
passes; then run its related test group. After the group passes, run full regression
once. If another failure appears, return to that focused case. Do not weaken
assertions or mock away the behavior being tested.

## Integration tests

Use dedicated disposable infrastructure, never a database containing private or
valuable data. The PostgreSQL fixtures call additive schema setup, so pointing them
at an existing application database can alter its schema even though fixture rows
are uniquely scoped. Redis fixtures remove only their generated synthetic state.
Do not run an application daemon against the test resources.

```powershell
$env:POSTGRES_URL = 'postgresql://TEST_USER:TEST_PASSWORD@127.0.0.1:TEST_PORT/TEST_DATABASE'
$env:REDIS_URL = 'redis://127.0.0.1:TEST_REDIS_PORT/0'
$env:TOMTE_RUN_POSTGRES_TESTS = '1'
$env:TOMTE_RUN_REDIS_TESTS = '1'
uv run python scripts/bootstrap_local.py --vector-dimension 1536
uv run pytest tests -q
```

Replace these placeholders with an isolated, extension-enabled test database and
Redis instance. Never paste real connection strings into a bug report. Tests cover
transactions, replay/ACK, worker fencing, tenant/user/session isolation, immutable
history, derived-index consistency and failure paths. They do not prove arbitrary
process-kill recovery or production load behavior.

`compose.test.yaml` provides separate ephemeral services with random loopback
ports and no application data volumes. Use a new project name:

```powershell
docker compose -f compose.test.yaml -p tomte-test up -d --wait
docker compose -f compose.test.yaml -p tomte-test port postgres 5432
docker compose -f compose.test.yaml -p tomte-test port redis 6379
```

Use those ports with the synthetic credentials in that file. Run bootstrap twice
to verify repeatability before the test suite. Stop/remove only this disposable
project when finished: `docker compose -f compose.test.yaml -p tomte-test down`.
PostgreSQL uses tmpfs; this deliberately loses its test data when stopped.

## Packages and CI

Build wheel and source distribution, inspect their contents and install the wheel
into a clean environment. Verify import outside the source directory using isolated
Python mode. Run the suite against the installed wheel and verify the source
distribution can build another wheel. Do not rely only on an editable checkout.
When running the repository tests outside the checkout, put the repository root
on `PYTHONPATH` for the checkout-only `demo` package. Do not add `src/`. Verify
that `tomte.engine.__file__` still points into the clean environment's site-packages.

CI must run without model credentials or paid calls. A Python-version matrix checks
the declared support range; isolated PostgreSQL/Redis tests verify persistence.
Remote CI results are distinct from local verification and must be recorded as such.

## AI evaluations

Live evaluation is opt-in and costs money. Freeze source, prompt, model, dataset,
gold criteria and pricing before dispatch. Use synthetic data, bounded output,
request/time limits and a persistent conservative budget ledger. Stop on unknown
pricing, unresolved usage or provider errors. Preserve failures and partial runs.

Score every required result and unsupported extra claim. Check identity continuity,
corrections/history, owner attribution, assistant-only claims, unresolved statements,
retrieval coverage and unrelated questions. Separate mechanical invariants from
semantic review. Small development probes are not a broad reliability estimate;
report their sample size and limitations. See [the evaluation plan](memory-quality-evaluation-plan.md).
