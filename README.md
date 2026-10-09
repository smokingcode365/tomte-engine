# Tomte Engine

Tomte is a Python memory-engine library. Local development runs Python on the
Windows host and PostgreSQL with pgvector plus Redis in Docker.

## Local development

Prerequisites: Python 3.10+, uv, and Docker Desktop running Linux containers
with Docker Compose. Run commands from this repository. Ports 5432 and 6379
must be available; Compose publishes both on loopback only.

Copy the template only when creating a new local configuration:

```powershell
Copy-Item .env.example .env
```

Choose a new local `POSTGRES_PASSWORD`, then put its percent-encoded form in
`POSTGRES_URL` (for example, `@` becomes `%40`). Keep `.env` private and ignored.
Keep the URL database/user consistent with `POSTGRES_DB` and `POSTGRES_USER`.
The image-created `tomte` role is a superuser for this dedicated local setup.
Redis has no password and is intended only for trusted local development.

Validate, start, and wait for container health:

```powershell
docker compose config --quiet
docker compose up -d --wait
docker compose ps
uv sync --locked
uv run python scripts/bootstrap_local.py --vector-dimension 1536
uv run pytest
```

Use `uv sync --locked --extra security` if optional PII scrubbing is needed.
Select the vector dimension explicitly to match the actual embedding provider
(current configured defaults: OpenAI/Anthropic 1536, Gemini 768). Existing vector
columns cannot be resized by rerunning bootstrap; mismatches fail before setup.

Docker init SQL creates `vector` and `pgcrypto` in the target database before
`PostgresProvider` is constructed. It runs only on an empty PostgreSQL volume.
The host bootstrap script verifies those extensions and uses Tomte's Python
helpers to create tables/indexes without deleting data. It can be rerun with the
same dimension. Core and dictionary setup share one PostgreSQL transaction.

Changing `.env` credentials or init SQL does not reconfigure an existing volume.
The provider still registers vector types during construction, so extension
provisioning is required outside this Docker workflow too.

These commands provision infrastructure and run unit tests; they do not start
the Python application or its daemon worker. A host application must explicitly
load `POSTGRES_URL`/`REDIS_URL`, construct providers, and initialize Tomte after
bootstrap. Local integration tests require explicit environment flags (below).

## Stop or reset local infrastructure

Stop and remove this project's containers/network while retaining data:

```powershell
docker compose down
```

**Destructive reset: `--volumes` deletes local Tomte PostgreSQL and Redis data.**
Use it only when deliberately discarding that data:

```powershell
docker compose down --volumes
docker compose up -d --wait
uv run python scripts/bootstrap_local.py --vector-dimension 1536
```

Redis uses AOF persistence with one-second fsync and no periodic RDB snapshots.
Normal shutdown preserves named volumes; persistence is not a backup.


## Consolidation replay safety

Queue messages receive stable UUIDs. A worker freezes one Redis snapshot per
session; later messages remain in the pending queue. The ordered item identities
and tenant/user/session define a stable batch key, independent of message content.
PostgreSQL claims that key in `tomte_processed_consolidations` in the same
transaction as memory writes and dictionary feedback. A committed replay succeeds
without reapplying changes; failed transactions leave no marker. Redis then
acknowledges only that snapshot. No distributed transaction is used.

Rerun the existing bootstrap command after updating to create the new table.
Processed markers contain no chat payloads and are retained indefinitely, even
after user-memory deletion, so stale batches cannot recreate deleted memories.
Do not remove markers while their batches may still be retried. Queue loss or restoration
from an older Redis backup remains outside this guarantee.

Run local integration checks with `TOMTE_RUN_POSTGRES_TESTS=1` and
`TOMTE_RUN_REDIS_TESTS=1` before `uv run pytest`. They use unique synthetic
fixtures, local deterministic model stubs, and no background daemon.
