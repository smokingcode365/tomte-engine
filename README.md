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

## Session entity continuity

For registered users, discovery finds memories from the **current user message**;
session continuity keeps them available for later short followups. Existing
persona and recent retrieval remain active alongside ungated raw-cosine semantic
retrieval. Session entities are deduplicated with those existing results.
Identity-domain persona records stay outside session activation/retention caps.
Freshly fetched session copies take precedence over older copies of the same
entity from the other streams; their existing relative precedence is unchanged.

Redis stores only canonical entity IDs in
`Tomte:session:entities:<tenant>:<user>:<session>`. PostgreSQL remains the source
of truth: each turn fetches the current payloads in MRU order with tenant/user
constraints, ignoring deleted IDs. Use one `session_id` per conversation.

Activation uses exact case-normalized full canonical-name/alias phrases (no fuzzy
or prefix matching), or raw cosine of at least **0.25**. That floor is empirical
for `text-embedding-3-small` and Tomte's current-message/entity-summary pairing;
it is not a universal embedding threshold. Session activation does not use
mention counts or age decay and does not update those statistics. Exact matches
take priority, then cosine descending, then stable entity ID. At most **5** IDs
are promoted per user turn, with the highest priority first in the MRU.

An atomic Redis operation deduplicates, promotes, retains **10** IDs, and refreshes
a **24-hour sliding inactivity TTL**. The session stream returns only the first
**5** IDs; other retrieval streams may add more context. Assistant messages only
refresh existing TTL and never discover or promote. A user turn with no new
activation also refreshes existing state. Missing state is not created by a
touch, and inactive conversations expire automatically without a close API.

The corresponding settings are `session_entity_cosine_floor`,
`session_entity_activation_cap`, `session_entity_retention_cap`,
`session_entity_return_cap`, and `session_entity_ttl_seconds`.
Temporary session-storage failures retain existing retrieval where possible;
invalid rows and namespace mismatches still raise. Existing persona/history/queue
failure behavior is unchanged. Redis and PostgreSQL do not share a transaction.
When a caller owns an explicit DB transaction, a failed DB child still aborts
that transaction, even when the optional workflow fallback handles its exception.

Known limitations: negated names can activate ("Not the Tesla, I mean the Honda"),
generic messages can activate marginal matches, and lexical matching scans the
scoped user's rows/aliases in SQL. No negation parser or new index is introduced.
The existing activation/session corpus can be replayed against the real services
with `uv run python scripts/evaluate_session_implementation.py` after populating
the ignored embedding cache using the earlier evaluation harnesses. This replay
uses cached real embeddings, synthetic rollback fixtures, and scoped Redis
cleanup; it makes no model calls and starts no worker.

## Live retrieval roles

Registered-user context combines persona (stable identity), recent memories
(continuity across nearby conversations), raw semantic memories (long-term
meaning-based retrieval), and session entities (continuity within a conversation).
Merge order is persona, recent, raw semantic, then session; fresh session payloads
win on duplicates without moving an entity's original position in context.

Live semantic retrieval returns the closest five memories with no similarity
cutoff and deterministic entity-ID/DB-ID ties. Age and mention frequency do not
affect this ranking. Session activation remains separate: exact name/alias or
cosine at least 0.25, with the existing caps and TTL. Frequent retrieval and
weighted semantic APIs remain available for comparisons and other callers; their
statistics and formulas are unchanged. Raw cosine is not universally superior,
and ungated retrieval still adds irrelevant context on unrelated questions.

Transient raw-semantic unavailability can fall back to the other streams.
Recent retrieval retains its existing strict failure behavior. Malformed rows,
programming/security errors, and failed outer transactions still raise.
