# Changelog

## 1.0.0 — first public release candidate (unpublished)

This is the first public release snapshot, prepared with fresh independent Git
history. Earlier internal version 0.2.0 and private development commit IDs are
not public release ancestors. Repository creation, tagging and publication are
separate approval steps; this document is not a publication announcement.

### Included

- Python SDK for registered/guest ingestion, memory context and assistant logging.
- LLM-driven extraction, canonical entity resolution, authorized fact mutation
  and terminology alignment with synthetic evaluation tooling.
- PostgreSQL JSONB/pgvector durable memory and Redis history, queues and session
  continuity; managed transactions, replay markers and worker snapshot ACK.
- Query-relevant and background memory lists; current-fact text projection while
  structured memories retain historical assertion state.
- Trusted-local Compose/bootstrap, an executable worker-disabled SDK example,
  deterministic unit/service tests, MIT licensing and prepared offline CI.

### Support and compatibility

`Tomte.init`, the client/admin factories and documented client methods are the
host integration surface. Providers remain injectable. Workflows, worker
entrypoints, schema helpers and raw storage contracts are internal. No public
HTTP application, authentication service or manual-consolidation facade ships.

Compared with private development, retrieval provider outputs are flat entities,
managed transaction scopes replace manual child commits, and formatted context
has two headings. These are not breaks from an earlier public release. Future
public SDK breaking changes should use major versions; database changes require
explicit migration instructions. No automatic private-database upgrade is promised.

### Known limits

The project is experimental, not production-ready. Semantic accuracy is limited
by small synthetic studies; ungated retrieval can return unrelated memories and
text context excludes historical assertions. Hosts own authentication, model
costs and deployment safety. User-memory wipe is partial, not comprehensive data
erasure. PostgreSQL/Redis have no distributed transaction. See
[evaluation evidence](docs/evaluation.md), [security policy](SECURITY.md) and
[verification scope](docs/release_verification.md). No hosted/support SLA applies.
