# Upgrading from the October 1.0.0 release

Engine **2.0.0** is the existing November implementation with a corrected release
identity. The original **1.0.0** remains at commit `393f34f` and is not overwritten.
The two implementations must not share a release version. The 2.0 number reflects
compatibility changes; it does not imply SAR or any new development phase was
implemented during release finalization.

## Why a major version

The original release identified `Tomte.init`, client/admin factories and documented
client methods as its host integration surface. The November implementation changes
public memory values returned through that surface, and adds requirements for
injected custom providers in managed execution. These are incompatible changes
under [Semantic Versioning](https://semver.org/), rather than a compatible patch or
minor update. Retaining method names alone does not preserve their entire contract.

## Host application changes

- Adapt consumers of `query_relevant_memories` and `background_memories` to the
  current entity/fact projection: canonical UUIDs, a `value` mapping of
  text/confidence items, and separate `history_values`. Preserve
  `history_is_partial` and other coverage notices. Old scalar/assertion parsing
  is not compatible. See [the read contract](memory-read-api.md).
- Treat `context_text` as formatted background. Exact text/heading parsers need
  review; do not infer that an absent returned item is false or deleted.
- Custom managed providers must supply `operation_factory(control)`, a context
  manager yielding independent `(db, llm)` resources with bounded I/O and correct
  cleanup. The default PostgreSQL/OpenAI path supplies its native factory. See
  [runtime ownership](managed_runtime.md).
- Replacing providers or options requires `Tomte.shutdown(...)` to return `True`
  before reinitialization. Identical repeated initialization is idempotent.
- The explicitly supplied OpenAI key takes precedence over the environment key.
  Review hosts that previously passed a placeholder while relying on environment
  credentials. Do not print credentials when checking this configuration.

## Existing data and embeddings

There is no automatic conversion of the old assertion/scalar memory format,
queued source snapshots or cached representations. Bootstrap creates/checks
infrastructure; it is not a migration tool. Do not point the new writer at old
application data and assume bootstrap makes it compatible.

1. Stop writers and background workers. Back up PostgreSQL and Redis and retain
   the old environment/configuration for rollback.
2. Choose either separately provisioned empty application stores for the new
   installation, or an explicitly designed and validated data migration. No
   generic migration is shipped. Keep existing stores intact until that choice
   and restoration procedure have been tested.
3. Match embedding model and dimensions across queries, stored memory and
   dictionary anchors. OpenAI defaults now use `text-embedding-3-large` at 1,536
   dimensions. Same dimensions do not make old small-model vectors compatible.
   Re-embedding and derived-index backfills need their own rollout and cost plan;
   see [embedding rollout](embedding-model-rollout.md) and the read contract.
4. Verify a synthetic ingestion/change/retrieval cycle in the new environment,
   including history, tenant/user/session boundaries and failure handling,
   before routing real application traffic to it.

Never use volume deletion, SQL truncation or Redis flush as an implicit upgrade.
This release neither resets existing resources nor runs a migration.

## Limits that remain

`wipe_user_memory` still deletes durable memory and the legacy persona cache only.
Other pending/history/cache state remains; queued messages can later create memory.
It is not complete user-data erasure. The live semantic evidence is preliminary,
and production readiness is unproven. See [operations](operations.md) and the
[verification record](release-verification.md).
