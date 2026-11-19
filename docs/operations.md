# Operations and troubleshooting

This runbook covers a trusted local installation. Follow [setup](../README.md#local-development)
and [the demo](demo.md). Do not expose the demo, PostgreSQL or Redis to public networks.

## Before starting

- Check `docker info`; start Docker Desktop if the engine is unavailable.
- Use `docker compose ps` to check health. Do not reset existing volumes to fix
  credentials, extension or vector-dimension errors.
- Keep `.env` private. It configures local services and nested settings using `__`.
  Configure the host's model API key without printing it. Settings load when the
  Python process imports configuration; restart after changing settings.
- Bootstrap is additive, not a data/vector migration. Existing vector dimensions
  must match the selected embedding provider. Changing embedding models requires
  re-embedding even if dimensions match; see [embedding rollout](embedding-model-rollout.md).

## Normal lifecycle

A host constructs providers, calls `Tomte.init`, serves SDK requests and calls
`Tomte.shutdown(timeout=...)` before closing its own template connections. Check
the boolean result: `False` means resources are still active. Retry shutdown;
do not replace providers while work remains. For an async host, invoke the
synchronous SDK from a host worker thread.

The demo disables the daemon and explicitly consolidates the selected session.
Do not run another worker against that profile. Use fictional profiles and a
different session ID per conversation. Sending a message returns a receipt, not
proof that PostgreSQL contains a fact.

## Failure diagnosis

- **No durable memory yet:** inspect pending messages and worker state. Processing
  may wait for inactivity, the batch boundary, worker slots or a retry due time.
  Empty extraction and ambiguous-owner abstention are also possible.
- **Provider failure:** distinguish transport/rate-limit/quota errors using bounded
  diagnostics, never raw private prompts. Fix configuration/credits before retrying.
  Do not repeatedly retry an exhausted account.
- **Capacity failure:** inspect which stage exceeded its bound. Do not truncate
  source messages, increase limits without evidence, or treat a skipped batch as
  successfully saved memory.
- **Commit succeeded, ACK failed:** allow marker-based recovery. Do not reinsert
  the chat under new IDs; that creates a different batch and can duplicate work.
- **Memory unavailable in chat:** combined chat hooks return an error payload;
  the host may continue its conversation without memory. Separate read APIs
  propagate failures; an error is not an empty successful result.
- **Unexpected search results:** check scope, index backfills and embedding identity.
  Cosine retrieval has no universal relevance guarantee. Bounded history/pending
  results carry coverage notices; preserve them for the caller.
- **Busy user or dictionary:** respect the existing guards/leases and retry policy.
  Never remove another owner's lock to force progress.

See [managed execution](managed_runtime.md) for exact state transitions, terminal
skip semantics and timeouts. Skipped terminal records do not contain the source
chat and are not a dead-letter archive. There is no supported automatic replay of
skipped batches.

## Backups, changes and deletion

Back up PostgreSQL and Redis before a real deployment change. Keep embedding
model/configuration versions with backups. Test restoration in separate resources;
restoring older Redis queues against a newer database is outside ordinary replay
guarantees. Redis AOF and named volumes are persistence, not backup.

Do not use `docker compose down --volumes`, SQL truncation or Redis flush commands
as routine troubleshooting. They destroy data. Stop infrastructure without deleting
volumes using `docker compose stop`.

The SDK memory-wipe method is intentionally partial. It does not erase queued chat,
history, all caches or replay markers; pending chat could later create new memory.
A host promising privacy erasure must design and verify a complete lifecycle before
making that promise. Keep private chat, secrets and diagnostics out of source releases.
