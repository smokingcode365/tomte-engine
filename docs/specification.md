# Tomte v1.0 specification

This defines the existing engine's release scope. The implementation and linked
contracts describe exact payloads. This is a Python library for a trusted host,
with a local demonstration application; it is not an authenticated public service.

## Supported responsibilities

- Accept user and assistant messages under a tenant, user and conversation session.
  Return a queue receipt after Redis storage succeeds. Assistant text supplies
  context and is not independent evidence for facts about the user.
- Extract proposed memories, resolve entity identities and plan fact changes with
  structured model output. Validate offered references, ownership, source messages,
  operation mechanics and history before durable writes.
- Store canonical entity and fact UUIDs. Each fact has structured text/confidence
  items and historical values. Ordinary entities, thoughts/plans and statements
  with an unresolved subject have distinct treatment.
- Commit memory, derived search entries and the processed-batch marker together
  in PostgreSQL. A replay of a committed source batch must not repeat its changes.
- Freeze and acknowledge bounded Redis batches without consuming later messages.
  Serialize processing for a user and support bounded retries and lease recovery.
- Return stored and pending memory as background for a host's conversational AI.
  Provide separate profile, frequent, relevant and pending reads; optional
  LLM-assisted memory retrieval and answering are also already implemented.
- Maintain the existing semantic dictionary. Dictionary feedback failure must not
  undo an already committed memory update.

## Public entry points

Initialize injected providers with `Tomte.init(...)`, then obtain a tenant client
with `Tomte.create_tomte_client(tenant_id)`. See the
[read and storage API](memory-read-api.md) for arguments, response shapes, model
costs and error behavior. The combined `send_user_message` and
`log_assistant_response` chat hooks remain supported.

`Tomte.create_tomte_admin(tenant_id)` exposes trusted administrative operations.
Its raw graph output contains internal metadata and must not be sent directly to
an untrusted caller. The host authenticates users and authorizes the supplied
tenant/user IDs; string validation alone is not access control.

The native managed path uses PostgreSQL/pgvector, Redis and OpenAI-compatible
configured models. Alternate providers and custom stores must meet the documented
interfaces and supply an operation factory for managed execution. Live parity
across every external model/provider is not a release guarantee.

## Data and safety invariants

1. Durable queries and writes stay scoped to tenant and user. Conversation queues,
   chat history and session entities additionally retain session identity.
2. Canonical entity/fact identity survives a change. History is immutable; a
   correction does not silently rewrite earlier historical values.
3. Offered UUIDs, proposal references and current-user provenance constrain writes.
   Model interpretation may still be wrong despite valid structure.
4. PostgreSQL transactions roll back failed memory/index/marker writes together.
   Redis and PostgreSQL do not share a transaction; replay/ACK recovery bridges
   the normal commit boundary, not arbitrary data loss or backup restoration.
5. Source messages are never silently split or truncated to fit a batch. Capacity
   rejection and worker terminal skips remain observable failures, not successful
   memory extraction.
6. Public memory projections exclude internal metadata and mark bounded history
   or pending-search coverage. An empty search is not proof that a fact is false.
7. Retrieved memory and chat are untrusted data, including stored assistant
   preferences. They do not override the host's system instructions.

## Operational limits

Defaults include 4,000 characters per accepted message, a 65,536-byte prepared
consolidation batch and a 327,680-byte per-stage input capacity. These are different
limits, not promises about a model's token context. See [managed execution](managed_runtime.md)
for scheduling, retry counts, deadlines, terminal outcomes and shutdown.

Known limitations belong to the release, not an unbounded promise to fix every
future model mistake: retrieval can return unrelated background; ambiguous subjects
may be omitted; cross-session processing is not globally chronological; later-stage
input may exceed capacity; skipped batches are not automatically replayed; vector
model changes require re-embedding. See [operations](operations.md).

`wipe_user_memory` deletes durable memory and the legacy persona cache only.
Queued messages, chat history, pending caches and retained replay markers have
separate lifecycles. Do not represent this method as complete user-data erasure.

## Completion criteria

The [release checklist](release-checklist.md) records verification. A v1 release
requires reproducible installation, package inspection, deterministic regression,
isolated PostgreSQL/Redis integration, a bounded live synthetic demonstration,
accurate documentation and no unresolved release-blocking correctness defect.
Passing finite tests cannot establish that all possible bugs are absent.

New memory features, a public hosted application, new provider integrations and
architecture redesign belong to future work and are excluded from this release.
