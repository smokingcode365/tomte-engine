# Tomte Engine

Tomte is a Python memory-engine library. Local development runs Python on the
Windows host and PostgreSQL with pgvector plus Redis in Docker.

## Memory contract

Entities use canonical UUIDs and stable fact UUIDs. Fact values contain multiple
text items with individual confidence; changes keep history inside the same fact.
See [the current entity/fact design](docs/memory-design.md) for public JSON,
internal metadata, dictionary alignment, search, safeguards and breaking-upgrade
requirements. Old scalar/assertion payloads are not automatically migrated.

## Primary SDK API

After `Tomte.init(...)`, create a tenant-scoped client with
`Tomte.create_tomte_client(tenant_id="my-app")`:

```python
client.store_user_message(user_id, session_id, message)
client.store_assistant_message(user_id, session_id, response)
context = client.get_context(user_id, message)
# Use LLM-assisted search when you want supporting memories, without an answer:
memories = client.retrieve_memories(user_id, information_needed)
# Alternatively, ask Tomte to generate an answer from memory:
answer = client.answer_question(user_id, question)

# Separate reads when the application needs structured data:
client.get_profile_entities(user_id)
client.get_most_mentioned_entities(user_id, limit=5)
client.get_relevant_entities(user_id, message, limit=5)
client.get_pending_messages(user_id, session_id=session_id, limit=20)
```

Storage returns a queue receipt; the background worker extracts memory later.
`get_context()` returns one text string with relevant stored memories and pending
conversation, ready to pass to the chatting agent as background information.
Use `get_relevant_memories()` when structured results or retrieval options are needed.
Profile, most-mentioned and pending reads make no model calls. Relevant retrieval
uses embeddings but no generative LLM. `retrieve_memories()` embeds the information
request and ranks stored fact snapshots by their existing plain-value vectors.
An LLM reads batches of 20, up to 100 candidates, selecting useful evidence and
scoring topical relevance. There is no catalogue navigation. Reading can stop
for a satisfied specific need or a low-relevance tail; finding some list items
does not establish that a collection is complete.
`answer_question()` first analyses the question, then requests focused information
through that retrieval workflow, for at most three passes. The answering model
receives selected information and coverage notices. Fact reading uses the existing
OpenAI `answer_reader_model` setting (default: `gpt-6.1-terra`). See the
[workflow boundaries](docs/fact-retrieval.md) for the internal interfaces.
Neither operation stores its request or generated output.
Combined retrieval lazily embeds pending windows and reuses their cached vectors.
Queue acknowledgement automatically removes affected cache entries; TTL is a
fallback. Pending search exposes coverage limits separately from returned memory.
Existing `send_user_message` / `log_assistant_response` methods remain available
for callers using the combined chat hook described later in this document.
See the [API contract](docs/memory-read-api.md) for return shapes, history,
pending batches, failure behavior and provider requirements.

## Local development

For a small runnable chat interface, see the [Tomte V1 local demo](docs/demo.md).
After local infrastructure/bootstrap, run `uv run python -m demo.app --profile playground`
and open `http://127.0.0.1:8765/`. It uses the existing configured real host/provider,
manual consolidation with the daemon disabled, separate sessions, and an active
memory inspector. Use synthetic data only; this is a trusted-local demonstration.

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

OpenAI embeddings default to `text-embedding-3-large` with an explicit
1,536-dimensional output. This uses the larger model while retaining the
existing PostgreSQL vector columns and IVFFlat dimension limit. Set
`OPENAI__EMBED_MODEL` and `OPENAI__VECTOR_DIMENSION` together when overriding it.
The provider applies the selected dimensions to both queries and documents.
Switching models changes the vector space even when the dimensions match:
re-embed existing entity summaries, fact/history fields, and dictionary anchors
before using the new query model. Stop workers during the switch and drain or
rebuild cached/queued vectors as well. Bootstrap does not migrate vectors.
See [embedding model rollout](docs/embedding-model-rollout.md) for details.

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

Supported statements whose particular subject cannot be identified can be retained
in `unresolved_memories`. Later user clarification can bind a pending statement to
an entity through the existing mutation planner; the target update and resolution
marker commit together. Pending statements remain separate from ordinary entities
and are returned with explicit attribution warnings by `get_relevant_memories()`
and the chat context. See [memory design](docs/memory-design.md) and
[read API](docs/memory-read-api.md) for lifecycle, candidate limits and API usage.

Queue messages receive stable UUIDs. A worker freezes one Redis snapshot per
session; later messages remain in the pending queue. The ordered item identities
and tenant/user/session define a stable batch key, independent of message content.
PostgreSQL claims that key in `tomte_processed_consolidations` in the same
transaction as memory and search-index writes. Optional dictionary feedback runs
after commit in a separate transaction. A committed replay succeeds
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
from earlier aggregate-summary evaluations. Search now also matches individual
current/history items, so activation quality needs a new model evaluation;
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
Successful chat payloads additionally expose `query_relevant_memories` and
`background_memories`: public entity views with canonical UUIDs, current values,
selected history and confidence; internal row metadata is excluded. Query candidates keep raw
semantic rank, followed by unique current-message lexical hits. Background keeps
persona, recent and session-only order, excluding entities already in the query
list. Session-read payloads win overlaps, preserving canonical DB identity.

`context_text` defaults to labelled rows with one column guide, followed by
`QUERY-RELEVANT MEMORIES`, `BACKGROUND/CONTINUITY` and any separately labelled
`UNRESOLVED MEMORIES`. Current values and selected history remain distinct.
`MEMORY_TEXT_FORMAT=compact_json` switches to compact JSON sections. The structured
lists remain available independently of the text format. Send only `context_text`
as the memory input to an LLM to avoid duplicating those lists. See the
[memory read API](docs/memory-read-api.md) for per-call options and history limits.
Existing history/message/user fields remain;
guest and assistant success payloads have empty memory lists, and errors retain
their error-only shape. Hosts can keep using the text or consume the new lists;
exact-text/order parsers must account for the new headings and order.

Live semantic retrieval unions the closest five summary entities with the closest
five distinct entities by plain-text fact-field cosine, returning up to ten unique
entities. Overlap is deduplicated; matching fields cannot crowd out other entities.
There is no similarity cutoff, and entity-ID/DB-ID ties are deterministic.
Age and mention frequency do not
affect this ranking. Session activation remains separate: exact name/alias or
cosine at least 0.25, with the existing caps and TTL. Frequent retrieval and
weighted semantic APIs remain available for comparisons and other callers; their
statistics and formulas are unchanged. Raw cosine is not universally superior,
and ungated retrieval still adds irrelevant context on unrelated questions.

Transient raw-semantic unavailability can fall back to the other streams.
Recent retrieval retains its existing strict failure behavior. Malformed rows,
programming/security errors, and failed outer transactions still raise.

## Read-only memory search

Use `client.get_relevant_entities(user_id, message, limit=5)` to retrieve public
entities from an arbitrary message without ingesting it or generating an answer.
See [the read API contract](docs/memory-read-api.md) for usage, history limits,
embedding cost and error behavior.

Caller retrieval uses a separate plain-text current/history field index;
mutation candidate search retains its labelled field vectors. New writes update
both atomically. Populated installations must backfill the derived caller index
with `admin.rebuild_retrieval_index(user_id)` after bootstrap and before serving
the new retrieval. See the read API's rollout instructions.
