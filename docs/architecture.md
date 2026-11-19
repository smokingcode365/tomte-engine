# Engine architecture

Tomte separates host-facing APIs, model decisions and persistence. A host supplies
providers and owns authentication, service credentials and application policy.

## Message to memory

1. `clients/chat_client.py` validates the request. Redis stores conversation
   history, pending source messages and scheduling activity. Storage alone does
   not establish that a durable memory was extracted.
2. `workers/runner.py` selects an eligible session and claims a per-user lease.
   It obtains independent operation resources and a PostgreSQL user guard.
3. `workflows/consolidation.py` freezes a source identity, checks an existing commit
   marker, and coordinates extraction, entity resolution and fact mutation.
4. Core models validate proposed references and provenance; dictionary alignment
   supplies established terminology without deciding subject identity.
5. PostgreSQL commits entity JSONB, search-index changes and a batch marker in one
   transaction. Optional dictionary feedback uses a later transaction.
6. Redis acknowledges the exact frozen snapshot. If acknowledgement fails after
   commit, recovery reads the marker and acknowledges without repeating model work.

The managed runtime owns an independent database connection, lazy model client
and event loop per operation. Redis uses its shared connection pool. Host-owned
template providers remain the host's responsibility. See [runtime ownership and
failure handling](managed_runtime.md) and [SQL transactions](postgres_transactions.md).

## Retrieval

The normal conversational path combines identity/profile, recent memory,
summary/field vector search and session continuity. The result separates
query-relevant and background memory. Explicit read APIs can also include
unresolved statements and bounded pending-message windows.

Current and historical values have separate search representations. Mutation
candidate vectors retain labelled context; caller retrieval has a plain-text
index. BM25/hybrid retrieval is an existing opt-in mode, requiring its derived
lexical index. LLM-assisted retrieval reads fact candidates; answering plans
focused requests and has a bounded number of passes. These paths are described
in [the read contract](memory-read-api.md), [fact retrieval](fact-retrieval.md)
and [BM25 retrieval](bm25-retrieval.md).

## Module boundaries

- `core/`: schemas, identity, fact/history invariants, projections and retrieval.
- `workflows/`: orchestration across model and storage interfaces.
- `providers/`: concrete database, Redis and model adapters.
- `database/`: interfaces and SQL operations/schema.
- `dictionary/`: terminology alignment and dictionary maintenance.
- `llm/`: model interfaces, prompts, embedding input and structured-output recovery.
- `clients/`: trusted host SDK; `workers/`: scheduling, leases and retry control.
- `demo/`: checkout-only local UI; `scripts/` and `evals/`: opt-in evaluation tooling.

## Deliberate trade-offs

JSONB keeps each entity's facts and history together; derived indexes support
search. Larger entities can increase model input and read cost. Immutable history
and optimistic expected-payload checks favor correctness over silent overwrites.
PostgreSQL and Redis have separate durability responsibilities. A processed marker
prevents normal batch replay but does not replace backup, complete erasure or
recovery from inconsistent restored snapshots.

Python validates structural authorization and operation mechanics. LLMs decide
meaning, identity ambiguity and semantic relevance. Those decisions require live
evaluation in addition to deterministic tests; schema validity does not prove truth.
