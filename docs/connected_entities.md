# Connected entity identity contract

Extraction uses unique `tmp_` handles across all eight domain lists. They are
correlation only, never persistent IDs. Historical stored `entity_id` strings are
unchanged. NEW workflow entities receive independent Python UUID4 row identities
and `entity_` UUID labels before any links are normalized. UUIDs are attempt-local:
rollback may allocate new ones on retry; after commit the batch marker prevents a
second allocation/write pipeline. No extraction suffix or name is an identity.

Python builds a handle-to-single-UUID map for accepted NEW/MATCH_ONE outcomes.
Multiple handles securely matching one row accumulate one target plan. MATCH_MANY
is a selection, not a stored entity or a link endpoint. It needs confident offered
members, a nontruncated set, equal applicability and a current-user group quote.
Known participant handles must map to exactly that selected set. An unresolved
member causes the entire selection to abstain, not arbitrary partial fanout.
Membership meaning remains a model judgment; quote membership is not semantic proof.

Extraction `evidence_context` (600 characters) and `mention_context` (400) are
optional untrusted interpretation aids. Facts retain current-user source IDs and
system timestamps. Current/reference messages remain available; these contexts
are not persistent summaries or new evidence. Mutation inputs preserve per-fact
context, canonical endpoints and group scope. Each resolved UUID has its own
current assertion state, mutation decisions and persistent assertion IDs.

New JSONB facts use `related_entity_uuids` for canonical row references. Historical
`related_entity_ids` remain opaque legacy annotations; they are never auto-mapped,
even if a string looks like a UUID. New extraction handles in that field are
rewritten and cleared before persistence. No graph traversal or semantic retrieval
algorithm is added. Raw row retrieval retains both fields; formatted context labels
canonical UUIDs explicitly and only flags old strings as unvalidated legacy annotations.
The demo projection exposes canonical UUIDs.

`incoming_fact_key` associates ADD/replacement proposals with offered semantics.
A unique exact source/attribute/value template is a compatibility fallback. Linked
or rejected dependent facts cannot use an ambiguous/missing template. Unresolved
endpoints drop affected incoming facts, preserving independent facts when safe.
An explicitly introduced NEW node without facts is retained only when an accepted
edge needs it; no redundant name fact is fabricated. If a NEW endpoint with rejected
proposals or one group member is later abstained by mutation, dependent
prepared operations are discarded to a fixed point. This conservative operation-
level fallback avoids committing historical transitions without their replacement;
it may omit otherwise independent facts within that operation. Other operations
survive. No optional feedback queue or extra LLM stage is added.

The PostgreSQL write transaction reserves all INSERT UUIDs first, validates every
canonical relationship endpoint against same-transaction inserts or same-tenant,
same-user rows locked FOR KEY SHARE, then writes memory. Logical-label collision
allocation cannot change row UUID references. Memory and processed marker remain
atomic. Optional dictionary feedback stays post-commit. Existing scoped optimistic
snapshot checks remain. Direct SQL-provider callers may retain legacy logical labels,
but cannot introduce new opaque relationship annotations or tmp_ identity labels.
Custom providers implementing commit_graph_updates must enforce the same endpoint
existence/scope/transaction contract; a Python model schema alone cannot do that.

No automatic historical relationship migration is performed. Historical annotations
can be retained unchanged during updates, but new canonical references require
explicitly resolved endpoints. F3 temporal semantics and F5 routing remain unchanged.
Deterministic tests establish transport/persistence correctness, not real-model
reference accuracy. No paid model evaluation is part of this change.
