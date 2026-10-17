# Extraction and reconciliation input contract

## Before this change

Redis consolidation items already contained `item_id`, `session_id`, `tenant_id`,
`user_id`, role, content, Unix timestamp, and an optional embedding. Their stable
processing snapshot supports replay. Live history contained role/content/timestamp,
without source IDs; it retained the last 20 turns by default with a sliding 24-hour
TTL. History is session-scoped by default; disabling session isolation intentionally
shares history across the user's sessions.

Extraction received only `role: content` for the current batch and dictionary terms.
No antecedents or source metadata reached the model. The database skeleton contained
UUID, canonical ID, name, domain/type, and the full payload; reconciliation discarded
aliases and fact values. Context formatting emitted canonical ID/type and fact
values/importance, but omitted names, aliases, status, and timelines.

## Current batch and antecedents

Chat Hook passes at most four preceding turns to enqueue. Redis persists that
reference-only snapshot with the queued item. Four turns cover the observed
purchase/paint/nickname/sale chain without sending the 20-turn window or whole graph.
Consolidation uses the first item's snapshot, so later messages and retries cannot
change the antecedents for that frozen batch. Legacy queues without a snapshot use
empty antecedents rather than guessing from changing live history.

Extraction input separates `CURRENT_BATCH` (actual IDs, roles, content, source times)
from `RECENT_CONTEXT_REFERENCE_ONLY` (roles/content). Only current messages may
produce new/repeated/updated facts. Historical turns exist for interpretation only;
prompt separation cannot mathematically guarantee model compliance.

Extraction does not receive Golden Record candidates or decide canonical identity.
Four turns are insufficient for cross-session/long-gap nicknames; narrow candidate
retrieval is a possible later step requiring evaluation, not implemented here.

## Provenance

`ExtractionMemorySchema` reuses domain semantic fields but excludes fact/event
timestamps. One-source results may omit the source; code assigns the actual queue
ID. With multiple sources, every fact/event must select an ID in the current batch.
Code validates selections and copies each source's actual timestamp as UTC seconds,
then constructs an internal resolved schema retaining ephemeral interpretation context.
See connected_entities.md for the tmp_ wire and canonical UUID relationship boundary.

Missing/invalid queue timestamps, unknown/invented/historical source selections,
and model-supplied timestamp fields fail before memory writes or processed markers.
Timeline timestamps represent when the source message stated the event, not an
inferred historical occurrence time. Inherited placeholders also block a merged write; old metadata is never automatically fabricated or repaired.

## Reconciliation and host context

Identity resolution now uses a bounded tenant/user-scoped candidate query per
extracted object, not the whole graph. Current message vectors are reused for
ranking; exact normalized names/aliases and active context rank before similarity.
The resolver receives current messages, at most four reference turns, semantic
objects, and at most five compact candidates per object. See identity_resolution.md
for target-set confidence gates and conservative group expansion.

Extracted IDs correlate results only. NEW, MATCH_ONE, MATCH_MANY and UNRESOLVED
are separate from deterministic INSERT/MUTATE/REINFORCE classification. Missing or
weak resolution never defaults to INSERT. Unknown target UUIDs fail before writes.
Candidate payloads expose name/domain/type/aliases and current values without
embeddings or historical timelines. Reference turns are never reapplied as facts.

Normal host context exposes name, type, aliases, fact values and status without
full timelines. Retrieved rows carry a JSON-safe DB UUID and namespace; context
merging deduplicates this identity, preserving distinct rows with corrupt logical-ID
collisions. Flat non-database entities retain logical-ID merging.

## Deterministic identity and state safeguards

Persistence serializes INSERT ID allocation per tenant/user with a transaction-scoped
advisory lock. NEW workflow identities use Python-owned UUIDs and independent logical labels;
collisions receive a UUID suffix and operation/payload IDs agree. Extraction labels
never persist. Canonical relationship UUIDs are unaffected by logical collisions. Matched MUTATE/REINFORCE retain the existing identity.
Bootstrap adds a scoped unique index; historical duplicates fail bootstrap clearly
without deleting or rewriting data. External writers must respect the same uniqueness.

The semantic dictionary has no durable synonym-to-canonical mapping: maintenance
removes losing keyword strings. A conservative explicit alias registry normalizes
ownership, vehicle_ownership and ownership_status to ownership_status. Unknown
attributes are unchanged. The production workflow no longer merges attribute slots
or converts sale wording into current ownership deterministically: mutation planning
proposes assertion transitions, and deterministic application validates and preserves
assertion identity, evidence and history. The older normalize_facts/merge_facts helpers
remain covered by legacy tests but have no production callers. They are not the current
mutation policy. See assertion_lifecycle.md for the active contract.

## Dictionary and compatibility

Feedback stores entity types as ENTITY terms and attribute names as ATTRIBUTE terms.
Calling these taxonomy keys and value lexicon is misleading; naming cleanup is
recommended separately. No dictionary retrieval/feedback behavior changed.

The cache enqueue API adds optional `recent_context`; third-party cache implementations
must accept it. Custom DB implementations must supply get_resolution_candidates for consolidation.
LLM implementations must honor the requested output schema, including ResolutionResponse. Existing
OpenAI direct domain-schema requests remain supported; the workflow requests the new
wire schema. Other providers receive the schema override but were not live-verified;
their pre-existing `.data`/SDK compatibility issues are outside this change.

Deterministic tests cover prompt separation, stable scoped history, aliases/values,
single/multiple sources, failure rejection, readable context, and existing real-local
PostgreSQL/Redis identity/replay/transaction/vehicle mechanics. The earlier real-model comparison passed same-batch continuity 3/3 and cross-batch
2/3, exposing logical-ID collisions and contradictory ownership keys. Subsequent
controlled evaluations exercised the current mutation and contextual fallback paths;
these small synthetic samples do not establish general accuracy. No external API
calls run in the regression tests.

Dictionary feedback creates/nudges entity-summary anchors and increments keyword
hits. Bootstrap queries message-topic centroids at cosine similarity >0.75. First
batches are cold; subsequent matching-vector tests return learned terms, while
orthogonal vectors do not. Historical real-call vectors were not retained, so the
exact reason each later real call missed terms is not confirmed. Threshold/embedding
representation mismatch remains a hypothesis; no dictionary thresholds or prompts
were changed.
