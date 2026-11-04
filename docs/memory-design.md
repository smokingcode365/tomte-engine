# Entity and fact memory

An entity belongs to one tenant/user and its `entity_id` is its PostgreSQL row UUID.
Its facts have stable application-generated UUIDs. An entity is a distinct subject
directly involved in the user's life, useful to remember in its own right. A fact
describes that subject; its value fields hold the details. Direct relevance does
not make every noun or property an entity: a car's colour is a fact about the car.

The user's two distinct jobs or cars normally have separate entities. A brother's
two jobs can instead be two employment facts on the brother entity, even with the
same `attribute_name`. Employer and workplace details do not require separate
entities unless they have their own supported direct relevance to the user. Detail
alone does not create an entity. Value changes preserve the same subject's identity.
Activities, events and emotional experiences can also be subjects; anxiety about
an interview can describe that event. These examples are not a fixed taxonomy.

Extraction, identity and mutation share this definition from
`llm/prompts/memory_subjects.py`. The requirement that all facts apply equally to
multiple targets governs shared updates; it does not prohibit distinct facts
belonging to one subject. The LLM chooses subjects by meaning in any language.

Entity names prefer durable supported names or labels. Mutable descriptions such
as colour belong in facts: use `Honda Civic` rather than `blue Honda Civic` when
blue is only its current colour. Genuine user-established nicknames such as
`Bluey` remain valid. Names are not unique identity keys; UUIDs and user evidence
distinguish subjects. Current updates retain existing canonical names; this policy
does not rename or repair previously stored descriptive names automatically.

`attribute_name` is a category, not a unique fact selector. Facts remain a list,
and mutations, history and search-entry identity use `fact_id`. Dictionary field
vocabulary deliberately shares the context `[domain, entity type, attribute name]`
across separate facts. The legacy `find_exact_fact` database method only checks
whether an entity contains that category and returns the entity row UUID; it must
not be used to select a fact. Current consolidation does not call it.

Each value is a map of named items. An item contains `text`, `fact_confidence`
(`high`, `medium`, `low`), `_recorded_at`, and `_updated_at`. Business content is
always text, including uncertain dates and qualified quantities. Confidence
describes support in the user's statements, not independently verified truth.
The LLM determines meaning and vocabulary; Python validates structure, scope,
IDs, offered targets, current-user provenance, and deterministic operations.

Facts retain `_importance`, `_mention_count`, `_last_mentioned_at`, and immutable
`history_values`. Each history entry contains the full previous `value` and
`_archived_at`. Recording timestamps are not event dates. A real change archives
the previous value; a correction or enrichment updates current information
without inventing a historical event. History cannot be edited in this version.
Historical enrichment appends a fuller entry with the original items preserved.

Partial updates preserve omitted fields; replacement removes omitted fields.
Explicit removals cannot leave an empty current value. An ended employment uses
descriptive text within its value, retaining enough context to understand what
ended. Questions may count as mentions without asserting their presuppositions.
Counts deduplicate current message IDs per fact within an atomic, replay-safe
batch. Source references authorize operations transiently and are not memory.

LLM wire values use lists of fixed-shape field entries, converted to maps only
after checking duplicates and reserved keys. The application owns UUIDs and
metadata. Public memory uses an explicit allowlist of identity, text, confidence,
and selected history; internal metadata is omitted. `history_is_partial` tells
the reader when any stored history is omitted. Storage retains all history.

## Public example

```json
{
  "entity_id": "f96eaad7-9362-4394-b971-6efa1aa85039",
  "entity_name": "Bob",
  "entity_domain": "relationships",
  "entity_type": "person",
  "aliases": [],
  "facts": [{
    "fact_id": "8b935a2e-1e12-469c-8f15-6f3e48929e43",
    "attribute_name": "employment",
    "value": {
      "job_title": {"text": "researcher", "fact_confidence": "high"},
      "employer_name": {"text": "Cedar Labs", "fact_confidence": "high"},
      "workplace_address": {"text": "25 Pine Avenue", "fact_confidence": "high"}
    },
    "history_values": [],
    "history_is_partial": false
  }]
}
```

Changing the workplace address updates that item on the same fact UUID. A real
move snapshots the entire old value. Correcting a mistaken address does not.
An ended job can add `end: {"text": "No longer working at Cedar Labs",
"fact_confidence": "high"}`. Current values describe the known state, including
endings; callers must read the text instead of treating every employment as active.
There is no stored source, relationship edge, note, lifecycle flag or schema version.

## Workflow and safeguards

Extraction receives current messages with their original timestamps and up to
four frozen reference turns. Ambiguous duplicate temporary handles are rejected.
Explicit group participants must all be present; a fact with missing participants
is omitted without synthesizing an entity. Independent facts survive. Ordinary fact values have no entity
references to resolve, so a missing Cedar Labs object cannot drop Bob's employer.

Identity resolution offers at most five scoped candidates per object. NEW needs
an explicit supported introduction; matches need an offered UUID and accepted
identity evidence. Existing identity thresholds remain separate from item
`fact_confidence`. Unknown identities abstain. Shared updates require all members,
with no partial fanout or inferred replacement targets.

Contextual routing runs for remaining references in mixed batches as well as
empty extraction. Quotes must occur in current user messages, and resolved entity
and fact UUIDs must be offered together. Mutation receives at most 50 facts per
target, prioritizing contextual targets, with the latest 15 history entries per
fact. Older unoffered history cannot be enriched by index; this is a recall limit.
The model may return an empty plan; no deterministic fallback forces a write.
Removing relationship references fixes that failure mechanism, not all possible
model omissions or incorrect semantic decisions.

Python enforces shape, source role, offered identities, group completeness,
immutable history, snapshot freshness and transaction scope. It does not classify
correction language, detect business synonyms or repair semantic content.
Stored memory/query content is untrusted data to the model. Structural checks
cannot prove that a model's interpretation is true.

## Proposal recovery

ADD creates a fact; UPDATE selects an existing offered fact. Both apply structured
text fields. `update_mode` and `change_type` only affect UPDATE. Recognized values
in those controls are ignored for ADD, MENTION and APPEND_HISTORY. Exact repeated
field entries, sources and removal names are deduplicated. Conflicting duplicate
field values remain invalid; Python never chooses their meaning.

Extraction, identity, contextual and mutation responses isolate malformed list
items. Built-in providers allow at most one native output-repair response per
call. Repairs may replace rejected items only, at their original positions, with
their original correlation/target keys and operation kinds. Accepted first-response
items cannot be rewritten by a repair. A malformed repair preserves independently
valid first-response items. An explicit empty plan is accepted without repair or
forced writes. Custom LLM adapters must enforce the same bounded repair policy;
the workflow does not add another provider call.

After structural parsing, unsupported sources, unoffered targets and ambiguous
identity mappings are rejected locally. Unknown UPDATE targets never become ADD.
Invalid optional aliases do not discard a supported identity, but every retained
alias still requires offered spelling and current-user source evidence. Duplicate
identity decisions or target plans are ambiguous and cannot authorize a write.

Mutation operations sharing a fact UUID or incoming proposal key form a dependent
unit. If one fails, that unit is omitted. Invalid replacement fields reject the
whole operation; partial replacement salvage could silently erase stored fields.
An explicit shared group requires every participant's accepted claim. Failed
groups remove their dependent units, then surviving operations are rebuilt from
the original stored snapshots. This repeats for dependent groups, preserving
unrelated facts on the same entity and preventing rejected history/mention changes
from leaking into the commit. New fact UUID generation uses the original operation
position, even when preceding items are rejected.

Rejected-item/repair counts are transient diagnostics, not stored memory. A fully
unusable proposal response after bounded repair can become a completed no-op for
that stage. This deliberately accepts some memory loss; it is not a provider outage
retry mechanism. Transport, cancellation, capacity, corrupt stored data, transaction
and snapshot failures still propagate. All surviving writes and the batch marker
remain atomic, with Redis ACK only after commit. These mechanics do not establish
semantic accuracy; that requires separate real-model evaluations.

## Dictionary and search

The dictionary first aligns ENTITY and ATTRIBUTE terms, then FIELD terms under
the selected `(domain, entity_type, attribute_name)` context. Only exact semantic
equivalence permits REUSE. KEEP_RAW is valid; no hardcoded business alias map
exists. Field renames cannot collide, and item text/confidence stay unchanged.
Embeddings are batched in groups of 16. Any alignment failure keeps the complete
raw extraction and discards speculative feedback. Maintenance can merge only
compatible keyword roles and parent contexts.

PostgreSQL JSONB is authoritative. `tomte_memory_search` is a derived index of
every current and historical item, including low-importance content. Long items
are chunked into bounded search text. Search uses the best matching item or
aggregate entity summary within tenant/user scope. Semantic writes atomically
update index deltas; metadata-only mentions do not embed again.

Small histories are returned completely. Larger histories use scoped semantic
ranking and bounded model selection (40 candidates, 20,000 candidate bytes,
12,000 returned history bytes by default). Recording recency is the fallback.
Coverage may omit an earlier snapshot only when the model identifies the same
episode and exact item/confidence containment preserves its information. Every
omission sets `history_is_partial` on that fact. Different episodes and uncertain
details must not be merged. Retrieval never edits stored history.

## Persistence and rollout

The entity ID equals the row UUID in both SQL and JSONB. A prepared NEW write
reserves that UUID; a collision rejects the write rather than renaming identity.
Updates compare the expected stored payload under scoped locks and retain fact
UUIDs and history prefixes. Memory, search-index changes and the processed-batch
marker commit together before Redis ACK. Optional dictionary feedback uses a
separate transaction; its failure must not replay committed memory.

This is a breaking memory contract. There is no automatic conversion of old
memories or queues. Bootstrap adds the search table, dictionary parent context
and a canonical UUID constraint without deleting data. The constraint initially
uses NOT VALID so old rows can be inspected before an explicitly authorized reset
or migration; it still applies to new writes. Validate it after old data is gone.
Stop writers before clearing obsolete test memories and their scoped queues,
processing snapshots, history/persona/session caches and replay markers. Preserve
the vocabulary dictionary and unrelated service data. Never use FLUSHDB or drop
volumes as an implicit upgrade step.

Custom database providers need history-candidate retrieval and atomic index-delta
writes. Custom LLM providers must support the new extraction, mutation, field
alignment and history-selection schemas. Public API/context/demo projections
share one allowlist; recursive underscore filtering is not the security boundary.

Offline tests prescribe model decisions and verify mechanics with mocked HTTP
and opt-in local PostgreSQL/Redis fixtures. Prompt-evaluation v2 fixtures cover
the new wire shape. Historical fixtures remain unchanged. No live-model accuracy
or server schema-acceptance guarantee follows from those deterministic tests.
