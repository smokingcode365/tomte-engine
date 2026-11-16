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

## Thoughts and plans

The ninth domain, `thoughts_and_plans`, uses the same entities, fact UUIDs, values,
history, dictionary and retrieval paths. It represents a specific idea, intention,
tentative plan or recommendation relevant to the user. It does not establish that
an imagined object exists or an action happened. A car purchase intention has a
descriptive name/type and qualified text such as "Considering buying a Tesla" and
"If buying a car, would prefer blue". Current possessions, activities and confirmed
arrangements retain their usual domains. Another person's thoughts can remain
attributed facts of that person; assistant-only advice is not user evidence.

Extraction also preserves specific advice the user gives during a personally
relevant exchange, including its details and known recipient. It stays attributed
advice, not evidence of either person's actions, health or adoption. A request
such as "Can I join you sometime?" can express a wish worth remembering without
establishing acceptance or participation. The LLM interprets these meanings from
context; Python does not classify them using language rules.

Thought entities require public `first_mentioned_at` in UTC ISO format, for example
`2026-01-05T10:00:00Z`. Python derives it from the earliest original current USER
message cited by an accepted mutation. Rejected operations and failed groups do
not affect it. It remains unchanged on later updates/mentions, except that a
later-processed older supporting message can move it earlier. This is the earliest
supporting message available to Tomte, not when the idea arose, processing time,
or confirmation that it remains active. Source IDs are still transient.

Prompts retain actual timing and conditional meaning in value text. A later question
may count as a mention without renewing intent. Changes or abandonment keep the
same entity/fact identities and use existing history mechanics. Explicit completion
can separately establish an actual subject and update the identified plan's outcome;
it never converts the thought UUID/domain into the actual object or supplies desired
details as facts. No automatic expiry or lifecycle flags are introduced.

For known incoming domains, candidate retrieval separates thoughts from non-thoughts
before the candidate limit; identity acceptance also enforces this boundary, including
groups. Existing cross-domain matches among the original eight domains remain possible.
Untyped contextual lookup and general memory retrieval can still return both kinds.
Domain/time precede long labels in thought search text. The public projection returns
`first_mentioned_at` for thoughts and continues to hide internal item metadata.

This addition needs no SQL schema migration: the date lives in JSONB and domains
are stored as text. Existing ordinary entities remain readable without a new field;
their index text is unchanged. Old memories are not automatically reclassified.
Custom candidate providers should apply the thought boundary before their limit,
while preserving untyped contextual searches. Workflow acceptance also checks it.

Every valid batch containing user messages reaches extraction without a separate
junk-filter LLM call. Extraction may return no facts; existing contextual-update
and processed-batch replay handling still apply.

The extraction prompt asks the LLM to review supported details before returning,
including concrete examples and outcomes that explain the user's interests or
experiences. These can remain facts about an existing topic. This coverage check
preserves ownership, uncertainty and negation; it does not require nonempty output.

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
(`high`, `medium`, `low`), optional `context`, `_recorded_at`, and `_updated_at`. Business content is
always text, including uncertain dates and qualified quantities. Confidence
describes support for the complete qualified claim, not independently verified truth
or the likelihood of a future action. A directly reported tentative intention can
have high confidence; its uncertainty about acting must remain in the text.
The LLM determines meaning and vocabulary; Python validates structure, scope,
IDs, offered targets, current-user provenance, and deterministic operations.

Extraction and mutation preserve original wording where possible, changing only
what is needed for independent understanding. A short `context` supplies necessary
supported background separately, for example text `We played chess afterward.`
with context `We refers to the user and Sam; after their tournament.` It is not
a source transcript, inferred explanation, or extra event. Qualifications stay in
the claim. Readers interpret text and context together; context is public content.
The LLM should omit unnecessary context and preserve unaffected wording on updates.

On the mutation wire, omitted/null context preserves existing context; an empty
string explicitly clears it. The LLM must recheck context when text changes.
Empty context is omitted from stored/public JSON, preserving old memory shapes.
Real changes archive the complete old value including context; historical context
cannot be corrected. Plain-text search inputs include the text and, when present,
its short context separated by a newline. Context-only edits refresh affected
search vectors. Old memories without context keep their previous embedding text.
No SQL schema migration or automatic rewriting/backfilling of old memories occurs.

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
four frozen reference turns. Each current message also has a readable UTC
`message_time_utc`, derived from that same timestamp and passed through the model
stages as transient context. The extraction and mutation prompts anchor relative
event times within the value text so public memory remains understandable later.
For a message dated 13 September 2026, "this week" can become "during the week
containing 13 September 2026". Vague timing keeps its uncertainty; recurring
schedules remain recurring. Explicit local dates/timezones take precedence over
assuming the user is in UTC. Later mentions must not redate earlier events.
Unknown original dates are not guessed from processing/mention timestamps, and
existing history is not rewritten. Python formats timestamps, without interpreting
date language or adding semantic rejection rules. No stored/public field or
additional model call is introduced. These semantic rules require model evaluation.

Ambiguous duplicate temporary handles are rejected.
Explicit group participants must all be present; a fact with missing participants
is omitted without synthesizing an entity. Independent facts survive. Ordinary fact values have no entity
references to resolve, so a missing Cedar Labs object cannot drop Bob's employer.

With the OpenAI adapter, identity resolution retrieves up to twenty scoped
candidates per object, filters them with the configured candidate-filter model, then asks
the strong model to decide identity. Each extracted
object is embedded separately as plain text: its name, aliases and every current
fact-field text, without generated type/domain/attribute labels or internal metadata.
These queries use the same embedding provider as stored vectors and are batched.
Stored summary and field embeddings remain unchanged; no reindex is required.
Candidate order is exact incoming name/alias match, descending best summary/field
cosine, then ordinary domain equality as a tie-break. Names mentioned elsewhere
in the chat do not receive priority. Tenant/user and thought/actual boundaries
remain enforced. Contextual and unresolved lookups still use the batch-message
vector, with the shared ranking and ten-candidate limit. A fixed limit cannot
guarantee all relevant entities, especially for larger groups.

The candidate filter sees one extracted object, the current conversation, the
four-turn reference window and candidate current facts. It uses request-local
labels (`c1`, `c2`, etc.) and returns only `keep`/`drop` lists. Python verifies an
exact partition of the offered labels and maps survivors back to existing UUIDs.
These labels are never persisted. Every surviving candidate reaches the strong
resolver, including more than ten survivors; there is no second top-k cutoff.
Ambiguous current and historical identities must be kept for the strong resolver.
This is model judgment, not guaranteed recall: a valid filter response can still
drop a plausible identity. A single survivor never proves identity.

Malformed output, unknown/duplicate/missing labels or an ordinary OpenAI provider
failure preserve that object's original candidate set. The filter has no model
format-repair retry; cancellation, input-capacity limits, usage/budget stops and
host-code errors propagate normally. Empty candidate sets require no filter call.
`last_candidate_filter` exposes per-object offered/retained counts and outcomes.
The resolver also receives `CANDIDATES_FILTERED` and the original retrieval
truncation metadata. Filtering never authorizes an identity, NEW or a database write.

OpenAI routes `CandidateFilterResponse` to `candidate_filter_model` (GPT-6.1 Terra
by default), configurable with `OPENAI__CANDIDATE_FILTER_MODEL`. This applies to
standalone providers and managed operations. Dictionary alignment still uses
`cheap_model` (Luna); identity and mutation still use `strong_model` (Sol).
Restart running processes to load a changed model setting. Other built-in and custom adapters retain
the ten-candidate path unless they explicitly set `supports_candidate_filter=True`
and implement this schema through `parse_structured`. An opted-in custom adapter
may raise `CandidateFilterUnavailable` for a recoverable filter failure. Scoped
search constraints, dictionary behavior, stored embeddings, fact history,
contextual corrections and unresolved-memory processing are unchanged. No migration
or reindex is required. The extra filter call adds latency and only saves total
tokens/cost when the downstream input reduction offsets its cost.

The request
stores each candidate's identity and current fact values once in `CANDIDATES_BY_ID`,
keyed by its UUID. `CANDIDATES_BY_OBJECT` retains each object's ordered candidate
references and per-pair similarity scores. A record in the shared table is only
eligible for objects whose candidate list includes it; existing target checks still
enforce that scope. These references are transient request structure, not stored
entity relationships. Fact history and internal value metadata are excluded from
this identity input. The assembled-input size limit still applies: deduplication
removes repeated bodies but does not bound growth of unique current facts.

NEW needs an explicit supported introduction. The LLM decides semantic identity
and returns MATCH_ONE, MATCH_MANY, NEW or UNRESOLVED. Each selected target contains
only an offered `db_id` UUID. Python does not veto that decision using a confidence
threshold or evidence-label whitelist. Older response targets may include retired
`resolution_confidence` and `evidence` fields; these are discarded, not evaluated.
Value-item `fact_confidence` remains unchanged and has no identity-routing role.
This target contract also applies to contextual routing and later unresolved binding.
Candidate truncation is model context, not an automatic group veto: the LLM must
identify the complete requested group or return UNRESOLVED. Python still requires
shared-applicability approval, current-user source quotes, offered UUIDs and complete
group application, with no partial fanout or inferred replacement targets.

Explicit UNRESOLVED identity decisions retain extracted supported claims in
`unresolved_memories` for later clarification. A binding model may select an offered
target when a current user message supplies the missing identity. The earlier
claim and its report time then enter mutation; a binding alone neither overwrites
newer values nor marks the pending record resolved. Completion requires accepted
mutation and atomic persistence. Invalid/missing decisions do not force a write,
and bounded retrieval means later recovery is not guaranteed.

Contextual routing runs for remaining references in mixed batches as well as
empty extraction. The model also reviews offered facts whose factual basis clearly
changed, even without a direct mention. Already resolved proposals cover their
specific claims, not every consequence for other facts. A changed circumstance
can qualify an earlier situation but does not by itself end a feeling, preference
or intention. Hypothetical changes and another person's changes must not be
treated as changes to the user. This remains model judgment over offered candidates;
there is no automatic cascade or guarantee that every affected memory is retrieved.
Quotes must occur in current user messages, and resolved entity
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

Group application uses explicit transient references, not equality of human
wording. An operation's `incoming_fact_key` acknowledges that the complete incoming
claim is represented for that participant. ADD, UPDATE, APPEND_HISTORY and MENTION
can all fulfil that contract; the LLM decides whether the claim is new, historical
or already known. Equivalent wording and existing canonical attribute names are
allowed. Python validates references, sources and successful application, then
requires every group participant. This does not prove semantic correctness.

An independent contextual operation selects an offered `contextual_action_key`
instead of an incoming proposal key. Its existing fact UUID and supporting sources
must match that action. It cannot ADD or claim both reference kinds. Group-bearing
targets require an explicit proposal/action reference, so omitting a proposal key
cannot silently bypass group coverage. Sharing a chat source alone does not make
two operations dependent. Neither key is stored in entity JSON.

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

Tomte supplies entity memories as conversational background to a calling agent.
The current chat selects relevant entities; the caller uses their facts to
understand the user's situation and decide what matters in its response. Returning
the whole entity is intentional: a fact need not directly answer the latest
message to be useful background. Related jobs, colleagues, preferences and concerns
can all provide context for a workplace conversation. This contract does not call
for filtering current facts down to the minimum needed for a direct answer.

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

OpenAI now defaults to `text-embedding-3-large` with an explicit 1,536-dimensional
output for both queries and indexed text. The database dimensions and ranking
strategy are unchanged. Old small-model vectors are incompatible even at the
same dimension; see [embedding model rollout](embedding-model-rollout.md) before
switching an existing installation.

The entity-summary builder preserves its complete JSON text, with no artificial
character or byte cutoff. For the supported OpenAI embedding models
(`text-embedding-3-large`, `text-embedding-3-small`, `text-embedding-ada-002`),
the document embedding adapter uses `tiktoken` to shorten only inputs exceeding
the provider's 8,192-token limit. The UTF-8-safe prefix may end inside JSON; it is
embedded, not parsed back into memory. Queries are not silently shortened.
Other providers/custom model deployments retain their own input-limit behavior.
The separate operational request-byte capacity check still applies.

`tiktoken` is a direct dependency. Its public tokenizer assets must be cached or
downloadable on first use. Token counting is local, with no generative LLM call.
Complete current and historical values remain in JSONB and losslessly chunked
field indexes; the 2,000-character field chunk size does not discard content.

The larger summary applies when new embeddings are generated. Existing summary
vectors are not rebuilt automatically, and metadata-only mentions do not refresh
them. Use `rebuild_retrieval_index(user_id, refresh_summary=True)` to refresh the
summary and plain-field vectors together. This is opt-in and scoped to one user;
no schema migration is required for the summary change.

Small histories are returned completely. Larger histories use scoped semantic
ranking and bounded model selection (40 candidates, 20,000 candidate bytes,
12,000 returned history bytes by default). Recording recency is the fallback.
Coverage may omit an earlier snapshot only when the model identifies the same
episode and exact item/confidence containment preserves its information. Every
omission sets `history_is_partial` on that fact. Different episodes and uncertain
details must not be merged. Retrieval never edits stored history.

## Persistence and rollout

### Statements with unresolved subject identity

`unresolved_memories` reuses entity/fact storage for earlier user-supported claims
that identity resolution explicitly classifies as `UNRESOLVED`. It is an
application-managed domain, not a real-world subject, and extraction must leave
its collection empty. Unsupported guesses, malformed responses and technical
failures are not automatically converted into memories.

Each retained record has a UUID and one complete extracted fact, preserving all
its text fields and confidence qualifications. Multiple extracted facts become
separate records so resolving one does not discard the others. The original
subject name/type are labels, not identified entities. There are no permanent
temporary IDs or saved source chats. Storage adds:

- `unresolved_detail`: the resolver's explanation of the missing identity, or the
  protocol label `unresolved_subject_identity` if its explanation was empty.
- `_unresolved.original_domain`: the original subject domain, preserving the
  thought/actual boundary during later binding.
- `_unresolved.reported_at`: UTC time of the supporting original user message,
  distinct from processing time and any event date in the text.
- `_unresolved.status`: `pending` or `resolved`; completed records also carry
  internal `resolved_at` and `resolved_entity_ids`.

For example, “My car was painted red” can remain pending without changing a
specific car. Later, “The car I painted is my Tesla” can bind the earlier repaint
to an offered Tesla. This resolver runs even if the new extraction is empty. It
considers at most 5 semantic and 2 additional recent pending records by default,
and at most 10 stored plus 10 current-batch identities per record. These are candidate
windows, not completeness guarantees. Current-user evidence must identify the
same subject/episode; a lone candidate or matching words do not establish that.

The existing mutation planner receives the earlier claim, original report time,
and current identity clarification explicitly separated. Its keyed mutation must
represent the complete retained claim. A previously reported red repaint belongs
in history when a newer blue repaint is current. Binding alone, a missing plan,
an invalid mutation, or a partial dependent group cannot retire the pending record.
An otherwise valid mutation without its proposal key can still proceed, but cannot
retire the pending record; Python does not guess which claim it acknowledged.
Python checks scope, UUIDs, evidence references and accepted operations; the LLM
interprets attribution, chronology, completeness and semantic equivalence.

When a current clarification and one retained claim describe the same fact or
episode, the mutation prompt asks for one complete representation keyed to the
retained claim. Overlapping ungrouped current proposals do not require duplicate
facts. Original dates and qualifications must survive; distinct episodes and
independent details remain separate. Required group acknowledgements and other
retained-record acknowledgements are not waived. This is an LLM planning rule,
not Python deduplication or a guarantee that every model response avoids overlap.

The target mutation, pending-to-resolved transition, search changes and batch
marker commit in one transaction using expected snapshots. Retries cannot apply
them twice or reopen a resolved record. The earlier record's content is immutable
and remains stored internally. All ordinary entity queries exclude this domain
before limiting. A separate bounded pending query supports public retrieval;
its projection exposes the explanation and `reported_at`, not lifecycle metadata.

If old evidence exceeds the resolver or combined mutation input capacity, whole
pending claims are deferred; the current batch's ordinary work can continue.
No evidence is truncated. This version binds one target per retained record,
does not guess individual members of ambiguous groups, and has no expiry or
background scan of all pending records. No-match claims remain pending for a
future relevant message. These limits and LLM nondeterminism can still leave
memories unresolved; deterministic tests establish mechanics, not accuracy gains.

No SQL schema migration or reset is required: domains are text and lifecycle
data is JSONB. Existing memories are unchanged. Custom providers need the new
pending lookup, ordinary-read exclusions and atomic transition validation.
Deploy updated readers and writers together; older engine versions do not
recognize the new domain.

### Existing entity contract

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

Caller retrieval embeds field text plus optional short context in a separate derived index,
including current and historical values. It selects the top five entity summaries
and top five distinct entities by their best plain-field match independently,
then unions them by UUID without trimming back to five. The limit is configurable
per ranking. Mutation candidate search retains its existing summary/labelled-field
vectors. Legacy weighted/session discovery and unresolved-memory searches also
use plain field vectors, retaining their own limits, weights and pending-status
rules. New and changed memory updates both indexes in the same transaction.
Populated installations need a scoped plain-index backfill before serving this
route; see [memory read API](memory-read-api.md#side-effects-and-cost).

Custom database providers need history-candidate retrieval and atomic index-delta
writes for both indexes. Custom LLM providers must support the new extraction, mutation, field
alignment and history-selection schemas. Public API/context/demo projections
share one allowlist; recursive underscore filtering is not the security boundary.

Offline tests prescribe model decisions and verify mechanics with mocked HTTP
and opt-in local PostgreSQL/Redis fixtures. Prompt-evaluation v2 fixtures cover
the new wire shape. Historical fixtures remain unchanged. No live-model accuracy
or server schema-acceptance guarantee follows from those deterministic tests.
