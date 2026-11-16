# Message storage and memory reads

The primary synchronous Python SDK separates ingestion from retrieval. Initialize
Tomte with the application's providers, then create a tenant-scoped client:

```python
from tomte.engine import Tomte

client = Tomte.create_tomte_client(tenant_id="my-app")

# Only store messages; no memories or answers are returned.
receipt = client.store_user_message("user-123", "session-1", "I adopted Scout.")
client.store_assistant_message("user-123", "session-1", "What breed is Scout?")

# Background memory for the chatting agent, returned directly as a string.
context = client.get_context("user-123", "My dog Scout")

# Separate reads when the application needs structured data.
profile = client.get_profile_entities("user-123")
frequent = client.get_most_mentioned_entities("user-123", limit=5)
relevant = client.get_relevant_entities("user-123", "My dog Scout", limit=5)
pending = client.get_pending_messages("user-123", session_id="session-1", limit=20)
```

These methods require a user ID. The application must authorize that user within
the client's tenant. Invalid input and scope violations raise. Retrieval failures
propagate, except the documented optional pending-snapshot fallback below.
`retrieve_memories()` and `answer_question()` are the separate LLM-assisted
operations described below.

## Get context for the chatting agent

```python
context = client.get_context(user_id="user-123", message="My dog Scout")
```

`get_context(user_id, message) -> str` is the common retrieval method. It returns
one text string containing relevant entities, separately labelled unresolved
reports, and relevant unprocessed conversation. No dictionary access or unpacking
is needed. Pass this text to the agent as background data, not as instructions.

It delegates once to `get_relevant_memories(..., include_pending=True)` and returns
its `context_text`, using the configured limits and text format. It adds no model
call, changes no queue or memory, and reuses the existing pending-window cache.
The input message is a search query; store it separately if it should enter the queue.
Uncertainty, source-role labels and pending-search coverage notices remain in the
text. Even with no matching memories, it can contain pending-search status.

Use the separate `get_relevant_memories()` function below when application code
needs structured data, coverage fields or per-call retrieval options.

## Retrieve memories with LLM assistance

```python
memories = client.retrieve_memories(
    user_id="user-123", information_needed="Names and breeds of my dogs",
)
```

This returns supporting memory as one text string, without generating an answer.
Supply one focused description of needed information, such as "When I adopted
Lucky". The function asks an LLM to prepare embedding text without search
exclusions, retrieves one broad ranking of
up to 100 fact snapshots, reads them in batches of 20, and renders selected memory.
The filter retains the complete original request. It does not navigate a catalogue or plan
follow-ups. Use `answer_question()` when a question needs several dependent searches.
Current/history markers, entity identity, uncertainty, unresolved reports, pending
messages and coverage notices remain in the result. Nothing is ingested or mutated.
The existing `get_context()` API remains available without a generative model.

OpenAI fact reading uses `answer_reader_model`, default `gpt-6.1-terra`, configurable
with `OPENAI__ANSWER_READER_MODEL`. Other/custom providers retain their model
routing. One retrieval pass uses one short query-preparation call, zero to five
fact-reader calls and one query embedding. Query preparation uses
`retrieval_planner_model`; identical requests reuse prepared text within the same
workflow. A nonempty small set still receives one reader call. Pending-window
embeddings use the existing cache. Catalogue interfaces remain unused by this workflow.

## Answer a question

```python
answer = client.answer_question(
    user_id="user-123", question="When did I adopt Scout?",
)
```

For a question naming the memory owner, supply the name known to the caller:

```python
answer = client.answer_question(
    user_id="user-123", question="What are the breeds of Audrey's dogs?",
    memory_owner_name="Audrey",
)
```

`memory_owner_name` is optional on both `answer_question` and `retrieve_memories`.
It tells the planner, fact filter and answering model who “the user” refers to.
It is call-local context, not a stored fact, a name search or an authorization key;
database and pending reads remain scoped by `tenant_id` and `user_id`. Supply it
only when that user-to-name mapping is known. Tomte does not infer that a named
friend or relative in the question is the memory owner. Without the name or
identity evidence, attribution can remain uncertain. Names must be nonempty
text, at most 256 characters, with no NUL character. Existing calls still work.
Standalone `retrieve_memories` includes the supplied name in its returned text
so a downstream reader retains this context. It does not relabel stored subjects.

This returns one answer string. Unlike `get_context()`, it uses the configured
LLM providers through the existing `parse_structured` interface:

1. A separate information planner analyses the question before retrieval. Its
   `extract_required_info_from_question(question, retrieval_results, suggestions)`
   method returns a list of short, self-contained information requirements.
   Prerequisites come first: request Lucky's adoption date before requesting pets
   adopted before that date. An empty list lets the answer model explain or clarify.
2. The retrieval workflow prepares search text with an LLM, removing exclusion
   phrases and excluded items (for example, "besides hiking"). It embeds this
   text while retaining the complete request for filtering. Negative facts being
   sought, such as "medicines I cannot take", keep their meaning. No catalogue
   or entity selection precedes the search.
3. In the native PostgreSQL/vector path, Tomte retrieves at most 100 snapshots
   from one broad fact search, scoped to the tenant and user. Existing plain-value vectors
   rank snapshots; no new embedding index is needed. Multiple relevant facts from
   one entity can precede weaker facts from other entities; this fact lookup does
   not apply the distinct-entity selection of the background entity retrieval API.
4. The reader sees at most 20 candidates per call, in vector rank order, plus
   previously selected evidence from this request. For every candidate it decides
   one category: `found`, `possible`, `related_only`, or `unrelated`. Only found
   and possible facts are selected. Python stops after a batch containing neither
   when the last 10 judgments are consecutively unrelated, when the list or
   100-candidate cap is reached, or when a consistently bounded request is
   satisfied with cited found evidence. Collections and
   ambiguous requests cannot stop on satisfaction. An empty selected batch alone
   never stops reading. Repeated information is not penalized for repetition.
   The answer model receives entity identity plus complete selected fact values,
   with current/history markers, rather than every fact of every entity.
   Unresolved memories and relevant pending conversation retain their separate
   qualified presentation. Pending embeddings reuse the existing cache.
5. The answering LLM receives the question and accumulated memory, without the catalogue.
   It answers immediately if the evidence is sufficient. Otherwise it identifies
   specific missing information, including known names, dates or event context.
   A separate planner turns these suggestions into further information requests.
   It sees every attempted request and its outcome, including empty results, plus
   accumulated evidence once. Returned evidence is not automatically considered a
   successful answer. The model should not repeat satisfied needs or reword failed
   searches; new evidence or a materially different route can justify another attempt.
   Retrieval executes each request without seeing the original question or earlier
   evidence. The answering model does not generate catalogue handles or database queries.
6. The process stops after at most three retrieval passes. A final answering
   decision always reviews the third result before returning the best supported
   answer with any uncertainty explained by the model.

One-pass answers take one information-planning call, one query-preparation call and one answering call,
plus zero to five fact-reader calls, with no catalogue calls.
The three-request ceiling counts each item in a planner list, not just planner
iterations. Follow-ups add a planner call before further retrieval. An empty
follow-up plan returns the existing best answer without another answering call.
Actual counts depend on the available facts and the model's relevance decisions.
The workflow adds no retry loop; provider/SDK transport or
structured-output retries may cause more underlying requests. This is not a dollar
cost guarantee. All calls use the existing foreground usage scope.

Within an answer operation, retrieval reuses one pending-message snapshot,
pending embeddings. The answering workflow owns the accumulated evidence and
three-pass ceiling; the separate reader owns the batch loop for each fixed ranking.
Standalone retrieval renders only its current request's results. Empty,
structurally invalid and exactly repeated requests are skipped without discarding
other valid list items. If none remain, the existing best answer is returned.

`answer_retrieval.fact_limit` defaults to 100; larger legacy overrides are clamped
to 100. Batches contain at most 20 snapshots. Each fact receives one category:
found, possible, related_only or unrelated, with no reason or numeric score.
`relevance_tail_size` defaults to 10 consecutive unrelated facts at the end of a
batch containing no found or possible facts. Related-only and invalid judgments
reset the counter. A useful batch earns another batch unless found evidence
satisfies a bounded request. The old numeric threshold has been removed.
Found and possible facts reach the answer separately, with possible evidence
clearly marked as tentative. The existing `lane_bytes` setting bounds each database fact packet
(24,000 UTF-8 bytes), not the sum of the 100 candidates. `memory_bytes` bounds
previously selected reader context and accumulated returned fact context (64,000
bytes). Whole snapshots are retained or omitted, never cut mid-sentence. If prior
reader context is partial, satisfaction cannot stop that batch. Catalogue limits
and `read_all_bytes` remain compatibility settings, unused by this path.
Coverage reports candidate limits, early stopping, oversized values and context
omissions. Stopping never proves that unseen facts are irrelevant. An invalid
entry does not discard valid judgments for other facts. Unclassified facts are
not returned or counted as unrelated. A wholly unavailable reader selects nothing
from its batch and continues; after a call failure, satisfaction is disabled for
the rest of that request. Classification gaps remain visible in coverage.
Scope violations, cancellation and capacity failures still propagate.

This path is read-only and requires no schema migration or re-embedding. Custom
database providers opt in to fact search through `supports_answer_fact_search`
and `get_answer_facts`, called with empty paths. Catalogue methods are not required.
The older catalogue methods remain available for compatibility. Providers without fact search,
and the opt-in BM25/hybrid modes, retain the prior entity-based answer retrieval.
The separate non-generative retrieval APIs and mutation candidate search do not change.

Python enforces scope, structural validity, exact-request deduplication and bounds;
it does not judge the meaning or truth of an answer. The model is instructed to
stop early, preserve uncertainty, distinguish plans from completed events, and
treat assistant messages as context rather than independent user facts. The
question itself is not evidence for assumptions it contains. It may reason or
calculate from available evidence, but must not invent missing personal facts.

Returning the same evidence does not itself prevent
another targeted search within the remaining round allowance. These controls
bound searching; they do not prove semantic answer quality.

One pending snapshot is captured when the first retrieval starts and reused for every round.
Its window embeddings are reused within the call and through the existing shared
cache. New messages arriving afterward are available on a later call. Durable
reads can observe newer commits between information requests; this is not a
distributed database snapshot. Each request freezes its returned ranking for the
reader loop, so rows do not shift between its batches.
The model sees labels warning that raw messages and entities can describe the
same event. Entities are accumulated by UUID. Different selected history indices
are retained when the stored history remains an unchanged prefix; identical text
does not collapse distinct history entries.

Question length uses the existing `max_input_length`. Each model input includes
only public memory fields and is subject to the existing `max_model_input_bytes`
capacity guard. Oversized input raises rather than silently deleting evidence.
There is a 120-second cooperative deadline, with the existing provider timeouts;
an in-flight external call cannot always be interrupted immediately. Failures
propagate instead of becoming a fabricated answer. No separate scoring/grading
call runs, and neither the question nor the answer is automatically stored.
No database schema change or provider-specific answer endpoint is required.

## Store user and assistant messages

Both storage methods return only `{"message_id": "<UUID>", "status": "queued"}`.
They make no embedding, retrieval or generative LLM call. Redis atomically stores
the message in recent chat history, appends it to the consolidation queue and
updates scheduling activity. Preceding turns are captured as reference context.
Existing message length/Unicode validation and optional PII scrubbing apply;
Python does not decide the meaning of the message.

The receipt acknowledges queue storage, **not completed extraction or permanent
memory**. The existing background worker processes the queue later and generates
any missing user-message embeddings there. Existing queued vectors are reused.
Assistant messages supply context, not independent facts about the user; an
assistant-only batch can finish without creating any entity. The existing
bounded batches, retry rules and PostgreSQL commit markers still apply.
With `start_daemon_worker=False`, the host must run consolidation explicitly.

These methods do not update session entity activation or PostgreSQL user-activity
heartbeats; scheduling uses the Redis activity index. History retains its existing
size/TTL and `isolate_chat_sessions` configuration; consolidation queues always
remain tenant/user/session scoped. Queue durability depends on Redis persistence.
Each successful call creates a new message UUID. A timeout after Redis accepted a
write can leave the caller uncertain; retrying storage is not an idempotent API.

## Profile and most-mentioned entities

`get_profile_entities(user_id)` returns all entities whose domain is
`identity_and_demographics`. It reads PostgreSQL directly, without a stale persona
cache or the legacy priority-fetch limit. Order is importance descending, then
mention count descending, then entity UUID.

`get_most_mentioned_entities(user_id, limit=5)` uses the existing entity counter:
the sum of its facts' `_mention_count` values. This is not a count of distinct
messages. Results sort by that count descending, with entity UUID as a stable
tie-breaker. Profile entities may appear; unresolved memories are excluded.
`limit` must be a positive integer.

Both methods return lists of public entities with all current facts and stored
history. Internal metadata is omitted. Neither uses Redis, embeddings or a
generative LLM, and neither changes mention counters. Full histories may make
these responses larger than a query-selected history view.

## Pending messages

`get_pending_messages(user_id, session_id=None, limit=20)` returns the newest
unprocessed messages across the user's sessions, or only the specified session.
The selected messages are ordered oldest to newest for use as chat context:

```json
{
  "messages": [{
    "message_id": "d79685a5-41c4-4ee1-b57a-9c8df28c7bd0",
    "session_id": "session-1",
    "role": "user",
    "content": "I adopted Scout.",
    "received_at": "2026-11-08T10:00:00+00:00",
    "status": "queued"
  }],
  "has_more": false
}
```

`queued` means waiting; `processing` means part of a frozen worker batch, including
a batch waiting for a retry. Both user and assistant messages are included, with
their roles preserved. `has_more` indicates older pending messages beyond the
requested limit. Embeddings and internal reference-context copies are not exposed.

Redis queue and processing lists are read atomically. A PostgreSQL commit-marker
check excludes batches already completed but still awaiting Redis acknowledgement,
including batches whose extraction produced no memories. The read never freezes,
acknowledges or removes a batch and makes no model calls. Redis and PostgreSQL do
not share a transaction: processing may complete immediately after this snapshot.
This is not an archive or proof that every message became a fact. Completed and
terminally skipped batches are no longer pending. Reading an invalid stored item
raises instead of claiming the queue is empty.

Without a session filter, Redis scans the existing active-session index to find
this user's sessions. A `session_id` avoids that scan when the caller only needs
one conversation.

## Search stored memory and pending conversation together

For callers needing structured results, the combined API accepts a statement or
question and performs retrieval only:

```python
memory = client.get_relevant_memories(
    "user-123", "My dog Scout", limit=5, include_pending=True,
)
entities = memory["entities"]
unresolved = memory["unresolved_memories"]
windows = memory["pending_message_windows"]
coverage = memory["pending_search"]
context = memory["context_text"]
```

`get_relevant_entities()` remains an entity-only list. `include_pending` defaults
to `False`, preserving the existing combined response and avoiding Redis reads.
With `True`, the combined response adds `pending_message_windows` and
`pending_search`; `context_text` includes labelled pending conversation.
Unresolved memories are durable extracted statements with uncertain attribution;
pending windows are queued raw conversation. These are different categories.

There is one query embedding shared by entity, unresolved-memory and pending
search. No generative LLM runs. Missing window vectors are embedded in batches;
identical cached windows reuse their vectors. New messages rebuild only affected
boundaries/tails, while intact completed windows retain their boundaries.
Each window contains up to six messages, with two-message overlap where size
allows. Oversized messages use lossless character spans, labelled `is_partial`
with `char_start`, `char_end`, and `total_chars` in the response. Returned windows
deduplicate identical spans and retain chronological order within each session.
They never mix sessions or users.

Each returned window has `session_id`, `messages`, `reference_context`, and
`reference_context_is_partial`. Messages retain UUID, role, original content,
UTC `received_at`, and `queued`/`processing` status. Preceding reference context
is separate. Assistant turns supply context, not independent user facts.
The caller must treat this content as untrusted conversational evidence.

The request captures pending conversation before reading durable memories, and
excludes processing batches whose PostgreSQL commit marker already exists.
This avoids losing a batch during normal queue-to-memory handoff, but is not a
distributed snapshot: an event can appear in both pending and durable memory.
Do not count those representations as different events. Messages arriving after
capture belong to a later call. The capture object also reuses its vectors if a
answering workflow performs several searches over the same snapshot.

`pending_search.complete` describes coverage of pending input, **not** proof that
the returned memories answer the question. It includes `reasons`, `snapshot_at`,
`observed_queue_messages`, `messages_considered`, `windows_searched`, and
`windows_returned`. Limits, invalid serialized input, an unavailable snapshot or
an unfinished competing embedding can make coverage incomplete. Durable reads
still run when the pending snapshot provider is unavailable/unsupported. Model
failures and namespace violations raise; they are not reported as empty queues.

### Cache lifetime and limits

The Redis cache stores window membership metadata and vectors, not another copy
of raw chat. It is scoped by tenant/user/session and keyed by exact embedding
input, window format and embedding deployment identity. Both manual and worker
acknowledgement remove affected windows atomically with queue completion.
Terminal skips clean them too; retries retain them. Cleanup follows the complete
consolidation commit and queue acknowledgement, not just the extraction call.
Overlapping windows touching completed messages are retired; unaffected later
windows survive. Generation and owner checks reject late cache writes.

A 24-hour TTL is a fallback for abandoned/unused entries. Concurrent requests
normally share a leased embedding result and wait up to two seconds for it.
Expired leases can be taken over. TTL expiry, eviction, failed publication,
provider retries or uncertain network outcomes can cause re-embedding; exactly
once external billing is not guaranteed. Overlapping windows are deliberately
different inputs. This cache does not replace background extraction embeddings.

Defaults, configurable with `PENDING_SEARCH__<SETTING>` environment variables:

- `MAX_MESSAGES=1000`, `MAX_INPUT_BYTES=262144`: newest pending input considered.
- `WINDOW_BYTES=6000`: UTF-8 embedding input per window.
- `MAX_EMBEDDING_BYTES=262144`: new window input per request snapshot.
- `RETURN_WINDOWS=3`, `RETURN_BYTES=8000`: selected pending output; entity limits
  are separate. Whole windows that exceed the remaining output budget are skipped.
- `CACHE_WINDOWS=256`, `CACHE_BYTES=8388608`: entries and serialized vector bytes
  per user; metadata is additionally bounded by entry count and window size.
- `CACHE_TTL_SECONDS=86400`, `LEASE_SECONDS=45`, `WAIT_SECONDS=2`.

These are byte budgets, not exact token counts. Snapshot discovery scans the
existing active-session index; this is not an indexed per-user session lookup.

## Compatibility and custom providers

`send_user_message` and `log_assistant_response` retain their existing combined
chat-hook behavior for current callers. New applications can use the methods
above. `get_relevant_memories` remains available for separately labelled unresolved
statements and formatted memory text (below).

Custom cache providers must implement `store_message` and
`get_pending_message_batches` to support the new storage/pending APIs. Custom
database providers must implement `get_profile_entities`, and pending reads use
their existing `is_consolidation_processed` capability. The new interface methods
raise `NotImplementedError` by default; legacy implementations are not made
abstract by this addition. No database migration is required.

Pending search additionally needs `get_pending_search_snapshot` and optionally
`claim_pending_window`, `publish_pending_window`, `release_pending_window` on the
cache provider. A custom implementation must invalidate generations and affected
windows on acknowledgement. Shared vector reuse requires the model provider's
`get_embedding_cache_identity()` to identify its actual deployment, model revision,
dimensions and task mode without credentials. OpenAI adapters implement this;
other adapters default to request-local reuse until they opt in. No cache identity
means no shared vector caching, even if dimensions happen to match.

## Read relevant entities

The synchronous SDK exposes a read-only operation on the existing client:

```python
from tomte.engine import Tomte

# After the application's existing Tomte.init(...) setup:
client = Tomte.create_tomte_client(tenant_id="my-app")
entities = client.get_relevant_entities(
    user_id="user-123",
    message="I changed my job last month",
    limit=5,
)
```

`message` can be a chat statement, question or other nonempty text. It is used
as a search input, not interpreted as an instruction to change memory. The API
does not require a session ID. The application must authorize the requested
user within the client's tenant before calling the SDK.
Custom database providers must include matching `tenant_id` and `user_id` on
retrieved rows; these scope fields are checked and omitted from the public result.

## Retrieval and result

With the default `STORED_RETRIEVAL_MODE=vector`, the operation embeds the message
once and runs two tenant/user-scoped rankings:
the top `limit` entity summaries, and the top `limit` **distinct entities** by
their best plain-text current/history field match. It unions those lists by UUID
and sorts the union by each entity's best cosine, with stable UUID tie-breaking.
Several matching fields on one entity consume only one field-ranking slot.
Results are independent of
mention frequency and recording recency. There are no Python language rules or
question classification. This is nearest-neighbor retrieval without a minimum
similarity threshold; a generic message can still return weakly related items.

The return value is a list of public entity dictionaries, using the same UUIDs,
facts, current text values and `fact_confidence` fields as the chat API. Internal
metadata and retrieval scores are omitted. All current facts remain present.
Entities in `thoughts_and_plans` also return `first_mentioned_at`: the earliest
supporting user-message time available to Tomte, in UTC. This is not the idea's
origin date or current confirmation. Read the qualified fact text; desired details
and plans are not established objects or completed actions. Ordinary entities do
not gain this field, and thought timestamps add no generative call to this API.
Small histories are returned in full; larger histories use the existing index
ranking, recording-order fallback and history byte/candidate limits. This method
disables the optional LLM history selector. Any omitted history is indicated by
`history_is_partial` on the affected fact; stored history is unchanged.

`limit` must be a positive integer. If omitted, it uses
`settings.semantic_retrieval_limit` (default: 5) **per ranking**. Five can return
up to ten unique entities; overlapping selections reduce that count. There is no
second truncation back to five after the union. This applies to `get_context`,
`get_relevant_memories` and `get_relevant_entities`. Native vector question answering
uses the separate fact-level workflow described above; its compatibility fallback
still uses this entity ranking.
Optional `bm25` and `hybrid` modes use a separate lexical index of saved field
text. Hybrid retains the default vector result count, while changing selection
and order; BM25 alone returns at most `limit` distinct entities. These modes
require an explicit lexical backfill for existing users. See
[BM25 retrieval](bm25-retrieval.md) for setup, scoring, history handling and limits.
The existing
`settings.max_input_length` applies to the message. An empty list means the read
succeeded with no returned entities. Invalid input, provider failures and scope
violations raise rather than masquerading as an empty result.

## Side effects and cost

Fact-field retrieval embeddings contain each field's `text` and optional short
`context`, separated by a newline, including historical values. UUIDs, generated names/types/domains, attributes, confidence
and timestamps are not prefixed. Long text is chunked without dropping content.
Entity-summary embeddings retain their JSON format without the former byte cap.
The OpenAI adapter shortens document inputs only beyond the model's token limit;
see [embedding limits](memory-design.md) for provider and operational boundaries.

Plain field vectors live in the derived `tomte_memory_retrieval` table. The
labelled `tomte_memory_search` index remains available to mutation candidate
resolution. Legacy session/weighted and unresolved-memory retrieval now use the
plain field index, keeping their existing ranking rules and limits.
New/changed text is embedded for both indexes, so write
embedding cost and vector storage increase. Both indexes and the payload commit
atomically. Querying the two rankings reuses one query vector.

For populated installations, stop writers, run normal infrastructure bootstrap,
and backfill each user before serving caller retrieval:

```python
admin = Tomte.create_tomte_admin(tenant_id="tenant-123")
rebuilt_entities = admin.rebuild_retrieval_index(user_id="user-123", refresh_summary=True)
```

This makes embedding calls and replaces only that user's derived vectors.
The default `refresh_summary=False` replaces just the plain field index. Choose
`True` to also apply the larger summary input to existing entities. The summary
vector is shared by caller retrieval and mutation candidate search, so refreshing
it can change both rankings. Labelled field vectors, payloads, UUIDs, history and
counters remain unchanged.
Each entity's summary/field replacement is atomic and rejects a changed payload; a failed user
rebuild can be rerun. Bootstrap alone does not backfill old data. Before a
backfill, old entities have summary-only caller retrieval, so do not claim the
new two-index behavior is fully deployed in a populated installation. This is a
text-format rollout, not an embedding-model migration.

This call makes one query-embedding request, which can incur embedding-provider
cost. It makes no generative LLM, junk-filter, extraction, summarization or
answering call. Normal in-process foreground usage reporting still applies.

The method performs database reads and does not access Redis. It does not save
the input, queue consolidation, update activity/mention counts, promote session
entities or append chat history. It reads committed memory without waiting for
any pending background consolidation. It adds no profile/recent/session-only
background entities to the results.

This is a Python SDK operation, not a new HTTP server or route.

## Pending statements with unresolved attribution

`get_relevant_entities()` keeps its list return type and excludes the
`unresolved_memories` domain. To include relevant pending statements, use:

```python
memory = client.get_relevant_memories(user_id="user-123", message="My painted car", limit=5)
entities = memory["entities"]
pending = memory["unresolved_memories"]
context_for_llm = memory["context_text"]
```

This uses one query embedding and no generative LLM. Normal entities retain their
own limit; pending records use `unresolved_return_limit` (default 3), ranked by
similarity independently. Both queries enforce tenant/user scope. As with normal
semantic retrieval, there is no relevance threshold, so candidates can be weakly
related. A pending record is an earlier user report with incomplete attribution,
not a fact about any particular entity in the other list. Each includes
`unresolved_detail` (what identity is missing) and `reported_at` (UTC report time,
not necessarily event time), together with its qualified fact text. Internal
lifecycle metadata stays private. Resolved records are excluded before ranking.

`send_user_message()` also returns an additive `unresolved_memories` list and
places that labelled context after ordinary relevant/background memory. Pending
records do not enter persona, recent, frequent, or session entity lists. Reading
does not resolve or update a pending claim; later background consolidation handles
new user clarifications. Assistant logging remains cache-only.

Custom providers must implement `get_unresolved_memories` as scoped storage rows
and exclude this domain from ordinary reads. Legacy providers without that
capability return no pending context; they do not provide the full new workflow.

## Memory text format

`get_context()` returns the memory text directly. `send_user_message()` and
`get_relevant_memories()` provide it as `context_text` in their structured responses.
It defaults to **labelled rows**: column names appear once in
a guide, followed by JSON-escaped entity, fact and field rows. `current` and
`history` labels keep values separate. UUIDs, confidence, selected history,
partial-history flags and public thought/unresolved dates are preserved.
The guide appears once per packet. Empty stored-memory sections produce no text;
including pending search can still add coverage information when no memories match.

Switch the application default with `MEMORY_TEXT_FORMAT=compact_json` in `.env`
(reload the application), or `settings.memory_text_format = "compact_json"`.
The two supported choices are `labelled_rows` and `compact_json`.
The read API also accepts a per-call override, without changing global settings:

```python
memory = client.get_relevant_memories(
    user_id="user-123", message="My painted car", output_format="compact_json",
)
context_for_llm = memory["context_text"]
```

Compact JSON uses the same section headings, with a compact JSON array inside
each section; the whole `context_text` string is not a single JSON document.
Both formats place ordinary memory before separately labelled unresolved reports.
This switch changes serialization only, with no additional LLM request or summary.
Storage and internal extraction/mutation inputs remain unchanged.

Structured lists remain available in the payload for application code, and
`get_relevant_entities()` continues returning its existing list of dictionaries.
Send **only `context_text`** as the memory portion of the LLM input; serializing
the entire response would repeat the same memories and lose the token savings.
Use the structured lists for parsing, rather than relying on the exact text layout.
