# Answering and batched fact retrieval

The public methods remain `answer_question(user_id, question) -> str` and
`retrieve_memories(user_id, information_needed) -> str`. Retrieval uses direct
fact-vector search followed by batched LLM reading. It does **not** navigate a catalogue.

Both methods accept optional keyword-only `memory_owner_name`. This is the
caller's known display name for the scoped user, passed as data to planning,
fact reading and answering. It connects named requests to “the user” in stored
values without guessing that every name in a request is the owner. Other people
and their belongings remain distinct. No name is extracted or matched by Python,
and database/cache scope still uses tenant/user IDs. The name is not persisted or
added to stored embeddings. Standalone retrieval includes it in returned memory
text; existing calls without a name retain their original output shape. Injected
planner/retrieval implementations must receive any required owner context from
their caller; the answer workflow does not mutate injected components.

## Responsibilities

- `InformationPlanner.extract_required_info_from_question(question, retrieval_results,
  suggestions)` returns short, self-contained information requirements. Follow-ups
  receive attempted requests and outcomes, available evidence once, and the answer
  model's suggestions. Empty results do not prove absence.
- `AnswerWorkflow.decide_next_step()` receives the question, accumulated evidence
  and remaining search allowance. It answers or suggests missing information.
- `prepare_embedding_query()` asks the retrieval-planner model to remove search
  exclusions and excluded items from the embedding text, retaining the remaining
  wording. A negative fact being sought is not an exclusion. Python does not parse
  human-language negation. The original request stays intact for the fact reader.
- `MemoryRetrievalWorkflow.retrieve_once()` embeds the prepared search text,
  gets one tenant/user-scoped ranking of fact snapshots, and searches pending and
  unresolved sources. It does not generate requests or answers.
- `BatchedFactReader.read()` filters the fixed ranking in batches of 20, with a
  hard maximum of 100 candidates per information request. The LLM judges meaning;
  Python validates offered handles and counts consecutive unrelated judgments.
- `EvidenceBuffer` merges results by UUID and history position. The answer owns
  its accumulated buffer. Standalone retrieval renders only its current request.

The answer workflow still accepts injected planner/retrieval implementations.
Shared provider capacity and cancellation checks remain in `llm/calls.py`.

## Candidate ranking and reading

The prepared search text is embedded once; the full request goes to the reader. The existing PostgreSQL
`get_answer_facts` method receives empty paths, so there is no catalogue filter.
Existing plain-value field vectors score each current or historical fact snapshot
by its strongest matching field. Multiple facts from one entity may appear in the
ranking. This differs from the distinct-entity selection of background retrieval.
Exclusion removal requires no embedding model change or reindex. Values with new
optional context include that context in their derived plain-text search entries.

Each reader call receives the information request, up to 20 complete fact snapshots
with entity context, and previously selected evidence from this request. Local
handles identify snapshots without sending database UUIDs. The model returns:

- one category per fact: `found` (supplies a requested detail), `possible`
  (a concrete near-answer with explicit uncertainty), `related_only` (topic/context
  without an answer), or `unrelated`;
- request kind (`bounded`, `collection`, `ambiguous`), satisfaction and supporting
  found handles. Singular wording alone does not identify a unique subject.

Production output contains no reasons, explanations or numeric scores. A plan can
be found when plans are requested, but only possible evidence that an event occurred.
General topic similarity does not qualify as possible evidence.

The filter input uses quoted labelled rows rather than nested JSON. Entity
name/type/domain and nonempty aliases are listed once per call; facts reference
those local entity labels. Equal names with different entity IDs remain separate.
Fact numbers preserve ranking across batches. Complete value text, optional
context, first-mention dates and history remain available. Current values and
high confidence are stated as defaults in the prompt; exceptions are explicit.
Quoting escapes literal newlines and control characters without rewriting text.
Neither storage UUIDs nor internal tracking metadata are sent.

The compact structured response contains `kind`, four numeric ID lists (`found`,
`possible`, `related`, `unrelated`), and `enough`. `related` maps to the internal
`related_only` category. An empty `enough` means unsatisfied; otherwise it cites
found evidence for a bounded need. Every current candidate must occur exactly
once. Missing IDs are unknown, never implicitly unrelated; duplicate IDs cannot
select facts. The numeric IDs are local to this retrieval, not database IDs.
This changes only the private model protocol, not stored memory or public APIs.

After each batch, reading stops when:

1. A consistently bounded request is satisfied by cited found evidence.
2. The candidate list or hard 100-candidate cap is reached.
3. The current batch has no found or possible facts, and ends with at least ten
   consecutive unrelated judgments. Related-only and unknown entries reset this
   counter. A useful batch earns another batch unless the request is satisfied.

Finding some members never satisfies an open collection. An empty selected batch
does not stop search when its facts are still topically relevant. Repetition does
not lower relevance. These are heuristics, not a proof that no useful fact exists
later in the ranking. Unexamined candidates and database limits remain visible as
partial coverage. The final short batch is still read, and an empty search makes
no reader call.

## Limits and failure behavior

OpenAI routes `FactBatchJudgment` to `answer_reader_model` with no format retry.
There are at most five reader calls per information request, plus one short query
preparation call and its embedding. Identical requests reuse prepared text within
the same workflow instance. Invalid preparation output fails before embedding;
it does not silently fall back to the excluded wording.
Answering still permits at most three information requests. Pending embeddings
reuse the existing cache. Provider transport behavior and actual token sizes mean
these counts are not a dollar-cost guarantee.

The no-generative-LLM `get_relevant_entities` / `get_relevant_memories` APIs keep
embedding the caller's text unchanged. Automatic exclusion removal belongs to
`retrieve_memories` and `answer_question`, which already use LLM interpretation.
Optional stored value `context` travels to filtering and answering with its text.

`answer_retrieval.fact_limit` defaults to 100; legacy larger values are clamped to
100. `relevance_tail_size` defaults to 10 consecutive unrelated facts. The old
`relevance_stop_threshold` numeric setting has been removed.
`lane_bytes` retains the 24,000-byte per-packet database limit. It no longer caps
the entire candidate list. `memory_bytes` bounds previous selected reader context
and accumulated returned fact context, default 64,000 UTF-8 bytes. Whole values
are never cut mid-sentence. Partial previous context disables satisfaction for
that batch. The existing overall model-input capacity guard still applies.

Malformed individual judgments are omitted without discarding valid siblings.
Missing, duplicate and unknown handles cannot select a fact or count as unrelated.
Invalid entries are reported in diagnostic coverage. A wholly unavailable/malformed
reader call selects nothing from its batch and cannot justify stopping. Later
batches are still read within the same limit; there is no automatic batch-retention
fallback or additional repair call. Reordered valid judgments are restored to
vector rank order. Satisfaction requires valid found citations, a bounded request,
and complete previous context. Scope violations, cancellation, budgets and
model-input capacity failures still propagate.

Retrieval returns separate found and possible packet lists internally. Both reach
the answering function; possible values are explicitly marked as tentative. The
same evidence snapshot is not repeated if it later becomes found. Original text,
identity and history stay unchanged. The shared context budget fits found evidence
first, then possible evidence; omissions remain visible. Public retrieval still
returns text, with a separate possible-candidates section.

The returned ranking is fixed within one request. Later information requests may
observe newer database commits. One pending snapshot is reused within an answer;
this is not a distributed database snapshot.

## Answering and other APIs

For "What pet did I get before Lucky?", the planner can request Lucky's adoption
date first, then use the returned date to request earlier pets. Alternatively it
can request all relevant adoption dates. The model chooses; Python does not encode
these language decisions. The final answer reviews every executed request,
including the third, and preserves uncertainty and missing evidence.

The answer renderer retains names, aliases, field labels, confidence, current/history
distinctions, time anchors and unresolved attribution, without storage UUIDs. Pending
messages keep speaker and received time. Brief coverage warnings replace detailed
search counters for answering. No LLM summarization is performed by this renderer.

Non-generative APIs, mutation candidates, and pending/unresolved retrieval are
unchanged. Custom databases without fact search, and opt-in BM25/hybrid modes,
retain their existing entity-based retrieval. The native vector path needs only
`supports_answer_fact_search` and `get_answer_facts`; no catalogue methods.

The older `CatalogueNavigator`, catalogue provider methods, catalogue settings and
legacy selection schema remain for compatibility and isolated experiments. They
are not called by the current retrieval/answering workflow. Adjust the active
reader in `workflows/fact_reading.py` and `llm/prompts/fact_reading.py`; adjust
question analysis in `workflows/information_planning.py` and answering in
`workflows/answering.py`. See [memory read API](memory-read-api.md) for the public contract.
