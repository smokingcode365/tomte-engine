# Legacy catalogue navigation (inactive)

This documents the previous retrieval design. The current retrieval and answering
workflows do **not** call the catalogue navigator. They use direct fact-vector
search and batched reading; see [current workflow](fact-retrieval.md). The older
navigator and provider interfaces remain available for compatibility and isolated
experiments. The behavior described below is historical, not the active workflow.

The public methods remain `answer_question(user_id, question) -> str` and
`retrieve_memories(user_id, information_needed) -> str`. Their internal boundaries
separate deciding what to find from locating and reading that information.

## Responsibilities

- `InformationPlanner.extract_required_info_from_question(question, retrieval_results,
  suggestions)` returns a list of short, self-contained information requirements.
  Its prompt and response are separate from answering. The first call contains
  only the question and reference time. Follow-ups receive each attempted request
  and its outcome, combined available evidence once, and the answering model's
  suggestions. Returned evidence is not automatically treated as answering the
  request; an empty result is not proof that the information does not exist.
- `AnswerWorkflow.decide_next_step()` receives the question, accumulated evidence
  and remaining allowance. It returns an answer and, when more information would
  help, suggestions for the planner. It sees no catalogue.
- `MemoryRetrievalWorkflow.retrieve_once()` executes that request and returns a
  `RetrievalResult`. It embeds the request directly, searches stored facts and
  pending/unresolved sources, and optionally filters large fact sets. It never
  generates a follow-up request or an answer.
- `CatalogueNavigator.select_paths()` receives only the information request. It
  returns concrete entity/fact paths and coverage. It uses its own prompt, schema,
  renderer and database read methods. It cannot execute evidence searches or
  change the caller's request.
- `EvidenceBuffer` merges results by identity and preserves history and source
  qualifications. The answering workflow owns this buffer. Standalone retrieval
  renders a fresh buffer, so results from an earlier request do not leak into it.

Answering uses a separate compact presentation in `core/answer_text.py`. It retains
selected value text, field labels, confidence, names/types/aliases, time anchors,
current/earlier grouping and unresolved attribution. Short local labels distinguish
same-name subjects and same-attribute records; they do not imply event chronology.
UUIDs remain internal for merging. Pending turns retain speaker, received time,
reference status and partial-message warnings, but omit queue IDs. Empty pending
sections and detailed search counters are replaced by brief notices when incomplete
coverage or unavailable search/filtering matters. No LLM summarization or language
matching is performed by this renderer. Existing selection/size limits and the
public retrieval format are unchanged.

The workflow constructors accept an injected `planner`, `retrieval` or `catalogue` instance
for isolated testing or replacement. They do not use a global service registry.
Shared model capacity/cancellation checks live in `llm/calls.py`.

## Catalogue traversal

The navigator starts with owned entities across all domains, within the row/byte
budgets. Domains are headings in this single entity-selection input, not a
separate filter or model call. These illustrative rows show the first input:

```text
relationships
  E1 Lucky | dog | aliases: Lucky dog
  E2 Coco | cat
assets_and_possessions
  E3 My journal | journal
```

After selecting Lucky:

```text
Lucky | dog
  F1 acquisition: when, location
  F2 breed: breed
```

The output schema contains only `handles`, such as `{"handles": ["F1"]}`.
Handles are local to that selection call, not stored IDs. Python resolves them
to offered entity/fact UUIDs. Different facts with the same attribute keep
different handles. Fact rows include current and historical field names but no
values. Backslashes and row separators in labels are escaped without interpreting
their language. UUIDs, repeated JSON property names, counts and internal metadata
are not included in the catalogue text.

The LLM may choose entities under any domain heading. A sole offered entity is passed
through automatically; up to six offered facts are all retained by default.
This skips a selection call, not evidence-size limits. Larger fact sets are
selected with the same focused request. At most two catalogue calls are made
per retrieval, with configurable row/byte limits and explicit partial coverage.

Selection focuses on likely evidence locations for the requested subject and
attribute, action or relationship. A related topic alone is insufficient. Broad
labels and indirect, uncertain, planned or historical evidence remain valid when
relevant. When a selection call is made, choosing a parent does not require
choosing a child or all its facts; an empty selection is allowed. This policy is
implemented by the catalogue LLM prompt, not Python language-matching rules.
Provider trust instructions treat catalogue labels and request text as data.

## Independent evidence searches

Selected paths read the exact facts and their history snapshots, within budgets.
A separate broad vector search always runs without the catalogue filter.
Unresolved memories are always searched separately. Pending conversation uses
its existing window search and cache.
Small sets of actual values bypass relevance filtering; large sets can use the
optional fact reader. The reader receives the request and values, not the original
question or catalogue.

Catalogue database reads and the pending snapshot are reused within an answer
operation. A later answer has a new navigator/snapshot. Durable reads are not one
transaction-wide snapshot, so current data can change between reads. Cached
catalogue branches may be stale; the broad lane can find new facts independently.

## Example of dependent searches

For "What pet did I get before Lucky?":

1. Information planning requests "When I adopted Lucky".
2. Retrieval selects likely acquisition facts and searches for that information.
3. Given a supported date, answering suggests finding earlier pets. The planner
   converts that suggestion and the actual retrieved information into the next request.
4. Retrieval executes that request. Answering compares the returned evidence
   and either answers, explains a limitation, or uses its remaining retrieval pass.

The model chooses the requests; this sequence is not hardcoded Python. An initial
request for all relevant adoption dates is also valid when the model chooses it.
At most three information requests are retrieved, including separate items in one
planner list. Lists run in priority order within that allowance. Answering sees
all results from the executed list, including the third request, before stopping.
An empty follow-up plan returns the best answer already obtained; an empty initial
plan lets answering explain the gap or ask for clarification. Python rejects only
structurally invalid and exactly repeated request strings. The planner owns
semantic repetition, relevance and whether new evidence justifies a different route.

## Adjustment points and limits

Change catalogue formatting/selection in `workflows/catalogue.py` and
`llm/prompts/catalogue.py`. Change question analysis/stopping in
`workflows/information_planning.py`, `workflows/answering.py` and
`llm/prompts/answering.py`. Change evidence retrieval in
`workflows/retrieving.py` and the database methods. The public API stays the same.

OpenAI catalogue calls keep the existing `retrieval_planner_model` configuration
name. No embedding-model change, re-indexing or schema migration is required.
Custom providers must support `get_catalogue_entities(..., domains=None, limit=...)`
to read owned entities across all domains; an explicit empty domain list still
means no matches. The domain-list read remains available for compatibility but
is no longer used by navigation. Tenant/user constraints, unresolved exclusion
and partial-coverage bounds apply to the broader entity read.
Custom databases without catalogue navigation retain broad fact retrieval;
those without fact retrieval and BM25/hybrid mode retain entity retrieval.

Entity/fact selection can miss a branch. Broad search reduces that risk but
does not prove complete recall. Removing domain selection saves one possible
call but enlarges the entity input. A large catalogue can reach the existing
row/byte bounds; partial coverage remains visible. Net token cost and latency
depend on the selected paths and model outputs, not just the number of calls.
See [memory read API](memory-read-api.md) for budgets, failure behavior and usage.
