# Optional BM25 retrieval

`STORED_RETRIEVAL_MODE=vector` remains the default. It retains the union of the
top entity summaries and the top distinct entities from plain-field vectors.
The optional `bm25` and `hybrid` modes apply to stored entity retrieval through
`get_relevant_entities`, `get_relevant_memories`, `get_context`, and every search
in `answer_question`. Public method signatures and entity JSON are unchanged.
Mutation candidate retrieval, legacy chat/session reads, profile/frequency reads,
unresolved memories and pending-message windows keep their existing behavior.

## Setup and rollback

1. Run the application's normal `db.ensure_infrastructure(vector_dimension)`
   bootstrap. This adds the derived `tomte_memory_lexical` table, its owner/GIN
   indexes and an internal SQL `lexical_indexed` marker on each canonical row.
   This schema step is required before running updated writers even in vector
   mode; the per-user backfill below is required only for lexical reads.
2. Initialize Tomte normally and backfill each existing tenant/user:

   ```python
   admin = Tomte.create_tomte_admin(tenant_id="my-app")
   count = admin.rebuild_lexical_index(user_id="user-123")
   ```

3. Set `STORED_RETRIEVAL_MODE=hybrid` in the environment or `.env` and restart
   application processes. Use `bm25` to inspect lexical-only retrieval.

The backfill makes no embedding or generative calls and does not rewrite memory,
history or vectors. It is atomic per user, idempotent, guarded by the existing
user lock, and reads canonical rows with `FOR UPDATE`. Busy locks and database
timeouts propagate; retry later. No new PostgreSQL extension or Python package
is required. An incomplete index raises an actionable configuration error in
lexical modes; vector mode continues to work. Bootstrap does not backfill data.
Enable the mode only after all users served by that process have been backfilled.

Rollback is a configuration change to `vector` followed by a restart; stored
memories and both vector indexes remain intact. New and changed memories maintain
the lexical index regardless of the active read mode, so switching modes later
does not make those rows stale. This adds database write/storage work even when
vector reads are selected.

## Indexed documents and scoring

Each current or historical field **text** is one lexical document. Entity names,
types, domains, attribute/field labels, UUIDs and confidence are not added to its
search text. A whole field is tokenized without the embedding index's 2,000-character
chunk boundary, so identifiers spanning that boundary are preserved.

PostgreSQL's built-in parser and `pg_catalog.simple` dictionary tokenize and
lowercase text. There is no Python language interpretation, stemming, synonym
mapping, manually chosen keyword, or language-specific stop-word list. Token
frequencies come from `ts_debug` lexemes, not the capped position lists in a
`tsvector`. The query uses the same configuration and unique query terms.

Stored documents retain frequencies and lengths in JSONB. SQL computes BM25:

```text
IDF(term) = ln(1 + (N - df(term) + 0.5) / (df(term) + 0.5))
score = sum over matching query terms:
    IDF(term) * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / average_length))
```

This is BM25 with `k1=1.2`, `b=0.75`, positive IDF and no repeated-query-term
boost. `N`, document frequency and average length use all indexed current/history
field documents belonging to the requested tenant/user, excluding the unresolved
domain. They are not calculated over vector candidates or only keyword matches.
Other users cannot affect scores. An entity's lexical score is its highest field
score; multiple matching facts do not consume multiple entity result slots.
BM25 mode returns up to `limit` matching entities, and an empty list when no terms
match. Internal scores and index markers are not exposed in public entity JSON.

## Hybrid ranking

Hybrid performs independent vector and BM25 searches; lexical candidates can
therefore rescue an entity outside the original vector selection. It takes a
candidate depth of `max(20, 2 * limit)` per channel and uses equal-weight
reciprocal rank fusion, `sum(1 / (60 + rank))`, with UUID tie-breaking.

The vector channel starts with the original two-list union in its existing order,
then fills the deeper candidate list with additional vector results. The final
result count equals that original union's actual size, at most `2 * limit`.
For example, if vector retrieval returns seven entities for `limit=5`, hybrid
also returns seven; it does not claim improvement by returning extra memories.
When no lexical document matches, hybrid returns the original vector list.

Small histories remain complete. For large histories, BM25 ranks historical
values from the same full-user corpus; hybrid fuses those handles with vector
history rankings per entity. Existing byte limits, recording-order fallback,
immutable history and `history_is_partial` remain in effect. A relevant value
can still be omitted if it cannot fit the configured history budget.

## Consistency and limitations

Canonical payloads, summary vectors, labelled mutation vectors, plain retrieval
vectors and lexical rows commit in the same PostgreSQL transaction. A semantic
update replaces that entity's lexical rows from its validated canonical payload;
removed/corrected current fields leave no stale hits, and archived fields remain
searchable. Mention-only updates need no re-tokenization. Deleting canonical
memory cascades to its lexical index. No queue/cache maintenance is added.

The mode adds database work, not generative model calls. Public search still
embeds queries through the existing API; hybrid, unresolved retrieval and pending
windows use those vectors. Normal answering calls remain unchanged.

Lexical matching is not substring search or reasoning. `ERR1001` and `ERR1002`
are different terms, but matching common words can still return a wrong device
or a row without the requested code. Punctuation and compound identifiers follow
PostgreSQL's parser rules. The simple dictionary is not a multilingual word
segmenter; languages without spaces and very long individual tokens need separate
evaluation. History repetition affects corpus statistics. Hybrid can displace a
useful semantic hit, so it is opt-in, not a guaranteed accuracy improvement.

Corpus statistics are computed on reads rather than maintained in a second
counter system. This favors simple transactional correctness at the current
scale; large-user latency needs measurement before claiming scalability. Custom
database providers must implement the optional ranked-search/history/rebuild
methods to support lexical modes; existing providers can keep vector mode.

## Verification

`tests/test_bm25_search.py` covers configuration, deterministic fusion, shared
read routing, scope validation and the guarded model-free admin API.
`tests/test_bm25_postgres.py` covers the score formula, exact codes, long fields,
owner-scoped statistics, distinct entities, atomic updates/rollback, history,
backfill/readiness, deletion and a concurrent writer/backfill conflict.
`tests/test_answer_question.py` verifies that answering uses the configured mode
without adding model calls.

Run the focused offline checks with `python -m pytest tests/test_bm25_search.py
tests/test_answer_question.py`. Real PostgreSQL checks are opt-in through
`TOMTE_RUN_POSTGRES_TESTS=1` and a local `POSTGRES_URL`; use an isolated database
with pgvector already bootstrapped. The integration fixture creates and removes
only its own generated tenant/user rows. No model HTTP is allowed by the test
suite. Retrieval-quality evaluation is separate from these deterministic tests.
