# Semantic dictionary prototype

The dictionary is a **soft vocabulary stabilizer**, not a canonical ontology or
mandatory schema. Prefer two words for the same meaning over incorrectly forcing
two meanings into one word. Precision matters more than reuse rate. KEEP_RAW is
a normal successful outcome, including when no candidates or decisions exist.

## Processing boundary

Consolidation junk-filters the frozen current batch, extracts without dictionary
guidance, and validates current-batch provenance. It then prepares one entity
query and one query per extracted fact:

```text
domain: <domain>
entity_type: <raw type>
```

```text
domain: <domain>
entity_type: <raw type>
attribute: <raw attribute>
```

Neither representation includes values, names, chats or whole entity summaries.
All E + F texts use the provider's batch embedding interface. Typed SQL returns
at most five ENTITY or ATTRIBUTE candidates per target above the existing floor;
filtering occurs before limiting. The old chat-centroid bootstrap interface
remains for compatibility/evaluation, but normal consolidation no longer uses it.

One structured terminology call handles the entire batch. Entity names and exact
fact values are read-only interpretation context. Each fact includes only source
text from its validated current-batch source, never arbitrary historical chats.
Opaque candidate handles map to a specific keyword, type and anchor for one
target. The model sees handles and words, not similarity as proof of synonymy.

Only terminology alignment uses the configured cheap model (`gpt-6-luna`) through
Responses with low reasoning effort. Extraction, identity resolution, mutation
planning and contextual interpretation retain the configured strong-model route.

## Equivalence and immutable application

REUSE requires exact semantic equivalence without loss of specificity. Vector
retrieval only proposes candidates. Hierarchy is not synonymy: car/vehicle and
sedan/car remain distinct; car/automobile may be equivalent in appropriate usage.
Interior color differs from paint color, and sale completion differs from
ownership status. Generic attributes require source interpretation.

The response contains decisions only. Strict schemas forbid replacement values,
names, IDs, facts, aliases, confidence, timestamps and provenance. Code validates
all handles, types and uniqueness before applying anything to a deep copy. Only
entity_type and attribute_name can change. Values such as `dark blue` and
`Senior Software Engineer` remain exact. Invalid output or provider/retrieval/
embedding failure rejects the entire stage and proceeds with RAW extraction and
no speculative dictionary feedback. Diagnostics omit content and exception text.

## Persistence and feedback

The exact raw query vectors are retained as internal operation evidence, not
stored in memory payloads. REUSE nudges only the explicitly accepted anchor and
increments its offered keyword. KEEP_RAW creates a separate anchor, even when a
similar anchor exists. Attribute feedback is limited to accepted mutation
attribute names; proposals omitted downstream do not teach vocabulary.

Identity resolution, assertion lifecycle, contextual empty-extraction fallback
and the temporary ATTRIBUTE_ALIASES safety rail are unchanged. The dictionary
stage does not depend on that alias map. All model and embedding work precedes
the existing atomic claim/memory/feedback transaction. Feedback persistence
failure rolls back that transaction and leaves the queued batch retryable.

## Limits and evaluation

Conservative feedback permits duplicate anchors and dictionary growth. Existing
anchors may still contain legacy whole-entity vectors; this prototype performs
no migration or threshold tuning. The global vocabulary is not a tenant ontology.
Legacy alias/mutation policy may still rename keys downstream; such differences
can skip attribute feedback rather than silently teaching a different meaning.
Custom database providers must support the optional typed keyword lookup
arguments to enable this stage. Older adapters fail safely to RAW with no
dictionary feedback; the existing three-argument lookup remains callable.

Maintenance's cross-anchor prompt now rejects hierarchy, but its algorithms
remain unchanged: word-only inputs lose domain/source context, merging mixed
keyword types is possible, no durable synonym map exists, and concurrent anchor
deletion/nudging remains a risk. Missing accepted anchors fail persistence rather
than redirect feedback. Model equivalence judgments can still be wrong despite
valid schemas; deterministic tests prove safety mechanics, not semantic quality.

Run the opt-in PostgreSQL/Redis pytest suite for rollback, replay and isolation
checks. `scripts/evaluate_terminology_alignment.py` measures eight synthetic
terminology cases three times using real embeddings, typed PostgreSQL candidates
and the configured model. It uses prescribed raw extraction, so it is not an
end-to-end extraction or memory-lifecycle evaluation. It requires an empty local
dictionary, removes only its own anchors, compares full dictionary row digests
and Redis counts, and writes results under ignored `log/`.

`scripts/compare_terminology_models.py` compares the configured strong baseline
and fast model with identical fixed candidates, prompt, schema and validation:
60 unique cases, including 20 repeated three times, yield 100 scored decisions
per model. It measures model-only latency and token usage, performs no embeddings
or persistence, and saves synthetic evidence under ignored `log/`. Fixed-candidate
scores do not establish end-to-end retrieval quality or safety on unseen cases.
