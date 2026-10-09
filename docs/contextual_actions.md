# Contextual memory actions after empty extraction

## Routing gap
A correction can contain no standalone extractable fact. Previously, an empty
extraction skipped identity resolution and mutation planning, then claimed and
acknowledged the batch without correcting the memory.

## Narrow fallback
The normal extraction/identity route remains unchanged. If there are no extracted
entities, one structured contextual-action call evaluates the current batch,
frozen four-turn reference context, and at most five scoped existing candidates.
Reference-only claims are not reasserted as new facts. Candidate embeddings are
reused from the batch; no extra candidate-search embedding is required.

The small contract distinguishes CORRECTION, RETRACTION, CONFIRMATION and
CLARIFICATION. It carries a reference hint, optional quoted replacement wording,
current source IDs, interpretation confidence, and an optional resolved existing
entity/assertion. There is no NEW/INSERT response. Entity resolution confidence
and evidence use the existing ResolvedTarget contract and acceptance threshold.

The normal resolver requires extracted object IDs. Rather than manufacture an
extraction, this fallback resolves only offered database/assertion IDs. The existing
normal identity resolver is not changed. Confidence and lexical name/alias evidence
must support the contextual referent; similarity is candidate evidence only.
Ambiguity is represented by an action without a target and retained for retry.

## Mutation and acknowledgement
Resolved targets go through existing matched payload preparation, mutation planning,
assertion application, scoped snapshot validation, transactions and batch replay.
Only the resolved assertion IDs are offered to mutation planning on this route.
There is no separate correction persistence API. Identity metadata remains protected.

A missing mutation for a resolved contextual action is an error, not successful
acknowledgement. Invalid target/source references fail before writes. Unresolved
contextual actions return False before claiming the batch. A validated empty action
list may finish as a no-op. Success still commits the batch marker, memory updates
and dictionary feedback atomically; the existing worker acknowledges only success.

## Limits
This adds one model call only after empty extraction. Interpretation remains model
judgment; schema validation cannot prove semantic correctness. Unresolved batches
remain pending under the existing retry mechanism; no unresolved-memory database
or terminal retry policy is introduced. Models, confidence thresholds, embeddings,
assertion lifecycle and dictionary semantics are unchanged.
