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
Ambiguity is represented by an action without a target and skipped as semantic
abstention. Targets below the existing acceptance policy are likewise skipped.

## Mutation and acknowledgement
Resolved targets go through existing matched payload preparation, mutation planning,
assertion application, scoped snapshot validation, transactions and batch replay.
Only the resolved assertion IDs are offered to mutation planning on this route.
There is no separate correction persistence API. Identity metadata remains protected.

The mutation planner may abstain even after contextual resolution selects an
assertion. An empty plan drops that target entirely: no alias, fact, embedding,
row mention-count/recency or dictionary write. Other targets with accepted mutations
remain eligible. Invalid target/source references and operational failures still
fail before acknowledgement. Semantic abstention, including an empty action list,
junk or an assistant-only batch, completes successfully with only a processed-batch
marker. USER messages still require their own embeddings. Success commits the batch
marker and any accepted memory/dictionary writes atomically; the worker acknowledges
only the exact frozen snapshot, leaving later queued messages untouched.

## Limits
This adds one model call only after empty extraction. Interpretation remains model
judgment; schema validation cannot prove semantic correctness. Semantic uncertainty
is a completed no-op, while operational failures and structural violations remain
retryable. No unresolved-memory database or new public result API is introduced.
Models, confidence thresholds, embeddings,
assertion lifecycle and dictionary semantics are unchanged.

## Correction source safety

Normal extraction and contextual actions share one deterministic CORRECT gate
after mutation planning, before assertion application or PostgreSQL writes.
Valid source IDs prove provenance, not semantic support. A correction must cite
current user text with corrective intent and a replacement value bound to the
resolved identity. Stored names/aliases and the already retrieved scoped candidate
identities provide binding context; proposed new aliases do not authorize it.
Pronoun-only corrections use the most recent named preceding turn, with competing
identities causing rejection. Reference context may identify a referent but cannot
supply replacement evidence. Questions and hypothetical wording cannot authorize
CORRECT. Invalidations without replacement also require the old assertion value
in the corrective clause or prior named target context; vague complaints about
wrong paperwork alone do not establish that the stored assertion was wrong.

Until the next correction-safety step, UnsupportedCorrection still fails the whole
prepared batch before its processed claim,
memory writes or dictionary feedback. Existing truth and the Redis frozen batch
remain unchanged. There is no new persistence state, automatic downgrade to ADD,
additional model call or confidence threshold. ADD/REINFORCE and existing
SUPERSEDE/RETRACT mechanics are unchanged. Destructive invalidation merits stronger
evidence than additive learning; this focused gate currently enforces that for
CORRECT only, not every destructive operation.

This conservative English lexical policy is not a semantic proof: multiword
paraphrases, multilingual corrections, complex negation and reference chains may
be rejected or misinterpreted. Rejection defers a whole batch, including other
useful updates. Pending-batch retry/backoff remains a separate worker concern.
Do not describe bounded evaluation success as universal correction safety.
