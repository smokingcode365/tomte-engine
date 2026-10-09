# Golden Record target-set resolution

## Current pipeline and former limitation

Chat Hook reads session history, embeds the current message, updates activity,
retrieves host context, appends history, and enqueues an item with its UUID/time and
four preceding turns. Redis freezes a processing list; retries reuse it while new
messages enter another queue. Consolidation validates that snapshot, runs junk
filtering and dictionary bootstrap, extracts semantic objects, and resolves source
IDs/times before identity resolution. After safe operations are prepared, embeddings
are generated and batch claim, memory writes and dictionary feedback commit atomically.
Only the worker acknowledges the snapshot; daemon startup is unnecessary for tests.

Previously the second model call mapped each extraction-local ID to one UUID plus
INSERT/MUTATE/REINFORCE. It received the entire graph, lacked a target-set result,
and missing decisions/UUIDs could fall back to INSERT. Multiple targets could not
be represented. String ID equality was explicitly discouraged but permanent INSERT
IDs remained model labels until persistence's collision allocator handled them.

## Candidate boundary

DatabaseProvider.get_resolution_candidates is additive; custom providers must
implement it for consolidation. PostgreSQL performs a scoped bounded query, ranking
normalized exact names/aliases, compatible-domain active mentions, domain compatibility,
then current-message centroid cosine similarity. No extra model/embedding request
is added. Five candidates per object is the current cap, centralized in core/resolution.py.
The graph API remains available but is not used by the runtime resolver.

Candidate matching never decides identity. Broad domain membership and close vectors
can propose distinct vehicles together. Exact aliases may also be ambiguous.
Explicit/context matches exceeding the cap are reported as truncation; policy refuses
multi-target writes against that incomplete set. Unretrieved semantic matches remain
a recall limitation. This is not a new ontology, search engine, or calibrated resolver.

## Identity-only result and policy

ResolutionResponse contains local object correlations and NEW/MATCH_ONE/MATCH_MANY/
UNRESOLVED decisions. Each existing target carries DB UUID, resolution_confidence and
EXPLICIT/ALIAS/RESOLVED_REFERENCE/GROUP_EXPLICIT/GROUP_INFERRED/SEMANTIC_CANDIDATE evidence.
Confidence is an uncalibrated model assessment; the separate 0.9 threshold is policy.
Targets must belong to that object's offered, scoped candidates. Repeated/foreign
object results, duplicate target UUIDs and invalid cardinality fail clearly.

Missing decisions and UNRESOLVED produce zero permanent operations. NEW additionally
requires an explicit-new assertion and a verbatim current-message introduction;
novelty is still a semantic model judgment, not proven by quote membership alone.
There is no unknown-name INSERT fallback or persistent unresolved-memory store.
The workflow exposes last_resolution and logs policy rejection counts/conditions
without logging private fact values. Successful unresolved batches are claimed once
and may be acknowledged/discarded; the absence of memory writes is intentional.

Only strong evidence above threshold can write. Similarity and GROUP_INFERRED alone
cannot. Reference/group evidence requires an active name/alias anchor in supplied
current/reference text. MATCH_MANY additionally requires applies_equally_to_all,
at least two strong targets and no truncated explicit set. An unspecified "one of"
member cannot be expanded. No generic NLP reference-object subsystem is introduced.
For "my cars" with Tesla/truck active and Honda only inferred, Honda gets no hard fact.
If a broad group has no stable anchors, this initial policy may conservatively abstain.

## Integration with assertion mutation planning

Accepted existing targets preserve canonical metadata and aliases while accumulating
incoming semantic proposals. Multiple extracted objects mapping to one UUID prepare
one operation. Group updates preserve each target's metadata and offer its own fact
IDs; resolution must say the update applies equally. A separate bounded mutation
planner then proposes ADD/REINFORCE/SUPERSEDE/RETRACT/CORRECT transitions. Deterministic
application validates sources and targets before SQL transport: NEW uses INSERT,
existing targets use REINFORCE only for all-REINFORCE plans and MUTATE otherwise.
Fact.confidence never authorizes an identity match. See assertion_lifecycle.md.

Empty extraction follows the separate contextual-action/reference resolver, then
the same mutation and persistence path. Unresolved contextual actions retain the
frozen batch; ordinary unresolved extraction results may finish as a no-op. See
contextual_actions.md for that distinction.

Model calls and embedding preparation stay outside PostgreSQL write transactions.
All expanded targets, processed marker and dictionary feedback share the existing
outer transaction. No child operation commits independently. A failed group write
rolls back even an earlier target write; retries use unchanged frozen context and,
when DB state is unchanged, identical candidate/resolution input. External DB changes
can legitimately change candidates on retry; they are not cached in Redis.

## Lifecycle and verification limits

The four-turn window and Redis capture/freeze/ack behavior are unchanged. Scoped
history separates tenants/users/sessions. Acknowledgement removes processing context
without deleting later queued context; history has its existing 24-hour TTL. Pending
unacknowledged batches intentionally persist for retries and require worker/manual
acknowledgement to finish their lifecycle.

Tests prescribe local semantic decisions and verify contract/policy/persistence;
they cannot prove real-language interpretation. They cover name/alias/pronoun matches,
similar distinct vehicles, safe/unsafe groups, uncertain nicknames, new introductions,
low confidence, inferred members, truncation, scoped search, group rollback/replay,
provenance and late same-batch alias input. No OpenAI evaluation was run here.
Controlled synthetic real-provider evaluations have since exercised normal mutation
and contextual fallback; these limited samples do not establish general AI accuracy.
The current checkpoint includes assertion correction/retraction, while models,
embeddings, confidence thresholds and dictionary behavior remain unchanged.
