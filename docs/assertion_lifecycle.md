# Assertion lifecycle design

## Baseline implementation and storage decision
Before this change, facts occupied attribute slots: normalization/merge collapses repeated keys,
replaces values and preserves ownership history only in descriptive timelines.
Identity resolution is already separate and matched metadata is protected.
PostgreSQL stores entity payloads as JSONB. Inspection found zero memory rows in
this local database; no historical private payloads were read. Keep JSONB: no SQL
migration, new table, dependency, model or vector change is needed.

## Proposed contract
A fact ID identifies one immutable attribute/value assertion. Retain existing Fact
fields and timelines; add assertion_state, origin, created_at, optional effective_at
and ended_at, supersedes_fact_id/corrects_fact_id, and embedded evidence. Separate
ACTIVE, HISTORICAL, RETRACTED and CORRECTED_INVALID from legacy active/past/future
status. Current and genuine-history projections exclude corrected-invalid facts;
audit projection includes everything. Multiple active values per attribute are valid.

New assertion IDs are deterministic UUID5 values scoped by entity, frozen batch,
operation and source IDs. Legacy IDs derive from stable assertion content and
entity scope; reads never allocate random UUIDs or write migration data. Preserve
legacy provenance and mark legacy past facts HISTORICAL; legacy future goals remain
valid ACTIVE assertions with their existing future status. Unknown legacy provenance
is not fabricated and remains an explicit write-validation blocker.

## Application and planning
Normal routing is extraction -> identity resolution -> mutation planning. Empty
extraction instead uses contextual action/reference resolution -> mutation planning;
see contextual_actions.md. Identity resolution remains unchanged. A separate bounded structured mutation call
receives accepted target handles, actual offered fact IDs, incoming semantic facts,
current messages and frozen reference context. It returns only ADD, REINFORCE,
SUPERSEDE, RETRACT or CORRECT proposals with source IDs and interpretation confidence.
This extra call avoids coupling uncertain identity decisions to target-specific fact
IDs. No new identity/confidence threshold is introduced. Confidence fields remain
separate, uncalibrated signals; no multiplication. Low-certainty semantics should
produce no operation. The model cannot provide new IDs, timestamps or old values.

Deterministic application validates the whole plan on copies before returning any
payload. Sources must belong to the current frozen batch; reference context is not
new evidence. Target fact IDs must belong to the particular offered target. Reinforce
keeps identity and accumulates each source once. Supersede creates a new assertion
and marks the prior assertion historical. Retraction makes no replacement.
Correction marks an assertion invalid and creates a replacement only when proposed;
it never restores earlier values automatically. Existing identity metadata and
aliases remain owned by the existing identity merge.

System recording times come from validated source-message times. Effective/ended
real-world times remain null unless independently known; relative-date resolution
is not implemented. Existing descriptive extraction timelines retain their validated
source metadata. New lifecycle evidence records which operation it supports.

## Persistence, retry and concurrency
Model/embedding work stays outside transactions. Batch claim, all entity payloads
and dictionary feedback share the existing outer transaction; Redis acknowledgement
remains after success. Batch replay returns before model/application work. Snapshot
checks under scoped row locks reject stale prepared payloads rather than overwriting
concurrent assertions; frozen batches can retry and replan. SQL transport INSERT /
MUTATE / REINFORCE is distinct from semantic mutation operations.

## Compatibility and limits
Extraction wire models exclude all new system-managed fields. Provider retrieval
remains flat and exposes adapted assertion IDs without writing. Current context and
embedding summaries project active assertions; explicit history/audit projections
are available. Terminology normalization stays conservative; the dictionary has no
reliable durable synonym map, and sale_status is not silently declared equivalent
to ownership_status. No cardinality ontology, inferred consequences, relation graph,
full event sourcing or historical auto-repair is introduced.

Tests must cover all five operations, invalid plans/sources/IDs, immutable old truth,
legacy stability across processes, multivalued attributes, corrected-history
exclusion, per-target group facts, transaction failure, stale writes and replay.

## Implementation and verified boundaries
The runtime now performs identity preparation without replacing facts, followed by
one MutationPlanResponse call. New IDs/evidence/states are owned by core/assertions.py.
SQL transport writes the deterministic full payload; snapshots are checked by
bulk_commit_updates while holding tenant/user-scoped row locks. Existing custom
providers must honor snapshot tokens to obtain the same concurrency guarantee;
transaction guarantees alone do not prove compare-and-write safety.

Legacy get_user_graph_skeleton/candidate SQL returns the original payload for snapshot
comparison. Flat retrieval adapts assertions without writes. Current formatting and
embedding project ACTIVE; explicit history exposes HISTORICAL/RETRACTED; audit retains
invalid assertions. Descriptive timelines stay on original assertions rather than
being copied into replacements as if all earlier events supported the new truth.

V1 literal values remain strings for API compatibility. JSON payloads and optional
additive schema fields leave room for future tagged entity references; no relationship
or value-reference engine is implemented. Relative effective dates are not parsed.
Legacy creation time uses the retained source-recording time, not a reconstructed
first-ever event. Already-lost historical facts cannot be recovered by lazy adaptation.
Legacy placeholder provenance is readable where structurally valid but blocks rewrites
until explicitly repaired; no migration or historical cleanup ran.

Plans omit unresolved targets. Missing mutation plans abstain from truth changes;
accepted aliases may still update the resolved identity. New empty targets are not
inserted. ADD of an already active identical value rejects the plan (use REINFORCE).
RETRACT targets ACTIVE assertions; CORRECT can invalidate active or genuine-history
assertions and never resurrect earlier assertions. Implied/derived evidence is labeled,
not governed by a newly tuned numeric threshold in v1. Single-valued semantics remain
a model planning decision; deterministic code does not enforce a cardinality ontology.
