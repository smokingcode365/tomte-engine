# Real-world conversation evaluation

Run from `tomte-engine` with its uv environment and healthy local PostgreSQL/Redis:

```powershell
uv run python scripts/evaluate_real_world_conversations.py --validate-only
uv run python scripts/evaluate_real_world_conversations.py
uv run python scripts/evaluate_real_world_report.py
```

This invokes the configured real OpenAI models and consumes API tokens. It does
not generate answers. Frozen traces require no generation-model calls. Use
`--limit 1 --repeat-critical 1` for a small harness pilot. Default execution runs
12 traces and repeats traces 01 and 09 three times in total.

## Inputs and hidden truth

`traces.json` contains only identifiers, roles, text, sessions, and relative time.
`truth.json` holds entity/fact expectations, turn annotations, and retrieval
checkpoints. Only conversation text enters ChatHook; labels never reach models.
Manual traces are authored; noisy traces are fixed seeded wording variations;
adversarial traces add ambiguity and long interruptions. These are synthetic
scenarios, not a representative sample of human chat. Each trace has 54 turns.

## Execution and safety

The harness calls real ChatHook and ConsolidationWorkflow, with the daemon
explicitly disabled. It freezes/acknowledges real Redis batches, checks successful
batch replay, and records actual structured model inputs/outputs and HTTP usage.
Two-day gaps use a queue clock hook and age only the harness's PostgreSQL rows;
this does not exercise wall-clock idle scheduling. All provider persistence uses
normal short transactions. An initially empty dictionary is required: dictionary
feedback is global, and an existing dictionary must never be overwritten merely
for evaluation. Run on otherwise idle local infrastructure.

Cleanup deletes only the unique synthetic tenant namespace and recorded new
anchor IDs, then compares all six table counts/row digests and Redis key counts/
digests. It never resets volumes or flushes Redis. BIGSERIAL sequence increments
are not reset. Cleanup is in `finally`, but forced process termination can bypass
it; do not kill the evaluation while it is running.

## Results and limits

Ignored outputs live in `log/real_world/`: progress, model calls, API usage,
cleanup evidence, JSON report, ranking CSV, and an HTML drill-down. Embeddings
are cached by unchanged model, dimension, role, and text. Model results are not
cached; critical repetitions use real model calls.

Ranking replays identical live candidates with no extra retrieval SQL. Missing
cosines for recent/session-only candidates are recorded and use zero in scored
policies. Persona is excluded. Scoring matches expected entity names/aliases and
attribute/value patterns. Checklist precision is not semantic fact precision;
unmatched supported facts, synonyms, and inferred facts need human review.
Unknown/assistant-only/filler-only evidence is flagged separately. Corrections
and color history have explicit expectations; conservative ambiguous resolution
is allowed. True race/load tests and broad semantic adjudication are out of scope;
the deterministic PostgreSQL/Redis suite supplies separate concurrency coverage.

## Offline scorer checks

`tests/test_real_world_evaluation.py` supplies four no-network tests for input/truth
separation, domain-qualified event versus vehicle identity, exact color values,
and empty-set ranking stability. The offline report helper corrects identity
bookkeeping and recomputes rankings from recorded candidate metadata. Raw live
results remain available separately. Prefer `analysis.json` and `analysis.html`
for the adjudicated deterministic checklist and ranking results.

Profiles share a deliberately small entity catalog. Multiple traces reuse six
users and dictionary vocabulary; executions are correlated rather than twelve
independent user samples. Rich natural-language accuracy, legal ownership
interpretation, and false-reuse synonym judgments are not automatically inferred
from lexical checklist scores.

## Bounded operational probes

```powershell
uv run python scripts/evaluate_real_world_queue.py
uv run python scripts/evaluate_real_world_junk.py
uv run python scripts/evaluate_real_world_report.py
```

The queue probe uses only isolated synthetic Redis keys. It tests sequential
freeze/late-arrival/acknowledgment ordering, duplicate delivery IDs, stale ack,
and frozen retry stability; it does not claim concurrent-client coverage. The
junk probe invokes the actual consolidation job synchronously with a selector
hook limited to two iterations. Real junk filtering is used; no daemon starts.
The probe records repeated selection rather than fixing production behavior.
Both probes verify scoped cleanup. Never remove the iteration guard.

The initial evaluation refined hidden labels and scoring after execution began;
inputs were frozen, but this is not a blinded held-out benchmark. All required
ranking targets had session evidence, so semantic-only recovery and eviction
under a much larger stale MRU population still need separate cases. New-topic
rise speed and general semantic fact precision are not established. A name-based
later retrieval hit does not prove its returned facts are correct.
