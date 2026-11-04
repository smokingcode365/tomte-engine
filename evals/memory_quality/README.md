# Memory-quality pilot

This is a small, authored end-to-end evaluation of the current memory format.
It is separate from deterministic engine regression tests and stage prompt
benchmarks. Five conversations cover extraction, nested changes, concurrent
jobs, correction versus history, uncertainty, multilingual input, and retrieval.
Bob's basic employer case repeats three times from fresh database/dictionary
state. Seven runs contain twelve consolidation batches.

`conversations_v1.json` contains only inputs, batching, repetitions and queries.
`expectations_v1.json` contains the frozen meaning checklist and review rubric.
Gold is never passed to the engine or its models. Field names and wording are
not scored as exact matches. The reviewer must inspect all stored items,
including extra claims, and record source/snapshot evidence for judgments.

Report wrong propositions, missed required facts, update/history integrity and
returned-memory usefulness separately. Wrong propositions count five error
units; missed requirements count one. This is a prioritization count, not an
accuracy percentage. Unresolved judgments must remain visible. This pilot uses
LLM-assisted manual review, not independent blinded human adjudication.

Run from the engine checkout:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_memory_quality.py
.\.venv\Scripts\python.exe -m pytest tests/test_memory_quality_pilot.py -q
.\.venv\Scripts\python.exe scripts/evaluate_memory_quality.py --preflight
.\.venv\Scripts\python.exe scripts/evaluate_memory_quality.py --live
```

For a focused recovery follow-up, append `--recovery-pilot` to validation or
live execution. This declared plan starts with concurrent jobs and the two-car
correction, repeats each three times, and runs the other three cases once:
nine runs and twenty consolidation batches. It overrides repetitions in the
manifest without changing the frozen conversations or expectations. Its cost
estimate is $1.50, hard ceiling $2.20, and HTTP-attempt limit 320; the other
controls below are unchanged. Stage recovery summaries are saved alongside
model outputs and batch snapshots, so a successful live run can be distinguished
from one that actually exercised malformed-output recovery.

`--subject-pilot` selects the separate frozen `conversations_subjects_v1.json`
and `expectations_subjects_v1.json` files. It tests two user-job entities, two
employment facts on one brother entity, and stable car naming through repainting.
Each case repeats three times (nine runs, eighteen batches). It uses the same
$1.50 estimate, $2.20 ceiling and 320-request cap as the recovery follow-up.
The variant flags are mutually exclusive. Record whether the brother case
actually exercises repeated attribute names; a semantic pass alone is not proof
of that coverage. Earlier data and evidence remain unchanged.

`--realistic-pilot` selects `conversations_realistic_v1.json` and
`expectations_realistic_v1.json`: five eight-message conversations, twenty-five
batches and ten final queries. The cases cover similar cars, ambiguous brothers,
correction followed by a real move, mixed topics and multilingual uncertainty.
Gold includes intermediate checkpoints so later clarification cannot hide an
earlier unsupported attribution. The estimate is $2.50, hard ceiling $3.70,
HTTP-attempt limit 450 and time limit one hour. The frozen satisfactory gate
requires all core meanings, structural/checkpoint and retrieval checks, with no
observed wrong or unresolved memory. Optional omissions are reported separately.

Default validation makes no model/database calls. Preflight creates and removes
only a new random PostgreSQL schema and its synthetic Redis keys, checking that
existing table and Redis content digests remain unchanged. Check Docker first.
Live execution retains the current configured models, provider, prompts,
dictionary alignment, consolidation, persistence and retrieval logic. It invokes
consolidation directly and does not run the background scheduler.

The default live plan estimates $1.25 and enforces a $1.80 ceiling, 200 HTTP attempts,
45 minutes and 120 seconds per request. It reserves conservative input/output
cost before each request, includes retries, and stops after unknown usage,
transport/provider errors, or two consecutive failed runs. Standard tier and an
8192 output-token cap are explicit evaluation controls; cap failures must be
reported separately from semantic errors. Model rates were verified against
[official pricing](https://developers.openai.com/api/docs/pricing) on 2026-11-04.
Usage-based cost is an estimate, not an invoice; cached-token discounts are not
assumed. The estimate and hard ceiling are distinct from worst-case request
reservations. The runner warns over 120% of estimate and disallows future calls
past the lower of its hard ceiling and 150% of estimate.

Each run has a fresh isolated schema and synthetic tenant. The existing public
dictionary is not cleared or used. Every cleanup checks protected-state digests.
Exact source/model/package/config/dataset/schema fingerprints, raw model outputs,
API usage, stored snapshots and public retrieval payloads are saved under a new
ignored `log/memory_quality_v1/` run directory. Headers/credentials are excluded.

Do not tune prompts or expected answers after seeing pilot results. Preserve
failures for a separately reviewed fix. Do not interpret this small corpus as a
population accuracy estimate, a mature-dictionary test, a stress test, or proof
of long-term retrieval quality: recent-memory fallback can help these queries.
