# Longer-conversation campaign preparation

The six `conversations_long_v1.json` cases and their separate
`expectations_long_v1.json` gold implement the Stage 1 dataset in the
[evaluation plan](../../docs/memory-quality-evaluation-plan.md). They have been
authored before any live execution. No new memory-quality result is claimed.

Each case contains 40 user messages and 16 authored assistant replies across
eight sessions spanning 28 simulated days. Sessions have two batches containing
three and two user messages, respectively, plus assistant context. Each case has
eight checkpoint expectations and eight queries: four at intermediate sessions
and four at the end. The full three-repeat campaign is 18 runs, 720 user messages,
288 batches and 144 queries. The first two development runs are included in those
totals, not additional repeats.

Development cases are `concurrent_work`, `brothers_and_work` and
`everyday_threads`. Held-out cases are `two_civic_cars`, `home_and_plans` and
`bilingual_sister`. Holdout means excluded from tuning; the same author prepared
all inputs and gold, so this is not independently blinded annotation.

English conversations contain 1,762-1,808 words each in user messages, averaging
44-45 words per message. The mixed-language case contains 6,566 user characters
(10,708 UTF-8 bytes); its 14 entirely English messages average about 46 words.
These counts exclude assistant replies. Provider token usage must be measured
during execution, including schemas, prompts, memory context and outputs; raw
chat word counts alone are not the API bill.

## Offline commands

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_memory_quality.py --long-pilot
.\.venv\Scripts\python.exe scripts/evaluate_memory_quality.py --long-pilot --long-phase smoke
.\.venv\Scripts\python.exe -m pytest tests/test_long_memory_campaign.py tests/test_memory_quality_pilot.py -q
```

`--long-phase development` selects the seven remaining development runs;
`validation` selects nine held-out runs. Phase selection in this command validates
the declared schedule only. `--long-pilot --live` fails before reading credentials
or opening services: live execution uses the separate campaign runner and its
persistent, approved budget instead of a short-pilot allowance.
The existing no-model `--preflight` checks new-schema/scoped-key isolation.

The executor now switches session IDs and message timestamps, takes snapshots
after each batch, and runs each query in its own session at its declared boundary.
Queries are never consolidated and never supply recent-chat context to subsequent
ingestion or other queries. A durable-memory change or error during retrieval
fails the case. Historical short pilots also get independent query sessions now;
their earlier saved outputs and datasets remain unchanged.

Simulated message timestamps do not advance the PostgreSQL server clock or Redis
TTL clock. These tests assess temporal language and session routing, not weeks of
real aging, retention or service uptime. No worker scheduler runs during the pilot.

## Proposed Stage 1 live budget

Estimate: **USD 32**. Proposed whole-campaign ceiling: **USD 48**. These amounts
cover all 18 Stage 1 runs, including the initial two, repetitions, repair attempts,
embedding and retrieval calls. They exclude later large-corpus and answering-agent
stages. This proposal does not authorize or enable a paid run.

At the [official standard short-context prices](https://developers.openai.com/api/docs/pricing)
checked on 2026-11-04, GPT-6.1 Sol costs $2/$10 per million input/output tokens,
and GPT-6 Luna costs $0.10/$0.50. [text-embedding-3-small](https://developers.openai.com/api/docs/models/text-embedding-3-small)
costs $0.02 per million input tokens. The existing models remain unchanged.

The estimate allows four strong-model calls per batch: 288 × 4 = 1,152 calls,
averaging 8,000 input and 1,000 output tokens. That contributes $29.952. The
remaining $2.048 covers cheap-model calls, embeddings and estimation headroom.
The earlier short pilot averaged 3.76 strong calls per batch, 3,309 input and 227
output tokens per strong call. The larger assumptions allow more context, more
facts and longer structured output; they remain estimates, not measured long-run
usage or a completion guarantee. Current prices must be rechecked if execution
occurs after they change. Do not infer cached-input discounts.

Proposed controls: at most 5,000 HTTP attempts, six hours of active execution and
120 seconds per request, standard tier, bounded output and existing request-size
limits. Use a single persistent campaign ledger across phases and restarts; reserve
conservative cost before dispatch, retain unknown reservations, and stop rather
than reset the budget after an error. Existing short-pilot budgets stay unchanged.
Warn above 120% of estimate. No future request may exceed the lower of the approved
ceiling and 150% of estimate. Conservative accounting can stop a run before the
nominal billed-cost ceiling; unused budget is not a reason to spend more.

`run_memory_campaign.py` prepares an unapproved manifest and implements the
larger-budget ledger and phase review gates. Its `--prepare --campaign-dir
log/memory_quality_campaign/<name>` mode makes no model/service calls. Each live
`--phase smoke|development|validation` requires the saved approval record; only
set `approved` after obtaining real spending authorization. Review evidence must
be saved before advancing beyond the first two runs or entering validation.
An incomplete attempt cannot silently rerun; unknown billing or corrupt ledger
state blocks resumption for review. A single-writer lock prevents overlapping
processes from spending the same allowance. Short-pilot defaults still reject
ceilings at or above $5 unless an explicit approved ceiling is supplied.

Record the accepted ceiling and frozen dataset/engine/prompt hashes in the
execution manifest. Stop after the first two
development runs for semantic review; proceed to remaining development only if
the harness and outcomes are satisfactory, and freeze any fixes before validation.
Any wrong-subject write, corrupted history or isolation breach stops progression.
Failed or incomplete runs remain visible and consume the same campaign allowance.

The stories are deliberately explicit about ambiguity, unknowns and corrections.
This makes the cases reviewable but also cues the model. Success would therefore
support these authored behaviors, not naturalistic accuracy; later cases should
include less explicit everyday phrasing without replacing these frozen inputs.

## Preparation verification

On 2026-11-04, offline verification passed 12 session/query harness tests, 11
campaign-budget tests and the related evaluation group. A final full regression
passed **868 tests, zero failures/skips**, with PostgreSQL/Redis integration
enabled and real model HTTP blocked. There were 21 serializer/deprecation
warnings. New temporary database/schema cleanup and protected-state checks passed.
These checks establish runner mechanics and budget behavior, not memory accuracy
on the new conversations. No paid Stage 1 request has run during preparation.
