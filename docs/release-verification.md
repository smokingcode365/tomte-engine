# Version 1.0.0 verification

Local verification completed on 19 November 2026. This records bounded evidence,
not a claim that the system is bug-free or production-ready. No new memory feature,
database migration, commit, push, Git tag or package publication was part of this
completion work.

## Changes

Three correctness fixes: injected OpenAI credentials now reach standalone and
managed clients; admin graph/wipe methods validate IDs before provider access;
the no-API evaluation preflight accepts the configured embedding model while
historical live-comparison pins remain enforced. Regression coverage accompanies
these paths. Existing tests retain their behavioral assertions; the model-selection
test now inspects the bound embedding model's name rather than its former string.

Release work adds the specification, architecture/runbook/testing documentation,
security/contribution guidance, MIT license, CI and distribution inspection.
The empty Makefile and unused `schedule` 1.2.2 dependency were removed. The direct
dotenv declaration and lock metadata were reconciled, including the already
declared tokenizer. Retained dependencies were not broadly upgraded. Local caches
and prototypes are excluded from release archives; existing local files are retained.

The source scan parsed all 104 Python files and found no unfinished concrete
method bodies. Pass/ellipsis-only methods were interface contracts; required
PostgreSQL, Redis and model-adapter methods have concrete implementations.
Historical catalogue/provider interfaces remain for compatibility rather than
being deleted solely because the default workflow no longer calls them.

## Deterministic and persistence checks

- Python 3.12.13: **1,320 passed, zero skips**, against the installed wheel in a
  clean environment with PostgreSQL/Redis integration enabled. Twenty non-fatal
  warnings came from synthetic Pydantic fixtures and an event-loop deprecation.
- Python 3.10.20: **1,058 passed, 262 skipped**, against a clean installed wheel.
  The skipped checks require explicit infrastructure flags; they passed in the
  Python 3.12 integration run. This is not a Python 3.10 infrastructure result.
- The affected credential/admin/runtime group passed 88 checks. The subsequent
  evaluator group passed 43 with one infrastructure skip before full regression.
- Demo HTTP/origin/token checks passed all 23 cases. Fresh disposable PostgreSQL
  bootstrap succeeded twice. The complete integration suite used only owned
  temporary databases and unique synthetic Redis state; cleanup and baseline
  digests matched afterward. Existing application containers/volumes were not used.
- Isolated `python -I` import resolved to clean site-packages. Dependency consistency
  checks passed. Runtime package bytes are compared with the fully tested wheel
  when producing the final version metadata.

See [testing](testing.md) for repeatable setup, `pytest tests -q`, explicit
integration flags and installed-wheel testing. The final package build uses
`uv build`; `scripts/check_distribution.py` inspects both archives. Builds also
exercise wheel generation from the source distribution.

CI defines Python 3.10/3.12/3.13 unit/package jobs and an isolated persistence job
without model credentials. Its YAML and action references were checked locally;
remote GitHub Actions has not run. Python 3.13 is configured in CI but was not
locally tested. Optional Presidio installations and alternate providers were not
live-tested; their deterministic adapter/scrubbing tests are included.

## Live semantic smoke

Three frozen synthetic development conversations ran once each: concurrent jobs,
two-car correction/repainting, and Chinese uncertainty with an assistant-only
fabricated title/salary. Seven batches committed and passed seven replay checks.
Five returned public entities matched stored public projections. Entity identities
survived seven transitions and fact identities survived sixteen comparisons.

Manual semantic review covered all 31 current-item occurrences and two historical
items across the batch snapshots, plus returned memory and generated answers.
All **16 required meaning groups**, seven structural groups and three retrieval
groups were supported; no unsupported proposition was observed. The job ending
left the other job active. The Tesla correction retained its fact UUID and did
not invent a genuine 2023 purchase history. The Honda repaint archived blue under
the same colour fact. Chinese uncertainty and early-2025 timing survived; neither
assistant-only claim became a user fact. Answering returned correct car details
and declined to invent an unknown bank account number. Read calls left durable
memory unchanged; scoped negative reads and cleanup passed.

The configured models were GPT-6.1 Sol, GPT-6 Luna, GPT-6.1 Terra and
`text-embedding-3-large` at 1,536 dimensions. Pricing was checked against
[OpenAI pricing](https://developers.openai.com/api/docs/pricing),
[Luna's model page](https://developers.openai.com/api/docs/models/gpt-6-luna) and
[embedding pricing](https://developers.openai.com/api/docs/models/text-embedding-3-large).
The persistent transport ledger recorded **100 HTTP requests**, estimated usage
cost **USD 0.33552986** and conservative accounted charge **USD 0.4613535575**,
under a USD 4.50 hard ceiling. All requests reconciled; no provider/budget failure
occurred. These are rate-based estimates, not an invoice. No other paid calls were
made for this completion work.

Source/prompt/schema hashes, model/settings versions, frozen inputs/gold, complete
HTTP/model output, budget accounting, failures and review evidence were retained
outside release source. Historical campaign results were not altered or resumed.

## Limits retained for v1

This is a small, non-independent development smoke test, not held-out benchmarking
or a reliability-rate estimate. Manual consolidation does not establish live
autonomous-worker scheduling quality. The corpus is small and recent-memory
retrieval can help. Broad campaign, large-corpus/load and process-kill testing
remain outside this release evidence.

Models can misinterpret or omit facts. Retrieval can include irrelevant background;
ambiguous owners may be omitted; capacity limits and terminal skips can lose
memory opportunities. Cross-session processing is not globally chronological.
`wipe_user_memory` remains partial erasure. The host owns authentication,
authorization, backups, deployment security and the response to memory failures.
See [specification](specification.md) and [operations](operations.md).
