# Engine verification and evaluation tools

Run commands from the source checkout after `uv sync --locked`. Scripts are not
installed wheel entry points. No HTTP application is included in this library.

## Routine checks

- `bootstrap_local.py --vector-dimension 1536`: explicit local schema setup after
  PostgreSQL's vector/pgcrypto extension initialization; no destructive reset.
- `evaluate_prompts.py`: validates 27 dev/13 validation stage-contract cases by
  default. Only explicit `--live` opts into paid provider calls.
- `prompt_eval_support.py`: shared validation/scoring, used by unit tests.
- `evaluation_context_contract.py`: offline two-list context assertion, tested
  independently and used by retrieval comparison scripts.

Use `uv run pytest tests` for all deterministic engine and evaluation-contract
tests. Set both documented service flags only against a dedicated disposable
PostgreSQL/Redis setup. No paid API is needed for these tests.

## Synthetic live evaluation families

- `evaluate_real_world_conversations.py --validate-only`: checks 12 synthetic
  multi-session traces without services/models. Without that flag it is paid,
  writes scoped fixtures and requires an initially empty dictionary.
- `evaluate_real_world_report.py`, `evaluate_real_world_queue.py` and
  `evaluate_real_world_junk.py`: bookkeeping and queue/junk diagnostics for that
  family. Read each script's prerequisites; the latter can call real models.
- `evaluate_terminology_alignment.py`, `compare_terminology_models.py`,
  `evaluate_dictionary_representations.py`, `evaluate_dictionary_granularity.py`:
  distinct terminology/dictionary experiments, including paid embeddings/models.
- `evaluate_memory_retrieval.py`, `evaluate_session_entities.py`,
  `evaluate_entity_activation.py`, `evaluate_retrieval_stack.py`,
  `evaluate_memory_ordering.py`, `evaluate_session_implementation.py`: retrieval
  baselines/ablations and shared datasets. These import one another; several need
  ignored historical embedding caches. They are not equivalent duplicate tools.
- `evaluate_correction_safety.py`, `evaluate_cross_batch_context.py`: historical
  correction/context diagnostics. They are not proof of current general semantic
  accuracy or a request to reintroduce a legacy evidence gate into workflows.

These are research/development tools, not automatic release benchmarks. Some
historical inputs/results/caches are private and absent; old studies may require
contract review before a new live run. Do not treat their ablations as current
production routing or reconstruct missing labels to improve a score.
See [evaluation evidence and limits](../docs/evaluation.md).

Live runs can write synthetic DB/Redis state and incur provider charges. Review
code and scoped cleanup before explicitly opting in, use disposable services,
and never substitute private user data or reset shared volumes. No live model
evaluation is part of normal CI. Cases and expected outputs remain unchanged.
