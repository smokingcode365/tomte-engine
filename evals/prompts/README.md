# Tomte stage prompt evaluation harness

This harness evaluates **raw structured model output**, separately from final
persistence. It performs no retrieval, candidate discovery, embeddings, PostgreSQL
or Redis operations. Passing a stage benchmark does not prove Tomte's complete
memory system is correct. Final persistence behavior will be tested separately.

The current v2 partitions are empty pending semantic re-authoring of the 40
historical v1 cases. Those original cases and their saved results remain unchanged;
see `cases/v2/README.md`. `fixtures/harness_smoke_v2.json` contains ten artificial
mechanics fixtures and fake outputs, explicitly not an accuracy benchmark.

## Validation and execution

Run from `tomte-engine` in the existing uv environment:

```powershell
uv run python scripts/evaluate_prompts.py
uv run python scripts/evaluate_prompts.py --cases evals/prompts/fixtures/harness_smoke_v2.json
uv run pytest tests/test_prompt_eval_harness.py -q
```

Without `--live`, the evaluator validates **all loaded files before filtering**,
prints counts, SHA-256 file hashes and a selected-case digest, then exits without
constructing a provider or writing results. Empty placeholder partitions are valid.
Invalid files/selection produce exit code 2. No default command calls a model.

Future approved execution requires explicit opt-in:

```powershell
uv run python scripts/evaluate_prompts.py --live --partition dev --stage junk --limit 1
```

Requires private `OPENAI_API_KEY` from the environment or the ignored `.env`;
neither database URL is required. **Do not run the smoke fixture as a paid benchmark.**
Live mode defaults to dev; validation requires explicit `--partition validation`.
Every repetition incurs additional model cost. `--repeat N` defaults to 1; repeats
are never inferred from severity. Failed model calls become sanitized failed records;
provider/configuration initialization failures exit before starting a run.

Options: repeatable `--cases FILE_OR_DIRECTORY`, `--stage`, `--case-id`;
`--partition dev|validation`, positive `--limit`, positive `--repeat`, `--live` and
optional `--prompt-set FILE`. Limits apply after filtering, before repetition.
Case discovery and selection are deterministic. Files must stay within this checkout.

## Partitions

Dev cases may be used while writing/tuning prompts. Validation is a locked comparison
set, **not statistically independent/blinded research data**. Once prompt-v2 tuning
begins, validation cases/results must not be used for iterative prompt editing.
Run validation only at designated checkpoints. This is a development protocol,
not an automated guarantee that humans have not seen validation data.

## Version-2 case contract

Each JSON file has `schema_version: 2` and a `cases` array. Each case requires
`case_id`, integer `schema_version: 2`, `stage`, `partition`, `severity`, `tags`,
`input`, and `expected`. IDs are globally unique across loaded files. Partitions are
dev/validation; severity is CRITICAL/HIGH/MEDIUM/LOW. Unknown stages, fields, metadata,
duplicate IDs and malformed expectations fail validation; none are silently skipped.

`case_schema_v2.json` describes an individual case, including conditional stage-specific
expectations, exported by `scripts.prompt_eval_support.case_json_schema()` and
checked for drift by an offline test. Cross-field coverage/reference checks are
runtime validations; JSON Schema does not express those input-dependent sets.
Runtime validation is authoritative. `input` is the frozen, exact
logical payload supplied to the stage, not a query from which the harness retrieves
or constructs candidates. JSON objects are serialized with Unicode preserved; exact
maintenance text stays unchanged. Input envelopes/consumed handles are validated;
this is not a duplicate implementation of every production input builder.

Expectations are invariant matchers. Strings are compared exactly, without synonym,
English-language, negation, ownership or plausibility heuristics. Required matchers
use maximum one-to-one matching so a single output cannot satisfy two requirements.
No LLM grades output; no stage is graded by exact numeric confidence or free-text
reasoning. Production output-schema validation still applies to model results.

## Stages and expectations

- **junk:** `expected_junk` boolean; exact bool, not integer truthiness.
- **extraction:** `required_entities`, `forbidden_entities`, `required_facts`,
  `forbidden_facts`. Entity selectors constrain `entity_name`, `entity_domain`,
  `entity_type`, optionally `entity_id`. Fact matchers constrain `entity_name` or
  `entity_selector`, `attribute_name`, `value`, `required_source_chat_id`. Optional
  `exact_entity_count`, `exact_fact_count`, `require_unique_entity_ids`. Temporary
  ID spelling otherwise does not matter. Required and forbidden propositions are
  frozen by case authors, never inferred from language by the scorer.
- **terminology:** `decisions` with `entity_handle`, optional `fact_handle`,
  `expected_decision` KEEP_RAW/REUSE and, only for REUSE,
  `expected_candidate_handle`. Case expectations must cover every offered entity
  and fact target, explicitly including KEEP_RAW. Omitted **model output** decisions
  count as KEEP_RAW. Unknown or
  borrowed candidate handles and duplicate target decisions fail.
- **field_terminology:** `decisions` use entity, fact and field handles. Candidates
  are offered only within the selected parent vocabulary context. KEEP_RAW and
  REUSE follow the same offered-handle checks as parent terminology alignment.
- **history_selection:** `choices` select exact offered fact UUID/history indices;
  optional `coverage` states expected same-episode coverage decisions. Duplicate
  or unoffered selections fail. The engine separately checks item containment.
- **identity_resolution:** `resolutions`, each with `extracted_entity_id`,
  `expected_kind`, `required_target_ids`, `forbidden_target_ids`; optional
  `allowed_evidence_labels`, `allow_additional_targets` (default false),
  `explicitly_new`, `evidence_quote_contains`, `exact_evidence_quote`. Target IDs
  use the production UUID string form. MATCH_ONE requires one expected target;
  MATCH_MANY requires at least two and an equal-distribution output. UNRESOLVED
  has no targets. The extracted objects must have unique production `id` values;
  both the candidate map and expected resolutions must cover exactly those IDs.
  Expected required/forbidden targets must be offered for that specific object.
  Aliases are not scored: the current schema cannot authorize them.
- **contextual_action:** `expected_action_count` integer or `{minimum, maximum}`,
  optional `require_no_action`, `required_actions`. Matchers constrain `action_type`,
  `required_target_db_id`, `required_target_fact_id`, `require_unresolved`,
  `required_source_ids`, `forbidden_source_ids`. Returned entity/fact targets must
  be offered; reference hints are not scored.
- **fact_mutation:** `required_operations`, `forbidden_operations`, `allow_noop`;
  optional `exact_operation_count`, `allow_additional_operations` (default false).
  Matchers constrain `target_key`, `operation`, `target_fact_id`, `attribute_name`,
  `fields` (a map of text/confidence items), `remove_fields`, `update_mode`,
  `change_type`, `history_index`, `required_source_ids`, `forbidden_source_ids`.
  Omitted matcher fields impose no constraint; explicit null fields require null.
  Required operations must still appear even when allow_noop is true. Extra
  operations fail by default. Item confidence is checked when a field-value matcher
  specifies it; offered handles remain checked.
- **cross_anchor_merge:** `expected_merge` boolean; false/missed merge distinguished.
- **internal_keyword_merge:** `required_pairs`, `forbidden_pairs`, each exactly two
  distinct words. Returned groups become unordered pairwise relationships. Keyword
  text/case stays exact; invented words fail. Unlisted pairs fail by default with
  EXTRA_SYNONYM_GROUP (counted as false merges); `allow_additional_pairs: true`
  explicitly permits partial scoring but still checks forbidden pairs. Input is the exact
  production JSON text containing `keyword_type`, `keyword_context`, and `keywords`;
  the harness validates that envelope before scoring offered words.

Expectation target/fact IDs must reference the frozen offered input, including
fact ownership when a target is specified. Required/forbidden source IDs must
exist in CURRENT_BATCH.item_id for mutation/contextual rules and extraction
FactRule source constraints. This structural check allows assistant sources;
it does not enforce persistence authorization or judge attribute/value text.

Each score contains `passed`, named `checks`, and machine-readable `failures`.
Schemas/expectations are versioned so future schema additions are explicit migrations.
Evaluator version 2.1 preserves rejected-proposal diagnostics while scoring.
Malformed items score `INVALID_STRUCTURED_OUTPUT` even if runtime recovery safely
discards them and the remaining empty output would otherwise satisfy a no-op case.
This does not change the engine's recovery behavior or the frozen case schema.

## Prompt comparisons and routing

Current prompts and output schemas are imported directly from production. Junk uses
`is_junk`; extraction uses `parse_structured` with ExtractionMemorySchema and no
taxonomy/lexicon; all other stages use the production generic structured route.
Current configured model routing and reasoning settings are preserved.

For future prompt-v2, `--prompt-set` accepts a JSON object mapping stage names to
alternate full prompt strings. No alternative prompts are authored in this step.
Generic stages pass overrides normally. Extraction/junk require temporary agent
bindings because those provider routes ignore a supplied system_prompt. Bindings are
restored on success or failure; production files/constants are never rewritten.
Version 2 does not enable legacy dictionary injection. Use the same case files and
scorers for comparisons; only prompt hashes should differ. Evaluator source hashes
also detect scoring-code changes between runs.

## Reports, usage and reproducibility

Live output is confined to ignored `log/prompt_evals/<run-id>/`:

- `manifest.json`: commit, dirty-tree flag, UTC time, partition/stages, repeats,
  selected IDs, configured strong/cheap/embedding models, exact imported prompt
  hashes, canonical output-schema hashes, raw case-file hashes, evaluator source
  hashes/version and case schema version. Bool schemas hash `{"type":"boolean"}`.
- `results.jsonl`: one record per case/repetition, score/checks/reasons, raw structured
  output, latency, retry count and safe usage metadata. No environment dump or input
  payload logging. Model exception bodies are replaced by MODEL_CALL_FAILED.
- `summary.json`: pass/fail counts by stage, severity and partition; reason counts;
  separate critical/high failures and mutation, identity, extraction and false-merge
  counters. Counts are **case/repetition records**; unique_cases is separately reported.
  Reason counters count failed checks, potentially more than one per record;
  wrong_target_count counts identity case/repetitions with any target violation. There is
  no weighted accuracy score.
- `usage.json`: observed model names, requests, tokens and retries; token completeness
  flags distinguish unavailable usage from measured zero.

Native Pydantic-AI capture provides usage/retry metadata; no HTTP transport is patched.
Request counts describe model turns, not transparent SDK transport retries. Failed
responses may have unavailable token usage. Logging levels and temporary agents are
restored in finally/context managers. Authorization headers, HTTP response headers,
API keys and provider exception bodies are never included in reports.

Keep all future case data synthetic. Stage scores do not replace end-to-end transaction,
Redis ACK, provenance-filtering or durable-memory tests.
