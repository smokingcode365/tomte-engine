# Correction safety regression

`honda_tesla_failure.json` preserves long-01 repetition 3: the complete trace,
source IDs, raw extraction, candidates/resolution, planner input/output and final
persisted synthetic records. It contains no private conversation or credentials.

`cases.json` freezes 20 legitimate and 20 ambiguous **target** corrections across
vehicles, employment, residences and school grades. Another entity may legitimately
be corrected in an ambiguous-target case; the expected target must remain intact.
The saved outputs retain the old assertion contract and confidence fields. The
current deterministic adapter maps those fixtures to stable fact updates to test
source/target authorization and immutable history. It does not reconstruct the
removed English-language correction gate. Meaning and target interpretation are
now LLM decisions; an offered, structurally valid but semantically wrong plan can
still be accepted. Labels are not supplied to model calls.

Run `uv run pytest tests/test_correction_safety.py`; enable both local integration
flags for the PostgreSQL/Redis cases. `uv run python scripts/evaluate_correction_safety.py`
runs 40 real planner cases plus ten real extraction/resolution/planning repetitions.
It uses fresh synthetic users, no daemon, unchanged configured models, cached real
fixture vectors and scoped cleanup. This is correction-policy coverage, not an
embedding/retrieval-quality evaluation. It does not claim full causal replication
of the earlier stochastic environment or comprehensive linguistic accuracy.

Generated v2 evidence stays in ignored `log/correction_safety_v2/`. The original
real-world logs are preserved. Re-run the unchanged conversation labels with:

```powershell
uv run python scripts/evaluate_real_world_conversations.py --trace-ids long-01 long-09 --output-dir log/correction_world_v2 --repeat-critical 3
```

These commands make paid real-provider requests. Run only on otherwise idle
local services so count/digest cleanup comparisons are meaningful. Live results
must be measured again for the new memory contract; historical scores do not
establish its correction accuracy.
