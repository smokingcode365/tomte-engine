# Correction safety regression

`honda_tesla_failure.json` preserves long-01 repetition 3: the complete trace,
source IDs, raw extraction, candidates/resolution, planner input/output and final
persisted synthetic records. It contains no private conversation or credentials.

`cases.json` freezes 20 legitimate and 20 ambiguous **target** corrections across
vehicles, employment, residences and school grades. Another entity may legitimately
be corrected in an ambiguous-target case; the expected target must remain intact.
The deterministic tests deliberately replay erroneous plans at confidence 1 as
well as the recorded 0.88 failure. Labels are not supplied to model calls.

Run `uv run pytest tests/test_correction_safety.py`; enable both local integration
flags for the PostgreSQL/Redis cases. `uv run python scripts/evaluate_correction_safety.py`
runs 40 real planner cases plus ten real extraction/resolution/planning repetitions.
It uses fresh synthetic users, no daemon, unchanged configured models, cached real
fixture vectors and scoped cleanup. This is correction-policy coverage, not an
embedding/retrieval-quality evaluation. It does not claim full causal replication
of the earlier stochastic environment or comprehensive linguistic accuracy.

Generated evidence stays in ignored `log/correction_safety/`. The original
real-world logs are preserved. Re-run the unchanged conversation labels with:

```powershell
uv run python scripts/evaluate_real_world_conversations.py --trace-ids long-01 long-09 --output-dir log/correction_world --repeat-critical 3
```

These commands make paid real-provider requests. Run only on otherwise idle
local services so count/digest cleanup comparisons are meaningful. No worker
reliability fix or retrieval-order change is included in this milestone.
