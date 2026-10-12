# dev cases

`benchmark_v1.json` contains the dev benchmark with the v1.1 scoring amendment below.
Dev may be used during prompt tuning. Validation must not be used for iterative tuning.
The original current-prompt dev baseline has run; prompt-v2 has not run yet.

## Post-baseline v1.1 dev-only scoring amendment

A post-baseline v1.1 DEV-ONLY scoring amendment removed one single-message raw-source
constraint and two restrictive exact fact-count constraints. This corrects a
test-interface mismatch, not LLM performance. Original benchmark-v1 DEV bytes are
recoverable at Git commit `d55b938`. Original baseline raw outputs remain unchanged
in ignored run artifacts. The validation partition and expected safety semantics
were never modified.
