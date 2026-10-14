# dev cases

`benchmark_v1.json` contains the dev benchmark with the v1.1 scoring amendment below.
Dev may be used during prompt tuning. Validation must not be used for iterative tuning.
Historical live baseline and prompt-development results are recorded in private
development evidence, not as a current public benchmark result. See
[release evaluation disclosure](../../../../docs/evaluation.md) for limitations.

## Post-baseline v1.1 dev-only scoring amendment

A post-baseline v1.1 DEV-ONLY scoring amendment removed one single-message raw-source
constraint and two restrictive exact fact-count constraints. This corrects a
test-interface mismatch, not LLM performance. Original benchmark-v1 DEV bytes are
recorded at private development commit `d55b938`, which is not included in the
clean public history. Original baseline raw outputs remain in private ignored
run artifacts and are not distributed with this candidate. The validation partition
and expected safety semantics were never modified.
