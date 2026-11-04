# Memory-format v2 benchmark partitions

These partitions intentionally have no scored benchmark cases yet. The 40 v1
cases in the sibling dev/validation directories retain their original bytes and
labels. They use retired scalar values, assertion operations and evidence labels;
using them with the new prompts requires explicit semantic re-authoring, not
silent conversion or relaxed assertions. Use the historical Git revision to
reproduce their original results. Validation cases must not guide prompt tuning.

The ten `fixtures/harness_smoke_v2.json` cases verify mechanics only. Deterministic
engine regression tests cover the new memory behavior separately. No v2 model
accuracy score or equivalence to the old benchmark is claimed.
