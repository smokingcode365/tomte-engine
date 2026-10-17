# Mutation output contract

Every mutation operation requires a non-null `evidence_type`, identifying how
the fact is supported, separately from its source IDs, interpretation confidence
and lifecycle operation. The prompt and field descriptions define five values:

- `USER_EXPLICIT`: directly stated or confirmed by the user.
- `RESOLVED_REFERENCE`: attribution requires contextual entity-reference resolution.
- `GROUP_EXPLICIT`: expressly asserted for each identified group member.
- `USER_IMPLIED`: indirectly conveyed by the user's wording itself.
- `DERIVED`: requires an additional reasoning step from current user evidence.

User role does not establish explicit support. No label authorizes unsupported
inference or relaxes the existing conservative attribution/mutation rules. If
meaning, target binding or category is uncertain, abstain. `LEGACY` remains a
system compatibility label for stored evidence and is forbidden for new mutations.
The mutation prompt includes a complete fictional output example; its handles
and source IDs must never be copied into a real plan.

Only the OpenAI provider's exact `MutationPlanResponse` output type uses
Pydantic-AI `ToolOutput(..., strict=True)`. This makes the Responses function
tool explicitly strict. Nullable target/attribute/value fields become required
nullable wire properties; `TargetMutationPlan.operations` is also required on the
wire. Their local defaults remain unchanged. An empty `plans` or `operations`
list is still a valid no-op.

Strict generation constrains shape, not truth. Pydantic still rejects missing,
null, invalid or legacy categories in new mutations and enforces lifecycle shape.
Engine preparation still validates current-user source IDs and target ownership.
No semantic classifier or threshold was added, and no retry, budget, persistence,
retrieval or model setting changed. Other output types retain their existing
declarations. The compatibility claim covers the installed SDK's current OpenAI
Responses route; other provider/profile combinations require separate verification.

Offline tests inspect an actual SDK request routed exclusively through a mock
transport, preserve provenance through mutation preparation and ensure the
diagnostic probe still reports malformed provider output. Synthetic replies
verify protocol behavior; they do not measure model quality. Live schema/server
acceptance and factual accuracy require a separately bounded real evaluation.
