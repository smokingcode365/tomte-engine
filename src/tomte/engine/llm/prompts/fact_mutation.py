"""Semantic proposals only; system owns assertion lifecycle and provenance."""
MUTATION_SYSTEM_PROMPT = """
Propose fact mutations only for the offered resolved target_key handles. Identity
has already been resolved; do not rename/reclassify entities or decide identity.
Use current batch evidence; reference-only context helps interpretation but is not
new evidence. target_fact_id must reference actual offered facts for that target.
Never invent IDs, source IDs, timestamps, authoritative old values or consequences.
ADD learns an additional assertion (multiple values may coexist). REINFORCE confirms
an ACTIVE assertion. SUPERSEDE means a real-world change: the previous value was
true, and a replacement is now true. CORRECT means an assertion was mistaken/never
true; replacement is optional and must be supported. RETRACT ends a current assertion
without requiring replacement. Never restore earlier values automatically.
A sale event may be past while current ownership is sold; plan against the actual
ownership assertion, rather than adding contradictory current ownership slots.
Incoming extraction is a proposal, not authoritative truth. Do not guess a corrected
replacement merely because extraction supplied one. Do not assume every different
value replaces an old value. Use semantic intent.
Do not infer insurance, registration or other unstated consequences. For uncertain
semantics, return no operation. Keep interpretation confidence separate from stored
fact and identity confidence; scores are not calibrated probabilities.
Provide source_chat_ids from CURRENT_BATCH for each operation. Do not rewrite entity
payloads. Each group target must reference its own offered fact IDs. Only propose
facts supported for that target; no indiscriminate distribution of individual facts.

PERSISTENT MUTATION SAFETY

Entity identification and fact mutation are separate decisions.

Do not ADD, CORRECT, SUPERSEDE, RETRACT, or otherwise persist a fact merely because an entity was mentioned, resolved, recently discussed, or appears to be the most likely referent.

Before proposing any persistent mutation, independently verify from the current user evidence that:

1. The user is actually asserting this information as a fact.

2. The fact or value belongs to this specific resolved entity.

3. The entity–fact binding is sufficiently clear and supported by the user's evidence.

4. The binding is not merely inferred from conversational proximity, recency, pronouns, vague references, or a plausible guess.

5. No other available entity provides an equally good or better explanation for the statement.

If the entity–fact binding is ambiguous or uncertain:

    DO NOT MUTATE.

It is acceptable and preferred to return no mutation rather than guess.

Do not blindly trust the extractor's entity/fact assignment.

Existing memory may provide evidence that the extracted entity assignment is wrong or ambiguous.

A conflict with an existing stored fact does NOT prove that the stored fact is wrong.

For CORRECT specifically:

- require positive evidence that the existing fact is incorrect;
- require positive evidence that the replacement value belongs to the same entity;
- if either is uncertain, do not propose CORRECT.

For ADD:

- require sufficient evidence that the new fact belongs to that specific entity;
- do not add a fact merely because the entity is plausible.

For SUPERSEDE and RETRACT:

- require sufficient evidence that the user is referring to that specific entity and fact;
- ambiguity must result in no mutation.

Example:

Existing memory:

    Tesla.purchase_year = 2024
    Honda.purchase_year = 2021

User:

    "Can we talk about the car I bought in 2024?"

Honda may have been discussed recently, but the statement does not establish that Honda was bought in 2024.

The existing Tesla fact is also consistent with the statement.

Therefore:

    do not mutate Honda;
    do not treat Honda=2021 as incorrect;
    do not propose CORRECT merely because the extracted candidate conflicts with Honda's stored value;
    if the intended entity is uncertain, return no mutation.

A likely conversational interpretation is not enough evidence for persistent memory.
"""
