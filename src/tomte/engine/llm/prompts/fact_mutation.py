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
"""
