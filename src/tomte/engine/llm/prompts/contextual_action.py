"""Route remaining current-user references, including question-only mentions."""
CONTEXTUAL_ACTION_PROMPT = """
Identify current USER references to offered existing facts that are not already
covered by ALREADY_RESOLVED_PROPOSALS. Mixed batches can contain a new entity and
a correction, end, confirmation, clarification, historical detail or mention of
an existing fact. Interpret any language; never rely on a fixed phrase list.
Reference/assistant context helps resolve references but is not independent
evidence. Supply a verbatim evidence_quote from a cited CURRENT USER message;
this quote is distinct from a later normalized value.
Return the offered entity UUID and fact UUID only when the identity is supported
by EXPLICIT/ALIAS/RESOLVED_REFERENCE/GROUP_EXPLICIT evidence. Similarity, a lone
candidate and a confidence score alone do not prove identity. Leave target null
when uncertain. Never invent entities, facts, UUIDs or aliases.
Return target and target_fact_id together: both null when unresolved, otherwise
the offered entity target and a fact UUID belonging to it. List each supporting
current USER source ID only once in source_chat_ids.
A question clearly mentioning a particular stored fact can be MENTION without
asserting truth or changing confidence. Do not count every fact of an entity,
ambiguous mentions, generic discussion, or assistant-only references.
This routing step changes no memory; the mutation step decides supported changes.
Historical correction is unsupported: do not silently treat it as new history.
Return actions=[] when no supported remaining action exists.
"""
