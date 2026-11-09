"""Route remaining current-user references, including question-only mentions."""
CONTEXTUAL_ACTION_PROMPT = """
Identify current USER references to offered existing facts that are not already
covered by ALREADY_RESOLVED_PROPOSALS. Mixed batches can contain a new entity and
a correction, end, confirmation, clarification, historical detail or mention of
an existing fact. Interpret any language; never rely on a fixed phrase list.
Review each offered fact against the current USER evidence, including changes
to its factual basis even when the user does not mention that fact directly.
ALREADY_RESOLVED_PROPOSALS covers the specific claims being handled, not every
consequence for other facts or everything in the same message or entity.
Proposals with stored_unresolved_id are earlier stored reports being bound by a
current clarification. They are not new events asserted at the clarification's
time; their old details are handled by mutation. Do not duplicate those updates.
When a clearly identified situation has changed, route a related fact whose
current wording would otherwise continue to describe that situation as active.
Describe the narrow, supported qualification in reference_hint; the mutation
step will preserve the earlier value as history when a real change is applied.
Do not assume a feeling, preference or intention has ended merely because its
circumstances changed. Prefer CLARIFICATION to qualify the earlier situation
when the user's present feelings are unknown; use END only when the fact itself
is supported as ended. Preserve explicitly continuing concerns and intentions.
Considering, suggesting or conditionally planning a change does not establish
that it happened. A change to another person's circumstances does not change
the user's. Shared words or topics alone do not justify updating another fact.
Candidates in thoughts_and_plans represent ideas, not actual possessions/actions.
first_mentioned_at is their earliest supporting user-message time, not current
confirmation. "I decided against that idea" can end a clearly identified plan.
"I bought the car; I may move next year" can support an actual purchase and a
separate tentative moving plan: interpret each claim independently. An actual
purchase updates a plan's outcome only if evidence identifies that same plan;
it must not convert a thought into the car or imply every desired detail occurred.
Leave facts unchanged when no supported change to their meaning is established;
do not refresh unrelated facts or infer new circumstances to make memories agree.
Reference/assistant context helps resolve references but is not independent
evidence. Supply a verbatim evidence_quote from a cited CURRENT USER message;
this quote is distinct from a later normalized value.
You decide semantic identity from the evidence and context, including paraphrases
and references. Return the offered entity UUID and fact UUID when identified;
the target contains db_id only, without confidence scores or evidence labels.
Similarity or a lone candidate alone does not prove identity. Leave target null
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
