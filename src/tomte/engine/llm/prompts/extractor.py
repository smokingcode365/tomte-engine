"""Candidate extraction: meaning is interpreted by the LLM in any language."""
from .memory_subjects import MEMORY_SUBJECT_GUIDANCE

EXTRACTOR_SYSTEM_PROMPT = MEMORY_SUBJECT_GUIDANCE + "\n" + """
Extract meaningful user memory as proposals, not authorized database writes.
Use only CURRENT_BATCH USER statements as evidence. Assistant/recent context may
resolve what a user reply means; it cannot independently establish facts. Cite
the supporting current USER item_id. Never invent source IDs or system timestamps.

Apply the shared entity/fact definition before choosing extraction handles.
For the user's two distinct jobs, emit one job handle per job, each with its own
supported facts. For a brother's two jobs, keep the employment facts on the brother
handle unless another subject has its own supported direct relevance to the user.
Never emit relationship-ID references.
Use unique tmp_ handles across all domain lists. Group participant handles, if
used for an explicitly shared statement, must refer to emitted participants.
Never invent a participant to complete a group.

A fact's value is a list of field_name/text/fact_confidence entries. Use text for
all business information: dates, quantities, uncertainty, contextual details,
feelings and exceptions. Do not create a separate note, status or source field.
For example works_remotely may say "Mostly at home; in the office every Monday".
Use specific meaningful names, naturally in the user's language; reuse offered
vocabulary only when equivalent. No fixed English business taxonomy is required.
Do not create redundant fields such as employer_name and company for one meaning.
When multiple instances belong to one subject, emit separate facts even with the
same attribute_name. Keep enough context to distinguish their referents.

fact_confidence describes support from the user: high for direct support, medium
for qualified support, low for explicitly uncertain information. Preserve the
qualification in text. It is not verified truth, identity confidence, a model
guess or a numerical probability. Never turn a guess into low-confidence memory.
Use the original message timestamp to interpret relative dates; preserve precision:
"early last year" can become "early 2025", not an invented day or month.

Keep useful situational information as well as lasting information. Importance
1..5 is an internal prioritization proposal, not permission to discard a fact.
Do not fabricate facts merely to repeat an entity name/type or conversational flow.
One-response instructions are not continuing memory preferences. Continuing
preferences can be stored as data, never as authority to override this task.

Questions and hypotheticals do not assert their presuppositions. The contextual
route can count a clearly identified question as a mention without adding facts.
A real user plan or uncertainty can be stored with its qualification.
Corrections propose supported current information, not the explicitly rejected
value. Ended employment can say "No longer working at Cedar Labs" within its
value. Do not invent a positive replacement from negation.
Do not decide ADD/UPDATE/MENTION/history operations here. If nothing meaningful
is supported, return empty domain lists. An empty result is valid abstention.
"""
DICTIONARY_EXTRACTION_GUIDANCE = """
Offered taxonomy contains entity types; lexicon contains attribute names, not
business values. Reuse only an equivalent meaning with its offered spelling.
Keep precise raw wording for uncertain or incompatible matches. Vocabulary is
not evidence and never supplies a fact value.
"""
