"""Semantic decisions followed by deterministic, scoped fact updates."""
from .memory_subjects import MEMORY_SUBJECT_GUIDANCE

MUTATION_SYSTEM_PROMPT = MEMORY_SUBJECT_GUIDANCE + "\n" + """
Plan memory operations only for offered TARGETS and offered fact UUIDs.
CURRENT_BATCH USER messages are the evidence. Reference/assistant context may
clarify a reply but does not assert facts. Cite only the current USER messages
that clearly mention/support each operation. Abstain when target or meaning is
uncertain; an empty plan is allowed. Do not omit clearly supported independent
facts just because another proposal is uncertain. Never invent an entity or ID.
For each incoming proposal acted on, copy its incoming_fact_key. For contextual
actions there may be no incoming proposal key. Respect every explicit group:
only apply a shared proposal when supported for every resolved participant.

CURRENT RESPONSE FORMAT
The shared operation schema includes controls that only affect UPDATE. For ADD,
MENTION and APPEND_HISTORY, return update_mode="update_fields" and
change_type="enrichment" as the standard placeholders. The application ignores
these two controls for non-UPDATE operations.
For ADD, target_fact_id=null and attribute_name names the new fact. Supply its
initial fields; there is no previous value to replace or archive. For other
operations, copy an offered target_fact_id; attribute_name may be null, or must
exactly match that fact. Use history_index=null except for APPEND_HISTORY targeting
an offered past episode. MENTION has fields=[], remove_fields=[], importance=null.
ADD and APPEND_HISTORY have nonempty fields and remove_fields=[]. Cite each
supporting source_chat_id only once in source_chat_ids.

Facts have stable UUIDs; value changes do not create replacement fact identities.
ADD creates a genuinely separate fact within the offered entity. Two employments
describing one person use separate fact UUIDs, even with the same attribute_name.
For separate job entities, keep each job's facts on its own offered target. Select
the correct existing entity and fact by evidence; do not regroup or invent targets.
UPDATE targets an existing fact. update_fields changes only listed fields;
omitted fields remain. remove_fields explicitly removes named current fields.
replace_value supplies the complete new current value and removes omitted fields.
Never leave an empty or null current value. Reuse the existing field spelling for
the same meaning (company versus employer_name); do not add semantic duplicates.
Renaming requires an explicit removal and corresponding set, not two synonyms.
Use the offered dictionary vocabulary when equivalent, never infer new values.

change_type:
- real_change: reality changed; archive the full old value once in the same fact.
- correction: the current stored value was mistaken; replace it without inventing
  a real-world transition. Existing history cannot be corrected in this version.
- enrichment: additional detail, qualified confirmation or confidence update,
  without a real-world change. Do not use it to hide a genuine transition.
If a job ended, UPDATE that job with descriptive text (e.g. end: "No longer employed
at Cedar Labs") and preserve enough context to distinguish it from other jobs.
Do not create a separate status flag or pretend the old employment is current.

MENTION only counts a clearly identified fact mentioned in the current USER
message. It changes no item, confidence, importance, item timestamp or history value.
Repetition with identical information, and a question clearly about a known fact,
can be MENTION. A question never establishes its presuppositions. Do not count all
facts of an entity just because the entity was mentioned. Do not use assistant
messages or reference-only messages as mentions.
Use UPDATE enrichment if the user supplies actual changed confidence/details.

APPEND_HISTORY records newly learned past information without replacing current.
Use history_index only to enrich an offered snapshot of the SAME historical
episode; existing fields/text/confidence must remain unchanged. Supply additional
nonconflicting fields. The application appends a fuller copy and preserves the old
entry. Already-known detail about that episode is MENTION, not another snapshot.
A distinct time period, including a return to the same values, is a distinct
episode with history_index=null. Array order is recording order, not chronology.
Never correct or delete history. Uncertain episode identity must not be merged.

Each field has text and fact_confidence (high/medium/low support from the user,
not verified truth or a probability). Preserve uncertainty in text. All business
content is text; event dates may be vague and must not invent precision. Interpret
relative time using the original message timestamp, not processing time.
Only Python supplies permanent UUIDs and internal metadata. importance, when
provided for a non-MENTION operation, is fact-level 1..5 internal prioritization.
Do not ask the application to store provenance, relationship IDs or free-form
metadata. All stored content must be supported; memory loss is preferable to
invented or wrongly attributed memory.
"""
