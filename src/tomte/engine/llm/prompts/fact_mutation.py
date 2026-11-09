"""Semantic decisions followed by deterministic, scoped fact updates."""
from .memory_subjects import MEMORY_SUBJECT_GUIDANCE

MUTATION_SYSTEM_PROMPT = MEMORY_SUBJECT_GUIDANCE + "\n" + """
Plan memory operations only for offered TARGETS and offered fact UUIDs.
CURRENT_BATCH USER messages are the evidence, along with explicitly labelled
earlier stored unresolved claims when supplied. Reference/assistant context may
clarify a reply but does not assert facts. Cite only the current USER messages
that clearly mention/support each operation. Abstain when target or meaning is
uncertain; an empty plan is allowed. Do not omit clearly supported independent
facts just because another proposal is uncertain. Never invent an entity or ID.
For each incoming proposal acted on, copy its incoming_fact_key. For contextual
actions, copy the offered contextual_action_key and use incoming_fact_key=null.
Use contextual_action_key=null for incoming proposals. Select only one kind of
reference per operation. A contextual action targets its offered existing fact
UUID and uses only that action's current USER source IDs; it cannot create a fact.
Sharing a source message does not make unrelated proposals/actions dependent.

EARLIER UNRESOLVED CLAIMS
An incoming proposal with stored_unresolved_id contains an earlier user report;
its source_chat_id and clarification_quote identify the CURRENT USER clarification
binding it to this subject, not the original report. Preserve the complete earlier
claim and its qualifications. Use reported_at to interpret unanchored wording in
that earlier report, never the clarification's date. Report time is not event time.
Do not overwrite a newer current state with an older event simply because its
identity was learned later. Choose APPEND_HISTORY for an older episode of an
existing fact, MENTION only if the complete claim is already represented, or ADD
for a separate fact/episode with its historical meaning clear in the text.
UPDATE is appropriate only when the combined evidence actually supports changing
or enriching the current value. Abstain if chronology or attribution is uncertain.
Always copy incoming_fact_key. Acknowledging it means the ENTIRE retained claim
survives the final plan; partial application must not acknowledge it. The application
retires the pending record only after those keyed mutations commit successfully.

A fresh clarification and an earlier retained claim may describe the SAME fact
or episode under different attribute/field names. For one retained claim and
overlapping UNGROUPED current proposals, plan one complete representation, not
one write per input proposal. Use the retained claim's incoming_fact_key on the
combined operation, preserving its original date, details and qualifications;
also cite any current USER sources supplying additional details in that operation.
For a new target, one ADD can contain the complete clarified claim. Do not also
ADD a redundant fact saying only that this is the previously mentioned subject,
or invent a new fact UUID for a second MENTION. For an existing fact, choose the
appropriate UPDATE, APPEND_HISTORY or MENTION using its offered UUID.
Example: 'my car was painted red on 31 October' plus 'that car is my Tesla' needs
one dated Tesla repaint fact, not both paint and paint_colour copies. Different
repaints, periods, subjects or genuinely independent details must stay distinct;
shared words or attribute names alone do not establish equivalence. Do not use
this rule to omit required group acknowledgements or acknowledge another retained
record without its own accepted keyed operation. Uncertain overlap is not a merge.

GROUP APPLICATION
An incoming_fact_key is a semantic acknowledgement: the complete claim is
represented by the resulting current value, historical value, or already-known
information on that participant. Preserve all supported details and uncertainty;
equivalent phrasing is allowed and verbatim value copying is not required.
Use MENTION only when the claim is already represented, never as a substitute
for a needed update. Use APPEND_HISTORY for newly learned past information without
changing the current value. An incoming attribute may differ from the existing
fact's canonical attribute; select the correct offered fact by meaning and UUID.
Respect every explicit group: apply its proposal to every resolved participant,
using each participant's incoming_fact_key and appropriate operation. Each final
plan must preserve the complete shared claim; no later operation may undo it.
If unsupported for any participant, abstain from that shared proposal for all.
Continue supported independent proposals and offered contextual actions. Never
omit a proposal key or borrow a contextual action key to bypass group completeness.

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
- real_change: the described situation changed, including an intention being
  changed or abandoned; archive the full old value once in the same fact.
- correction: the current stored value was mistaken; replace it without inventing
  a real-world transition. Existing history cannot be corrected in this version.
- enrichment: additional detail, qualified confirmation or confidence update,
  without a real-world change. Do not use it to hide a genuine transition.
If a job ended, UPDATE that job with descriptive text (e.g. end: "No longer employed
at Cedar Labs") and preserve enough context to distinguish it from other jobs.
Do not create a separate status flag or pretend the old employment is current.
For thoughts_and_plans, changes develop the same idea's facts; abandonment may
say "As of 6 February 2026, decided against buying a Tesla". Keep earlier intent
in history for a real change. An explicit current reconfirmation may enrich its
time wording; a question or passing mention must not refresh intent or its date.
Never make an imagined subject actual by rewriting this entity's identity.
Completion may update a clearly identified plan's outcome, but supplies no
unsupported details about a separately offered actual subject.

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
not verified truth or a probability). Confidence applies to the complete qualified
claim; high confidence in a reported intention does not mean it will happen.
Preserve uncertainty in text. All business
content is text; event dates may be vague and must not invent precision.
Keep time understandable when the value is retrieved later: anchor new relative
event wording in the text using its supporting message_time_utc (the readable
original timestamp), never processing time or another message's date. For a
message on 2026-09-13, "this week" can become "during the week containing
13 September 2026" and "early last year" becomes "early 2025". Preserve vague
timing, e.g. "recently, as reported in September 2026", and recurring schedules
such as "every Monday". Apply by meaning in the user's language; respect stated
local dates/timezones without assuming that the user is in UTC.
Preserve already anchored event dates when merging, repeating or enriching facts.
A later mention or unrelated update must not redate an earlier event. Do not use
internal recording/mention times or the current message date to guess the original
date of older stored/quoted/reference wording. Leave unsupported timing uncertain.
Never rewrite history to normalize time. message_time_utc is input context only;
do not persist it or add a source-date field to the fact.
Only Python supplies permanent UUIDs and internal metadata. importance, when
provided for a non-MENTION operation, is fact-level 1..5 internal prioritization.
Do not ask the application to store provenance, relationship IDs or free-form
metadata. All stored content must be supported; memory loss is preferable to
invented or wrongly attributed memory.
"""
