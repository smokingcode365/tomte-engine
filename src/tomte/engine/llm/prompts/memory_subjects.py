"""Shared entity/fact meaning for extraction, identity and mutation."""

MEMORY_SUBJECT_GUIDANCE = """
ENTITIES AND FACTS IN TOMTE
An entity is a distinct subject directly involved in the user's life that is
useful to remember in its own right. People, possessions, jobs, events, activities,
projects, experiences and emotional states are examples, not a fixed taxonomy.
A fact is supported information describing that subject: its properties,
circumstances, relationships, experiences or changes. Value fields hold the details.

Direct connection to the user establishes relevance; it does not make every noun
or detail an entity. Choose the subject being remembered. A user's car is an
entity; its colour describes it. More detail alone never requires another entity.
Keep independently identifiable subjects separate. Two explicitly distinct jobs
or cars belonging to the user should normally be two entities, not a collection
called "My jobs" or "My cars" merely because they share a category.

Information about another person normally belongs to that person's facts. For
"My brother Bob is a doctor at a hospital", Bob is the entity; his job, employer
and workplace details describe Bob, however extensive. A separate job or hospital
entity needs its own supported direct relevance to the user, not just a connection
through Bob. "My friend Bob works at Cedar Labs" needs no separate employer entity.

One entity may have multiple facts with the same attribute_name when they describe
different instances. The user's two distinct jobs normally have separate job
entities; a brother's two jobs can be two employment facts on the brother entity.
Neither shared attribute names nor a shared entity type establish duplicate identity.
Updating a subject's details does not by itself create a new entity. Preserve its
identity when the evidence describes the same subject changing over time.

For example, running can be an activity with frequency/distance facts. Anxiety
about an interview can describe that interview; ongoing loneliness can itself be
the emotional experience being remembered. Choose by the user's meaning and
context in any language, not by a fixed list of words or categories.

The requirement that every fact applies equally concerns distributing an update
to multiple entity targets. It does not require distinct facts about ONE subject
to describe the same instance. Keep each fact's referent and evidence clear.

ENTITY NAMES
Prefer a durable supported name or label. Keep changeable descriptions such as
colour and current status in facts rather than baking them into entity_name.
For example, use "Honda Civic" rather than "blue Honda Civic" when blue only
describes its current colour. Preserve a genuine user-established nickname such
as "Bluey" even if the car is repainted. Names need not be unique: distinguish
subjects using user evidence and their facts/UUIDs, not invented names or merging
subjects merely because their names match. Existing canonical names are retained;
this guidance does not authorize automatic renaming of an existing entity.
"""
