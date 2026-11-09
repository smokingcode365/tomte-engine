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

THOUGHTS AND PLANS
Use thoughts_and_plans for a specific thought, intention, tentative plan or
recommendation directly relevant to the user. The entity represents the IDEA,
not an imagined possession, job or completed event. Name/type it accordingly,
e.g. "Car purchase intention" / "purchase_intention", in the user's language.
Keep separate ideas separate; do not create one generic bucket of all thoughts.
An actual car, current job, established activity or confirmed arrangement retains
its usual domain. Temporary feelings and preferences are not automatically plans.
Details about another person's ideas can remain facts of that person, with clear
attribution. A specific recommendation reported by the user can be remembered
without assuming the user adopted it; assistant-only advice is not user evidence.

Each thought value must preserve its qualification even when read separately:
"Considering buying a Tesla" and "If buying a car, would prefer blue", not
"Owns a Tesla" or an unqualified "blue". A question alone does not establish a
plan. Changes, elaboration and abandonment of the SAME idea retain its identity.
Do not convert its UUID/domain into an actual car/job when an action happens.
Explicit completion can separately establish an actual subject and a supported
outcome for the identified plan. Never copy desired details into actual facts.
Thought/non-thought identity matches are forbidden, including mixed MATCH_MANY.

first_mentioned_at, when supplied, is Tomte's earliest supporting USER-message
time for this thought. It is not when the idea first arose, a deadline, or proof
that the thought is still active. Python manages this field from accepted source
IDs; never generate it. Preserve actual timing, conditions and uncertainty in
value text. An old plan's age alone establishes neither completion nor abandonment.

The requirement that every fact applies equally concerns distributing an update
to multiple entity targets. It does not require distinct facts about ONE subject
to describe the same instance. Keep each fact's referent and evidence clear.

UNRESOLVED STATEMENTS
unresolved_memories is application-managed storage for user-supported statements
whose particular subject is not identified. It is not a subject domain to extract.
Leave that extraction collection empty; extract the supported claim under its
natural subject/domain and let identity resolution identify missing attribution.
An uncertain identity does not make the reported information false. Preserve
qualifications without inventing identities, group members or unsupported facts.
When reading unresolved memories, treat unresolved_detail as a warning about
missing attribution; never silently attach these statements to a known entity.

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
