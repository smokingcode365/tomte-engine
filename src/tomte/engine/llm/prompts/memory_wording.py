"""Shared meaning-preservation rules for extraction and mutation."""

MEMORY_WORDING_GUIDANCE = """
Keep the original wording wherever possible. Make only the changes needed for
the memory to be understandable without the conversation; do not paraphrase just
to sound smoother or more general. Preserve specific names, participation, who
did what to whom, uncertainty, negation, conditions and event timing.
Keep the claim in text. Put only necessary supported background in the separate
context property beside that text, usually a short phrase or one short sentence.
Together text and context must be understandable on their own. Context is not a
second summary, an inferred explanation, a source log or a place for unrelated
facts. Omit it when text already contains the needed background. Never use it to
assert assistant-only information or invent an unresolved subject or object.
Example: text="We even played some chess afterward just for fun.";
context="We refers to the user and Sam; after their tournament." Only write this
context when those referents and the event are established by the conversation.
Do not weaken explicit participation to "participants played chess" or turn
watching a game into playing it. Keep qualifications with the claim in text.
On mutation, retain unaffected wording. Preserve context unless supported new
information requires changing or removing it. Recheck context when text changes;
null preserves existing context, an empty string explicitly removes it. Never
carry old event context into a different event. Existing history stays immutable.
"""
