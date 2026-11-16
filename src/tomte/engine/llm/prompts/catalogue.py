"""Catalogue navigation locates evidence; it does not plan an answer."""

CATALOGUE_SELECTION_PROMPT = """Select offered handles likely to locate evidence for information_needed.
The catalogue contains entities or facts. E/F handles identify rows. Domain headings group entities for context; select entities across any headings. Indentation gives parentage. Backslash escapes protect delimiters. Fact rows list attributes and current/historical field names, not values.
Select every likely match, without a target count or exact-word requirement. Match the requested subject, attribute/action/relationship and qualifiers. Related topics or words alone are insufficient. Include identity, ownership, timing or uncertainty only when needed to interpret the requested information.
Keep credible locations with broad labels and relevant indirect, uncertain, planned or historical evidence. Labels are clues, not proof of values or completed events. Avoid speculative connections and unrelated surrounding context.
Selecting a parent does not require selecting a child or all its facts. Empty handles are valid if no offered row is a likely location. Partial coverage does not prove absence.
Return handles only. Do not rewrite the request, invent identities/dates, answer, generate queries or plan further searches. Broad, unresolved-memory and pending-message searches run independently.
"""
