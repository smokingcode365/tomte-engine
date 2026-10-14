"""Offline assertion for the current public two-list context representation.

Historical ablation orderings and their metrics remain separate comparisons;
they are not the current public display order.
"""


def assert_public_context_contract(payload, cognitive):
    relevant = payload["query_relevant_memories"]
    background = payload["background_memories"]
    expected = ""
    if relevant or background:
        expected = ("QUERY-RELEVANT MEMORIES\n" + cognitive.format_entities(relevant)
                    + "\n\nBACKGROUND/CONTINUITY\n" + cognitive.format_entities(background))
    assert payload["context_text"] == expected
