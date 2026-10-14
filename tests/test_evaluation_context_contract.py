"""No services, credentials or embedding cache are needed."""
import importlib.util
from pathlib import Path

import pytest

from tomte.engine.utils.cognitive import CognitiveProcessor

_spec = importlib.util.spec_from_file_location(
    "evaluation_context_contract",
    Path(__file__).resolve().parents[1] / "scripts" / "evaluation_context_contract.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@pytest.mark.parametrize("relevant,background", [([], []), (["query"], []), ([], ["persona"]), (["query"], ["persona"])])
def test_public_context_uses_two_distinct_sections(relevant, background):
    class Formatter:
        def format_entities(self, entities):
            return ",".join(entities)

    text = ("QUERY-RELEVANT MEMORIES\n" + ",".join(relevant)
            + "\n\nBACKGROUND/CONTINUITY\n" + ",".join(background)) if relevant or background else ""
    payload = dict(query_relevant_memories=relevant, background_memories=background, context_text=text)
    _module.assert_public_context_contract(payload, Formatter())
    payload["context_text"] = "old single-list output"
    with pytest.raises(AssertionError):
        _module.assert_public_context_contract(payload, Formatter())


def test_real_formatter_preserves_entity_blocks_in_separate_sections():
    def entity(eid):
        return dict(entity_id=eid, entity_name=eid, entity_type="vehicle", facts=[], aliases=[])
    cog = CognitiveProcessor()
    query, background = [entity("bicycle")], [entity("profile")]
    text = ("QUERY-RELEVANT MEMORIES\n" + cog.format_entities(query)
            + "\n\nBACKGROUND/CONTINUITY\n" + cog.format_entities(background))
    _module.assert_public_context_contract(dict(query_relevant_memories=query, background_memories=background, context_text=text), cog)
    assert text.index("[[ENTITY: bicycle") < text.index("BACKGROUND/CONTINUITY") < text.index("[[ENTITY: profile")
