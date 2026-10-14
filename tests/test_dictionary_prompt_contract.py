"""Dictionary guidance describes term types accurately; no model/API calls."""
import re
from types import SimpleNamespace

from tomte.engine.llm.prompts.extractor import EXTRACTOR_SYSTEM_PROMPT, DICTIONARY_EXTRACTION_GUIDANCE
from tomte.engine.llm.structured import TomteDeps, add_dictionaries


def test_entity_vocabulary_guides_types_and_attribute_vocabulary_guides_names():
    prompt = DICTIONARY_EXTRACTION_GUIDANCE.lower()
    assert re.search(r"entity-type vocabulary.*entity types", prompt)
    assert re.search(r"attribute vocabulary.*attribute names/keys.*not values", prompt)
    assert "schema taxonomy (keys)" not in prompt
    assert "value lexicon (values)" not in prompt


def test_vocabulary_reuse_is_conservative_and_does_not_supply_fact_values():
    prompt = DICTIONARY_EXTRACTION_GUIDANCE.lower()
    assert ">90% semantic overlap" in prompt
    assert "only when it is a strong semantic match" in prompt
    assert "create a new" in prompt and "precise meaning" in prompt
    assert "values must come from the user/content" in prompt
    assert "vocabulary reuse never supplies a value" in prompt


def test_injected_dictionary_lists_keep_their_real_semantic_roles():
    ctx = SimpleNamespace(deps=TomteDeps(taxonomy=["vehicle"], lexicon=["paint_color"]))
    prompt = add_dictionaries(ctx)
    injected = prompt[len(DICTIONARY_EXTRACTION_GUIDANCE):].lower()
    assert re.search(r"entity-type vocabulary[^\n]*vehicle", injected)
    assert re.search(r"attribute vocabulary[^\n]*names/keys[^\n]*paint_color", injected)
    assert "values" not in injected
    assert "tenant" not in injected  # Current dictionary storage is global, not tenant-scoped.
