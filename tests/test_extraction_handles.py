"""Saved synthetic model output exercises cross-domain object-handle validation."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tomte.engine.core.extraction import ExtractionMemorySchema, resolve_extraction


def saved_output():
    data = json.loads((Path(__file__).parent / 'fixtures/extraction_duplicate_handles.json').read_text(encoding='utf-8'))
    # Adapt historical wire spelling only; retain saved evidence unchanged.
    for entities in data.values():
        for e in entities:
            e['entity_id'] = 'tmp_' + e['entity_id']
            e['mention_context'] = None
            e["facts"] = [{"attribute_name": f["attribute_name"],
                "value": [{"field_name": "detail", "text": f["value"], "fact_confidence": "high"}],
                "source_chat_id": f["source_chat_id"]} for f in e["facts"]]
    return data


def test_saved_cross_domain_duplicate_is_rejected_before_alignment():
    data = saved_output()
    original = deepcopy(data)
    response = ExtractionMemorySchema.model_validate(data)
    source = data['identity_and_demographics'][0]['facts'][0]['source_chat_id']
    resolved = resolve_extraction(response, [{'item_id': source, 'role': 'user', 'timestamp': 1735808400.0}])
    entities = [entity for _, group in resolved for entity in group]
    handles = [entity['entity_id'] for group in data.values() for entity in group]
    assert {entity.entity_id for entity in entities} == {key for key in handles if handles.count(key) == 1}
    assert all(handles.count(entity.entity_id) == 1 for entity in entities)
    assert data == original


def test_unique_control_preserves_business_text_and_transient_source_without_edges():
    data = saved_output()
    data['activities_and_skills'][0]['entity_id'] = 'tmp_act_01'
    response = ExtractionMemorySchema.model_validate(data)
    source = data['identity_and_demographics'][0]['facts'][0]['source_chat_id']
    resolved = resolve_extraction(response, [{'item_id': source, 'role': 'user', 'timestamp': 1735808400.0}])
    facts = [f for group in resolved for entity in group[1] for f in entity.facts]
    assert {f.value["detail"]["text"] for f in facts} == {"Lena", "Lena works at Harbor Labs", "industrial designer"}
    assert all(f.source_chat_id == source for f in facts)
    assert all("related_entity" not in json.dumps(f.model_dump()) for f in facts)
