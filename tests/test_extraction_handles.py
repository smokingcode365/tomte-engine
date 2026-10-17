"""Saved synthetic model output exercises cross-domain object-handle validation."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tomte.engine.core.extraction import ExtractionMemorySchema, resolve_extraction


def saved_output():
    data = json.loads((Path(__file__).parent / 'fixtures/extraction_duplicate_handles.json').read_text(encoding='utf-8'))
    # Adapt historical wire spelling only; retain saved evidence unchanged.
    for entities in data.values():
        for e in entities:
            e['entity_id'] = 'tmp_' + e['entity_id']
            e['mention_context'] = None
            for f in e['facts']:
                f['related_entity_ids'] = ['tmp_' + x for x in f['related_entity_ids']]
                f.update(evidence_context=None, group_participant_handles=[])
    return data


def test_saved_cross_domain_duplicate_is_rejected_before_alignment():
    data = saved_output()
    original = deepcopy(data)
    with pytest.raises(ValidationError, match='Temporary extraction IDs must be unique within a batch'):
        ExtractionMemorySchema.model_validate(data)
    assert data == original


def test_unique_control_preserves_facts_references_and_provenance():
    data = saved_output()
    data['activities_and_skills'][0]['entity_id'] = 'tmp_act_01'
    response = ExtractionMemorySchema.model_validate(data)
    source = data['identity_and_demographics'][0]['facts'][0]['source_chat_id']
    resolved = resolve_extraction(response, [{'item_id': source, 'role': 'user', 'timestamp': 1735808400.0}])
    assert response.model_dump(mode='json') == data
    assert resolved.relationships[0].facts[0].related_entity_ids == ['tmp_per_01']
    assert resolved.activities_and_skills[0].facts[0].related_entity_ids == ['tmp_org_01']
    facts = [f for group in resolved for entity in group[1] for f in entity.facts]
    assert {f.value for f in facts} == {'Lena', 'Lena works at Harbor Labs', 'industrial designer'}
    assert all(f.source_chat_id == source and f.last_mentioned_at == '2025-01-02T09:00:00Z' for f in facts)
