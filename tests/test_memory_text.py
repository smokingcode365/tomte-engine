"""Public memory text preserves data while switching reader representations."""
from copy import deepcopy
import json

import pytest

from memory_fixtures import simple_entity, simple_fact, STAMP
from tomte.engine.core.memory import public_entity
from tomte.engine.utils.cognitive import CognitiveProcessor
from tomte.engine.core.config import settings, TomteSettings
from tomte.engine.core.memory_text import ROWS_GUIDE, UNRESOLVED_HEADING
from test_relevant_entities_api import retrieval
from test_live_raw_retrieval import live_workflow
from test_unresolved_memories import pending_payload
from unittest.mock import Mock


def test_labelled_rows_are_the_default_and_preserve_selected_history():
    fact = simple_fact('employment', 'Doctor at 松林医院\nentity ["quoted"]')
    fact['history_values'] = [
        {'value': simple_fact('employment', text)['value'], '_archived_at': STAMP}
        for text in ('Old workplace', 'Another workplace')
    ]
    entity = simple_entity('Bob', facts=[fact, simple_fact('employment', 'Weekend tutor')])
    entity['_history_selection'] = {fact['fact_id']: [1]}
    entity['private_metadata'] = 'DO NOT EXPOSE'
    before = deepcopy(entity)

    text = CognitiveProcessor().format_entities([entity])

    assert text.startswith('Memory rows use JSON-escaped arrays with these columns:')
    rows = [line.partition(' ') for line in text.splitlines()]
    facts = [json.loads(rest) for label, _, rest in rows if label == 'fact']
    fields = [json.loads(rest) for label, _, rest in rows if label == 'field']
    assert [row[:3] for row in facts] == [
        [fact['fact_id'], 'employment', True],
        [entity['facts'][1]['fact_id'], 'employment', False],
    ]
    assert fields == [
        ['detail', 'Doctor at 松林医院\nentity ["quoted"]', 'high'],
        ['detail', 'Another workplace', 'high'],
        ['detail', 'Weekend tutor', 'high'],
    ]
    assert sum(label == 'entity' for label, _, _ in rows) == 1
    assert sum(label == 'history' for label, _, _ in rows) == 1
    assert 'Old workplace' not in text and 'DO NOT EXPOSE' not in text
    assert '_recorded_at' not in text and '_archived_at' not in text
    assert entity == before


def test_compact_json_option_has_the_same_public_information():
    entity = simple_entity('Bob', facts=[simple_fact('employment', 'Doctor')])
    text = CognitiveProcessor().format_entities([entity], output_format='compact_json')
    assert json.loads(text) == [public_entity(entity)]
    assert text == json.dumps([public_entity(entity)], ensure_ascii=False, separators=(',', ':'))


@pytest.mark.parametrize('output_format', ['labelled_rows', 'compact_json'])
def test_empty_entities_have_empty_context(output_format):
    assert CognitiveProcessor().format_entities([], output_format=output_format) == ''


@pytest.mark.parametrize('output_format', ['labelled_rows', 'compact_json'])
def test_public_read_formats_same_memories_once_without_extra_calls(retrieval, output_format):
    pending = pending_payload()
    retrieval.db.get_unresolved_memories = Mock(return_value=[dict(
        id=pending['entity_id'], payload=pending, tenant_id='tenant-a', user_id='user-a')])
    before = deepcopy((retrieval.row, pending))
    result = retrieval.client.get_relevant_memories('user-a', 'My car', output_format=output_format)
    assert result['entities'] == [public_entity(retrieval.row)]
    assert result['unresolved_memories'] == [public_entity(pending)]
    text = result['context_text']
    assert text.index(retrieval.row['entity_id']) < text.index(UNRESOLVED_HEADING) < text.index(pending['entity_id'])
    assert text.count(ROWS_GUIDE) == int(output_format == 'labelled_rows')
    assert 'reported_at' in text and 'unresolved_detail' in text
    assert '_unresolved' not in text and 'private_metadata' not in text
    if output_format == 'compact_json':
        first, second = text.split('\n\n' + UNRESOLVED_HEADING + '\n')
        assert json.loads(first.split('\n', 1)[1]) == result['entities']
        assert json.loads(second) == result['unresolved_memories']
    assert (retrieval.row, pending) == before
    assert [call[0] for call in retrieval.model.mock_calls] == ['embed_text']
    assert [call[0] for call in retrieval.db.mock_calls] == ['get_raw_semantic_context', 'get_unresolved_memories']


def test_read_override_does_not_change_default_for_other_calls(retrieval, monkeypatch):
    monkeypatch.setattr(settings, 'memory_text_format', 'compact_json')
    default = retrieval.client.get_relevant_memories('user-a', 'My work')
    override = retrieval.client.get_relevant_memories('user-a', 'My work', output_format='labelled_rows')
    assert not default['context_text'].startswith(ROWS_GUIDE)
    assert override['context_text'].startswith(ROWS_GUIDE)
    assert settings.memory_text_format == 'compact_json'
    assert default['entities'] == override['entities']


def test_unknown_format_is_rejected_before_retrieval(retrieval):
    with pytest.raises(ValueError, match='Memory text format'):
        retrieval.client.get_relevant_memories('user-a', 'My work', output_format='unknown')
    retrieval.context.assert_not_called()


@pytest.mark.parametrize('output_format', ['labelled_rows', 'compact_json'])
def test_chat_formats_three_sections_with_one_legend(live_workflow, monkeypatch, output_format):
    monkeypatch.setattr(settings, 'memory_text_format', output_format)
    workflow, db, _ = live_workflow
    pending = pending_payload()
    db.get_unresolved_memories = Mock(return_value=[dict(
        id=pending['entity_id'], payload=pending, tenant_id='t', user_id='u')])
    result = workflow._handle_registered_user('t', 's', 'u', 'My car')
    text = result['context_text']
    assert text.count(ROWS_GUIDE) == int(output_format == 'labelled_rows')
    assert text.index('QUERY-RELEVANT MEMORIES') < text.index('BACKGROUND/CONTINUITY') < text.index(UNRESOLVED_HEADING)
    for key in ('query_relevant_memories', 'background_memories', 'unresolved_memories'):
        for entity in result[key]:
            assert text.count(entity['entity_id']) == 1
    assert '_unresolved' not in text
    assert [call[0] for call in workflow.llm.embed_text.mock_calls] == ['']


def test_thought_date_and_confidence_are_preserved():
    entity = simple_entity('Considering a car', facts=[simple_fact('intention', 'Maybe buy a green car', confidence='low')])
    entity.update(entity_domain='thoughts_and_plans', first_mentioned_at=STAMP)
    text = CognitiveProcessor().format_entities([entity])
    assert '"thoughts_and_plans"' in text
    assert '"first_mentioned_at":"' + STAMP + '"' in text
    assert 'field ["detail","Maybe buy a green car","low"]' in text


def test_empty_read_does_not_return_a_legend(retrieval):
    retrieval.db.get_raw_semantic_context.return_value = []
    assert retrieval.client.get_relevant_memories('user-a', 'my work') == {
        'entities': [], 'unresolved_memories': [], 'context_text': ''}


def test_environment_switch_accepts_only_supported_formats(monkeypatch):
    from pydantic import ValidationError
    monkeypatch.setenv('MEMORY_TEXT_FORMAT', 'compact_json')
    assert TomteSettings(_env_file=None).memory_text_format == 'compact_json'
    monkeypatch.setenv('MEMORY_TEXT_FORMAT', 'invalid')
    with pytest.raises(ValidationError):
        TomteSettings(_env_file=None)
