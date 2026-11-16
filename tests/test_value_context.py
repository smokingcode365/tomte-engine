"""Short interpretation context travels with its value and immutable history."""
from copy import deepcopy
import json
import pytest

from memory_fixtures import simple_entity
from tomte.engine.core.memory import public_fact, public_entity, Fact
from tomte.engine.core.memory_text import render_memory_context
from tomte.engine.core.answer_text import render_answer_evidence
from tomte.engine.core.answer_facts import checked_packets
from tomte.engine.core.extraction import ExtractionMemorySchema, resolve_extraction
from tomte.engine.llm.embedder import changed_search_entries, changed_retrieval_entries, get_resolution_query
from tomte.engine.database.postgre.lexical_ops import value_entries
from test_memory_values import apply, update, fields, LATER


def test_context_survives_creation_mutation_and_history():
    value = dict(fields(detail='We even played some chess afterward just for fun.')[0],
                 context='We refers to the user and Sam; after their tournament.')
    old = apply([], {'operation': 'ADD', 'attribute_name': 'chess', 'fields': [value]})[0]
    before = deepcopy(old)
    assert public_fact(old)['value']['detail']['context'] == value['context']
    changed = apply([old], update(old, fields=[{**value, 'text': 'Now we play at the club.',
        'context': 'The user and Sam play together.'}], change_type='real_change'), now=LATER)[0]
    assert changed['history_values'][0]['value'] == old['value']
    assert public_fact(changed)['history_values'][0]['value']['detail']['context'] == value['context']
    assert old == before


def contextual_fact():
    return apply([], {'operation': 'ADD', 'attribute_name': 'game', 'fields': [
        dict(fields(detail='We played afterward.')[0], context='The user and Sam played chess.') ]})[0]


def test_context_patch_omission_preserves_and_explicit_empty_clears():
    old = contextual_fact()
    same = apply([old], update(old, fields=fields(detail='We played afterward.')), now=LATER)[0]
    assert same['value'] == old['value']
    cleared = apply([old], update(old, fields=[dict(fields(detail='The user and Sam played chess.')[0], context='')]))[0]
    assert 'context' not in cleared['value']['detail']
    assert 'context' not in public_fact(cleared)['value']['detail']
    assert Fact.model_validate(cleared).model_dump(mode='json') == cleared


def test_history_context_cannot_be_rewritten_or_cleared():
    old = contextual_fact()
    current = apply([old], update(old, fields=fields(detail='We play weekly now.'), change_type='real_change'))[0]
    op = {'operation': 'APPEND_HISTORY', 'target_fact_id': old['fact_id'], 'history_index': 0,
          'fields': fields(detail='We played afterward.')}
    same = apply([current], op)[0]
    assert same['history_values'] == current['history_values']
    for context in ('', 'Someone else played chess.'):
        with pytest.raises(ValueError, match='cannot correct'):
            apply([current], {**op, 'fields': [{**op['fields'][0], 'context': context}]})


def test_context_reaches_readers_and_search_and_context_edit_refreshes_vectors():
    old = simple_entity('Chess game', facts=[contextual_fact()])
    new = deepcopy(old)
    new['facts'][0]['value']['detail']['context'] = 'The user and Sam; after the local tournament.'
    assert len(changed_retrieval_entries(new, old)) == 1
    assert len(changed_search_entries(new, old)) == 1
    search_text = changed_retrieval_entries(new, old)[0]['text']
    assert search_text == 'We played afterward.\nThe user and Sam; after the local tournament.'
    assert value_entries(new)[0]['text'] == search_text
    assert search_text in get_resolution_query(new)
    public = public_entity(new)
    for mode in ('labelled_rows', 'compact_json'):
        rendered = render_memory_context([('MEMORIES', [public])], mode)
        assert new['facts'][0]['value']['detail']['context'] in rendered
        assert '_recorded_at' not in rendered
    packet = {'entity': {k: v for k, v in public.items() if k != 'facts'},
        'fact': public['facts'][0], 'history_index': -1, 'value_kind': 'current'}
    checked = checked_packets({'facts': [{'tenant_id': 't', 'user_id': 'u', 'packet': packet}]}, 't', 'u')
    text = render_answer_evidence(packets=checked, entities=[], unresolved=[], windows=[],
                                 fact_coverage=[], pending_coverage=[])
    assert 'Context: The user and Sam; after the local tournament.' in text
    assert '_recorded_at' not in text


def test_extraction_keeps_context_through_wire_resolution():
    from test_memory_values import CHAT
    fact = {'attribute_name': 'game', 'source_chat_id': 'm1', 'value': [
        dict(fields(detail='We played afterward.')[0], context='The user and Sam played chess.')]}
    extracted = ExtractionMemorySchema.model_validate({'activities_and_skills': [{
        'entity_id': 'tmp_game', 'entity_name': 'Chess game', 'entity_type': 'game',
        'entity_domain': 'activities_and_skills', 'facts': [fact]}]})
    resolved = resolve_extraction(extracted, CHAT).model_dump(mode='json')
    assert resolved['activities_and_skills'][0]['facts'][0]['value']['detail']['context'] == fact['value'][0]['context']
