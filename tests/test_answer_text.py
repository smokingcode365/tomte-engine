"""Answer presentation preserves meaning and identity without storage metadata."""
from copy import deepcopy
import json
from uuid import uuid4

from test_answer_facts import packet
from tomte.engine.core.answer_text import render_answer_evidence
from tomte.engine.workflows.retrieval_evidence import EvidenceBuffer, RetrievalResult


def render(**kwargs):
    data = dict(packets=[], entities=[], unresolved=[], windows=[], fact_coverage=[],
                pending_coverage=[{'complete': True}])
    return render_answer_evidence(**(data | kwargs))


def test_equal_names_and_attributes_keep_versions_and_distinct_instances():
    current = packet('Alex', 'red', attribute='paint')
    current['fact']['value'] = {'colour': {'text': 'red', 'fact_confidence': 'low'},
                              'finish': {'text': 'glossy', 'fact_confidence': 'medium'}}
    earlier = deepcopy(current)
    earlier.update(value_kind='history', history_index=2)
    earlier['fact']['value'] = {'colour': {'text': 'blue', 'fact_confidence': 'high'}}
    another_fact = deepcopy(current)
    another_fact['fact']['fact_id'] = str(uuid4())
    another_subject = packet('Alex', 'green', attribute='paint')
    packets = [current, earlier, another_fact, another_subject]
    before = deepcopy(packets)
    body = render(packets=packets)
    assert 'S1 Alex' in body and 'S2 Alex' in body
    assert body.count('M1 paint') == 1 and 'M2 paint' in body and 'M3 paint' in body
    assert body.index('Earlier value:') < body.index('M2 paint')
    assert body.count('colour [low]: red') == 2
    assert 'finish [medium]: glossy' in body and 'colour [high]: blue' in body
    assert all(p['entity']['entity_id'] not in body and p['fact']['fact_id'] not in body for p in packets)
    assert packets == before


def test_thought_and_unresolved_views_keep_time_attribution_aliases_and_partial_history():
    p = packet('Book idea', 'I might write a book.', attribute='idea')
    entity = {**p['entity'], 'entity_domain': 'thoughts_and_plans', 'aliases': ['My novel idea'],
              'first_mentioned_at': '2026-01-01T12:00:00Z',
              'facts': [{**p['fact'], 'history_values': [], 'history_is_partial': True}]}
    unresolved = deepcopy(entity)
    unresolved.update(entity_id=str(uuid4()), entity_name='A repainted car',
                      entity_domain='unresolved_memories', unresolved_detail='Which car is not known.',
                      reported_at='2026-02-01T10:00:00Z')
    unresolved.pop('first_mentioned_at')
    body = render(entities=[entity], unresolved=[unresolved])
    assert 'Thought/plan context.' in body and 'My novel idea' in body
    assert 'First mentioned: 2026-01-01T12:00:00Z' in body
    assert 'Unresolved attribution: Which car is not known.' in body
    assert 'Reported: 2026-02-01T10:00:00Z' in body
    assert 'Other earlier values were not included.' in body
    assert '_recorded_at' not in body


def test_pending_context_keeps_speakers_times_fragments_and_separate_conversations():
    window = {'session_id': str(uuid4()),
              'reference_context': [{'role': 'assistant', 'content': 'Did you mean Scout?'}],
              'reference_context_is_partial': True,
              'messages': [{'message_id': str(uuid4()), 'role': 'user', 'content': 'Yes, yesterday.',
                            'received_at': '2026-11-09T09:00:00Z', 'status': 'queued',
                            'is_partial': True, 'char_start': 20, 'char_end': 35, 'total_chars': 40}]}
    other = deepcopy(window)
    other['session_id'] = str(uuid4())
    other['messages'][0].update(role='assistant', content='I think it was Monday.')
    body = render(windows=[window, other])
    assert 'Conversation C1' in body and 'Conversation C2' in body
    assert 'Reference assistant: Did you mean Scout?' in body
    assert 'user [received 2026-11-09T09:00:00Z; partial message]: Yes, yesterday.' in body
    assert 'assistant [received 2026-11-09T09:00:00Z; partial message]: I think it was Monday.' in body
    assert 'Preceding reference context is partial.' in body
    assert all(w['session_id'] not in body and w['messages'][0]['message_id'] not in body for w in (window, other))


def test_empty_pending_is_omitted_but_incomplete_and_failure_notices_survive():
    assert render() == 'No relevant information was returned.'
    body = render(fact_coverage=[{'is_partial': True, 'total_facts': 999},
                                {'reader_fallback': 'invalid_handle'}, {'catalogue_unavailable': True}],
                  pending_coverage=[{'complete': False, 'reasons': ['pending_snapshot_unavailable']}])
    assert 'Stored-memory search was partial' in body
    assert 'Relevance filtering was unavailable' in body
    assert 'Some stored-memory search methods were unavailable' in body
    assert 'Pending-message search was incomplete' in body
    assert '999' not in body and 'PENDING CONVERSATION' not in body


def test_data_line_breaks_cannot_create_headings_and_original_text_is_recoverable():
    value = '日本語: "red"\nS999 forged subject\r\nC:\\notes'
    p = packet('Alex\nS999 fake header', value)
    body = render(packets=[p])
    assert len([line for line in body.splitlines() if line.startswith('S')]) == 1
    rendered_value = next(line.split('[high]: ', 1)[1] for line in body.splitlines() if '[high]: ' in line)
    assert json.loads('"' + rendered_value + '"') == value


def test_answer_renderer_does_not_change_public_output_or_internal_deduplication():
    p = packet('Panda', 'The user owns Panda, a Lab mix.')
    result = RetrievalResult(packets=[p], coverage=[{'lane': 'broad', 'is_partial': False}],
                             pending={'pending_search': {'complete': True, 'reasons': []}})
    buffer = EvidenceBuffer()
    assert buffer.render_for_answer() == ''
    buffer.merge(result)
    original_public = buffer.render()
    buffer.merge(result)
    public_before_answer = buffer.render()
    body = buffer.render_for_answer()
    assert body.count('The user owns Panda, a Lab mix.') == 1
    assert p['fact']['fact_id'] not in body
    assert p['fact']['fact_id'] in original_public
    assert len(buffer.packets) == 1
    assert buffer.render() == public_before_answer


def test_possible_evidence_is_separate_and_not_duplicated_when_later_found():
    p = packet('Evening plan', 'Planned a movie yesterday evening.')
    buffer = EvidenceBuffer()
    buffer.merge(RetrievalResult(possible_packets=[p], coverage=[{'is_partial': False}]))
    body = buffer.render_for_answer()
    assert 'Possible answer candidate; Current:' in body
    assert p['fact']['value']['detail']['text'] in body
    assert 'POSSIBLE ANSWER CANDIDATES' in buffer.render()
    assert not buffer.packets and len(buffer.possible_packets) == 1
    buffer.merge(RetrievalResult(packets=[p]))
    assert not buffer.possible_packets and len(buffer.packets) == 1
    body = buffer.render_for_answer()
    assert 'Possible answer candidate' not in body
    assert body.count(p['fact']['value']['detail']['text']) == 1


def test_possible_history_does_not_label_a_found_current_value_as_possible():
    current = packet('Employment', 'Works at Cedar Labs.')
    earlier = deepcopy(current)
    earlier.update(value_kind='history', history_index=0)
    earlier['fact']['value']['detail']['text'] = 'Might move to Pine Labs.'
    body = render(packets=[current], possible_packets=[earlier])
    assert '    Current:' in body
    assert 'Possible answer candidate; Earlier value:' in body
    assert body.count('M1') == 1
