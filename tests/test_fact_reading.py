"""Deterministic tests of paging and stopping, not model relevance accuracy."""
import json
from unittest.mock import Mock

import pytest
from filter_fixtures import model_input

from test_answer_facts import packet, judgment, fact_answering, response
from test_answer_question import answering, decision, plan, model_inputs
from test_relevant_entities_api import retrieval
from tomte.engine.core.answering import AnswerReaderUnavailable
from tomte.engine.core.capacity import CapacityExceeded
from tomte.engine.core.config import settings
from tomte.engine.workers.control import WorkControl, WorkStopped
from tomte.engine.workflows.fact_reading import BatchedFactReader


def read(choices, count=100):
    llm = Mock()
    llm.parse_structured.side_effect = choices
    packets = [packet(f'Dog {i}', f'Saved detail {i}') for i in range(count)]
    selected, possible, coverage = BatchedFactReader(llm, WorkControl()).read('Information needed', packets)
    return llm, packets, selected, coverage


def test_one_invalid_category_does_not_admit_or_discard_other_facts():
    first = {
        'kind': 'collection', 'enough': [], 'found': [2], 'possible': [], 'related': [],
        'unrelated': [i for i in range(1, 20) if i != 2] + ['20'],
    }
    llm, packets, selected, coverage = read([first], count=20)
    assert selected == [packets[1]]
    assert coverage[0]['invalid_handles'] == ['F020']


def test_relevant_topic_without_answer_continues_to_later_useful_fact():
    llm, packets, selected, coverage = read([
        judgment(20, keep=[], category='related_only'),
        judgment(20, start=21, keep=['F035'], category='unrelated'),
        judgment(20, start=41, keep=[], category='unrelated'),
    ])
    assert selected == [packets[34]]
    assert llm.parse_structured.call_count == 3
    assert coverage[-1]['stop_reason'] == 'unrelated_tail'
    assert coverage[-1]['examined_facts'] == 60
    assert coverage[-1]['is_partial'] is True


@pytest.mark.parametrize('last_category, expected_calls', [('unrelated', 1), ('related_only', 2), ('possible', 2)])
def test_only_an_unrelated_tail_can_stop(last_category, expected_calls):
    first = judgment(20, keep=[], category='unrelated')
    first.unrelated.remove(20)
    getattr(first, 'related' if last_category == 'related_only' else last_category).append(20)
    llm, _, _, _ = read([first, judgment(20, start=21, keep=[], category='unrelated')])
    assert llm.parse_structured.call_count == expected_calls


def test_related_only_breaks_a_consecutive_unrelated_run():
    first = judgment(20, keep=[], category='unrelated')
    first.unrelated = list(range(1, 6)) + list(range(16, 21))
    first.related = list(range(6, 16))
    llm, _, _, _ = read([first, judgment(20, start=21, keep=[], category='unrelated')])
    assert llm.parse_structured.call_count == 2


@pytest.mark.parametrize('category', ['found', 'possible'])
def test_useful_early_fact_earns_another_batch_despite_unrelated_tail(category):
    first = judgment(20, keep=[], category='unrelated')
    first.unrelated.remove(1)
    getattr(first, category).append(1)
    llm, _, _, _ = read([first, judgment(20, start=21, keep=[], category='unrelated')])
    assert llm.parse_structured.call_count == 2


def test_successful_bounded_request_stops_with_cited_selected_evidence():
    llm, packets, selected, coverage = read([
        judgment(20, keep=['F003'], kind='bounded', satisfied=True, satisfied_by=['F003'])])
    assert selected == [packets[2]]
    assert coverage[-1]['stop_reason'] == 'bounded_need_satisfied'
    assert llm.parse_structured.call_count == 1


@pytest.mark.parametrize('initial_kind', ['collection', 'ambiguous'])
def test_satisfaction_cannot_override_collection_or_earlier_ambiguity(initial_kind):
    llm, _, _, coverage = read([
        judgment(20, kind=initial_kind, satisfied=True, satisfied_by=['F001']),
        judgment(20, start=21, kind='bounded', satisfied=True, satisfied_by=['F001']),
    ], count=40)
    assert llm.parse_structured.call_count == 2
    assert coverage[-1]['stop_reason'] == 'candidate_list_exhausted'
    assert coverage[0]['invalid_satisfaction'] is True


@pytest.mark.parametrize('citations', [[], ['F999'], ['F002']])
def test_satisfaction_requires_offered_selected_evidence(citations):
    llm, _, _, coverage = read([
        judgment(20, keep=['F001'], kind='bounded', satisfied=True, satisfied_by=citations),
        judgment(20, start=21),
    ], count=40)
    assert llm.parse_structured.call_count == 2
    assert coverage[-1]['stop_reason'] == 'candidate_list_exhausted'


def test_previous_selected_evidence_is_available_for_bounded_satisfaction():
    llm, packets, selected, coverage = read([
        judgment(20, keep=['F001'], kind='bounded'),
        judgment(20, start=21, keep=['F021'], kind='bounded', satisfied=True,
                 satisfied_by=['F001', 'F021']),
    ])
    second = model_input(llm.parse_structured.call_args_list[1].kwargs['user_text'])
    assert [p['handle'] for p in second['previously_kept']] == ['F001']
    assert selected == [packets[0], packets[20]]
    assert coverage[-1]['stop_reason'] == 'bounded_need_satisfied'


def test_reader_never_reads_past_100_and_keeps_duplicate_information_relevant():
    llm, _, selected, coverage = read([judgment(20, start=i) for i in range(1, 101, 20)], count=120)
    assert len(selected) == 100
    assert llm.parse_structured.call_count == 5
    sent = [model_input(c.kwargs['user_text']) for c in llm.parse_structured.call_args_list]
    assert [len(p['candidates']) for p in sent] == [20] * 5
    assert sent[-1]['candidates'][-1]['handle'] == 'F100'
    assert coverage[-1]['stop_reason'] == 'candidate_cap_100'


def test_final_short_batch_is_read_and_empty_search_makes_no_model_call():
    llm, _, selected, coverage = read([judgment(20), judgment(3, start=21)], count=23)
    assert len(selected) == 23 and llm.parse_structured.call_count == 2
    assert coverage[-1]['examined_facts'] == 23 and not coverage[-1]['is_partial']
    llm, _, selected, coverage = read([], count=0)
    llm.parse_structured.assert_not_called()
    assert selected == [] and coverage[-1]['examined_facts'] == 0


def test_response_order_does_not_change_vector_order_or_tail():
    first = judgment(20, keep=['F020'], category='unrelated')
    first.unrelated.reverse()
    llm, packets, selected, _ = read([first, judgment(20, start=21, keep=[], category='unrelated')])
    assert selected == [packets[19]] and llm.parse_structured.call_count == 2


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'unknown', 'invalid_category', 'malformed', 'offline'])
def test_invalid_entry_or_call_does_not_admit_unclassified_facts(failure):
    first = judgment(20, keep=['F002'], category='unrelated').model_dump()
    if failure == 'missing': first['unrelated'].pop()
    elif failure == 'duplicate': first['unrelated'][-1] = first['unrelated'][0]
    elif failure == 'unknown': first['unrelated'][-1] = 999
    elif failure == 'invalid_category': first['unrelated'][-1] = '20'
    elif failure == 'malformed': first = {'found': []}
    else: first = AnswerReaderUnavailable('offline')
    llm, packets, selected, coverage = read([first, judgment(20, start=21, keep=['F025'])], count=40)
    expected = [] if failure in ('malformed', 'offline') else [packets[1]]
    assert selected == expected + [packets[24]]
    assert llm.parse_structured.call_count == 2
    assert coverage[0]['is_partial']


def test_found_and_possible_are_returned_separately_and_possible_cannot_satisfy():
    llm = Mock()
    llm.parse_structured.side_effect = [
        judgment(20, keep=['F001'], possible=['F002'], kind='bounded',
                 satisfied=True, satisfied_by=['F002']),
        judgment(20, start=21, keep=[], category='unrelated', kind='bounded'),
    ]
    packets = [packet(str(i), str(i)) for i in range(40)]
    found, possible, coverage = BatchedFactReader(llm, WorkControl()).read('Details', packets)
    assert found == [packets[0]] and possible == [packets[1]]
    assert coverage[0]['invalid_satisfaction']
    assert llm.parse_structured.call_count == 2
    second = model_input(llm.parse_structured.call_args_list[1].kwargs['user_text'])
    assert [p['category'] for p in second['previously_kept']] == ['found', 'possible']


def test_production_schema_has_no_reasons_explanations_or_scores():
    from tomte.engine.core.answering import FactBatchJudgment
    schema = json.dumps(FactBatchJudgment.model_json_schema())
    assert all(word not in schema for word in ('reason', 'explanation', 'relevance_score', 'contains_requested_information'))


@pytest.mark.parametrize('failure', [WorkStopped('cancelled'), CapacityExceeded('capacity')])
def test_cancellation_and_capacity_are_not_mistaken_for_irrelevant_facts(failure):
    with pytest.raises(type(failure)):
        read([failure])


def test_partial_previous_context_cannot_claim_bounded_satisfaction(monkeypatch):
    monkeypatch.setattr(settings.answer_retrieval, 'memory_bytes', 1)
    llm, _, _, coverage = read([
        judgment(20, kind='bounded'),
        judgment(20, start=21, kind='bounded', satisfied=True, satisfied_by=['F021']),
        judgment(1, start=41, kind='bounded'),
    ], count=41)
    second = model_input(llm.parse_structured.call_args_list[1].kwargs['user_text'])
    assert second['previously_kept_is_partial'] is True
    assert coverage[-1]['stop_reason'] == 'candidate_list_exhausted'
    assert llm.parse_structured.call_count == 3


def test_answer_receives_second_batch_evidence_without_catalogue(fact_answering):
    r = fact_answering
    packets = [packet(f'Dog {i}', f'Evidence {i}') for i in range(40)]
    r.db.get_answer_facts.return_value = response(packets, total=200)
    r.model.parse_structured.side_effect = [plan('Dog information'),
        judgment(20, keep=[]), judgment(20, start=21, keep=['F035'], category='unrelated'),
        decision('Supported answer.')]
    assert r.client.answer_question('user-a', 'Dog question') == 'Supported answer.'
    memory = model_inputs(r)[-1]['memory']
    assert 'Evidence 34' in memory and 'Evidence 0' not in memory
    assert 'Stored-memory search was partial' in memory
    r.db.get_catalogue_entities.assert_not_called()
    assert r.db.get_answer_facts.call_count == 1


def test_legacy_larger_limit_cannot_expand_request_past_100(fact_answering, monkeypatch):
    r = fact_answering
    monkeypatch.setattr(settings.answer_retrieval, 'fact_limit', 200)
    r.model.parse_structured.side_effect = [judgment()]
    r.client.retrieve_memories('user-a', 'Dog information')
    assert r.db.get_answer_facts.call_args.args[4] == 100
