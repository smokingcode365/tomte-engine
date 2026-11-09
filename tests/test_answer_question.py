"""Scripted models test orchestration and evidence delivery, not live answer quality."""
from copy import deepcopy
import json
from unittest.mock import Mock
import pytest

from test_relevant_entities_api import retrieval
from test_pending_search import source
from memory_fixtures import STAMP, simple_fact
from tomte.engine import Tomte
from tomte.engine.core.answering import AnswerSearchPlan, AnswerDecision
from tomte.engine.core.capacity import CapacityExceeded
from tomte.engine.core.config import settings
from tomte.engine.core.memory import public_value
from tomte.engine.exceptions import TomteInvalidIdError, TomteInvalidInputError, TomteSecurityViolationError
from tomte.engine.workers.control import WorkStopped


def decision(answer='The available memory does not establish that.', *, action='answer', searches=(), missing=''):
    return AnswerDecision(action=action,answer=answer,searches=list(searches),missing_information=missing)


@pytest.fixture
def answering(retrieval,monkeypatch):
    r=retrieval
    snapshot=source()
    snapshot['sessions'][0]['queued'][0].update(tenant_id='tenant-a',user_id='user-a',
        content='I took Scout to the vet this morning.')
    cache=Mock(spec=['get_pending_search_snapshot'])
    cache.get_pending_search_snapshot.return_value=snapshot
    monkeypatch.setattr(Tomte,'get_cache',lambda:cache)
    r.db.get_unresolved_memories=Mock(return_value=[])
    r.model.get_vector_dimension=lambda:2
    r.model.embed_batch=Mock(return_value=[[1.,0.]])
    r.model.parse_structured.side_effect=[AnswerSearchPlan(searches=['Scout vet visit']),
        decision('You reported taking Scout to the vet this morning.')]
    r.cache=cache
    return r


def model_inputs(r):
    return [json.loads(c.kwargs['user_text']) for c in r.model.parse_structured.call_args_list]


def test_answer_returns_text_after_one_search_and_preserves_source_context(answering):
    r=answering
    answer=r.client.answer_question('user-a','When did I take Scout to the vet?')
    assert answer=='You reported taking Scout to the vet this morning.'
    assert r.model.parse_structured.call_count==2
    assert r.context.call_count==1 and r.lifecycle==['open','closed']
    r.usage.scope.assert_called_once_with('tenant-a','user-a','foreground')
    r.model.embed_text.assert_called_once_with('Scout vet visit')
    r.model.embed_batch.assert_called_once()
    r.cache.get_pending_search_snapshot.assert_called_once_with('tenant-a','user-a')
    sent=model_inputs(r)
    assert 'memory' not in sent[0]
    assert 'Doctor at Harbour Hospital' in sent[1]['memory']
    assert 'I took Scout to the vet this morning.' in sent[1]['memory']
    assert 'PENDING CONVERSATION' in sent[1]['memory']
    assert '_recorded_at' not in sent[1]['memory']
    assert sent[1]['search_rounds_remaining']==2
    r.db.get_raw_semantic_context.assert_called_once_with('tenant-a','user-a',[1.,0.],limit=settings.semantic_retrieval_limit)


def test_third_round_forces_return_and_reuses_one_pending_snapshot(answering):
    r=answering
    rows=[]
    for i in range(3):
        row=deepcopy(r.row)
        row['facts']=[simple_fact('visit',f'Additional event detail {i}')]
        rows.append([row])
    r.db.get_raw_semantic_context.side_effect=rows
    r.model.parse_structured.side_effect=[
        AnswerSearchPlan(searches=['first search']),
        decision('First partial answer.',action='search',searches=['second search'],missing='The date.'),
        decision('Second partial answer.',action='search',searches=['third search'],missing='The event location.'),
        decision('Supported final answer.',action='search',searches=['forbidden fourth'],missing='More context.'),
    ]
    assert r.client.answer_question('user-a','A multi-step question')=='Supported final answer.'
    assert r.model.parse_structured.call_count==4
    assert r.model.embed_text.call_count==r.db.get_raw_semantic_context.call_count==3
    r.model.embed_batch.assert_called_once()
    r.cache.get_pending_search_snapshot.assert_called_once()
    inputs=model_inputs(r)
    assert [p['search_rounds_remaining'] for p in inputs[1:]]==[2,1,0]
    assert inputs[-1]['previous_searches']==['first search','second search','third search']
    assert inputs[-1]['memory'].count('I took Scout to the vet this morning.')==1


@pytest.mark.parametrize('missing,searches',[('', ['different']),('Need a date.', ['Scout vet visit'])])
def test_no_gap_or_only_repeated_search_returns_best_partial_answer(answering,missing,searches):
    r=answering
    r.model.parse_structured.side_effect=[AnswerSearchPlan(searches=['Scout vet visit']),
        decision('The date is not established.',action='search',searches=searches,missing=missing)]
    assert r.client.answer_question('user-a','When?')=='The date is not established.'
    assert r.model.parse_structured.call_count==2 and r.model.embed_text.call_count==1


def test_unchanged_evidence_still_allows_a_different_final_search(answering):
    r=answering
    later=deepcopy(r.row)
    later['facts'].append(simple_fact('visit','Scout visited the vet on 8 November 2026.'))
    r.db.get_raw_semantic_context.side_effect=[[r.row],[r.row],[later]]
    r.model.parse_structured.side_effect=[AnswerSearchPlan(searches=['Scout']),
        decision(action='search',searches=['Scout date'],missing='Date.'),
        decision(action='search',searches=['veterinary appointment'],missing='Date of the appointment.'),
        decision('8 November 2026.')]
    assert r.client.answer_question('user-a','When?')=='8 November 2026.'
    assert r.model.parse_structured.call_count==4 and r.model.embed_text.call_count==3
    assert model_inputs(r)[-2]['search_rounds_remaining']==1
    assert model_inputs(r)[-1]['search_rounds_remaining']==0
    assert 'Scout visited the vet on 8 November 2026.' in model_inputs(r)[-1]['memory']


def test_plan_searches_are_exactly_deduplicated_and_capped(answering):
    r=answering
    r.model.parse_structured.side_effect=[AnswerSearchPlan(searches=[' ', 'one','one',' two ','three','four']),decision()]
    r.client.answer_question('user-a','Question')
    assert [c.args[0] for c in r.model.embed_text.call_args_list]==['one','two','three']
    assert model_inputs(r)[-1]['memory'].count(r.row['entity_id'])==1
    r.model.embed_batch.assert_called_once()


def test_empty_plan_searches_original_question_and_unknown_answer_is_model_owned(answering):
    r=answering
    r.db.get_raw_semantic_context.return_value=[]
    r.cache.get_pending_search_snapshot.return_value.update(sessions=[],observed_messages=0)
    r.model.parse_structured.side_effect=[AnswerSearchPlan(),decision('目前的记忆不足以确定日期。')]
    assert r.client.answer_question('user-a','是哪一天？')=='目前的记忆不足以确定日期。'
    r.model.embed_text.assert_called_once_with('是哪一天？')
    r.model.embed_batch.assert_not_called()


def test_history_selections_accumulate_by_index_without_collapsing_equal_entries(answering,monkeypatch):
    r=answering
    fact=r.row['facts'][0]; fid=fact['fact_id']
    archived=simple_fact('employment','Worked at Cedar Labs')['value']
    fact['history_values']=[{'value':deepcopy(archived),'_archived_at':STAMP} for _ in range(3)]
    size=len(json.dumps({'value':public_value(archived)},ensure_ascii=False).encode('utf-8'))
    monkeypatch.setattr(settings,'memory_history_bytes',size)
    r.db.get_history_candidates.side_effect=[
        [dict(entity_uuid=r.row['entity_id'],fact_id=fid,history_index=i)] for i in (0,1)]
    r.model.parse_structured.side_effect=[AnswerSearchPlan(searches=['earlier role']),
        decision(action='search',searches=['another role'],missing='Another employment record.'),decision('Two offered records.')]
    assert r.client.answer_question('user-a','Which earlier records are available?')=='Two offered records.'
    body=model_inputs(r)[-1]['memory']
    assert body.count('Worked at Cedar Labs')==2
    assert body.count(r.row['entity_id'])==1
    assert 'true' in body  # A third stored history remains outside this selected view.


def test_foreign_entity_is_rejected_before_answer_model(answering):
    r=answering
    r.row['tenant_id']='other-tenant'
    with pytest.raises(TomteSecurityViolationError): r.client.answer_question('user-a','Question')
    assert r.model.parse_structured.call_count==1
    assert r.lifecycle==['open','closed']


def test_model_failure_is_not_disguised_as_an_answer(answering):
    r=answering
    r.model.parse_structured.side_effect=RuntimeError('Synthetic provider failure')
    with pytest.raises(RuntimeError,match='Synthetic provider failure'):
        r.client.answer_question('user-a','Question')
    r.model.embed_text.assert_not_called()
    assert r.lifecycle==['open','closed']


def test_capacity_failure_does_not_silently_remove_evidence(answering,monkeypatch):
    r=answering
    monkeypatch.setattr(settings,'max_model_input_bytes',6000)
    r.row['facts'][0]['value']['detail']['text']='x'*9000
    with pytest.raises(CapacityExceeded): r.client.answer_question('user-a','Question')
    assert r.model.parse_structured.call_count==1


def test_cancellation_after_planning_stops_before_retrieval(answering):
    r=answering
    def stop(**kwargs):
        r.context.call_args.args[0].stopped.set()
        return AnswerSearchPlan(searches=['Scout'])
    r.model.parse_structured.side_effect=stop
    with pytest.raises(WorkStopped): r.client.answer_question('user-a','Question')
    r.model.embed_text.assert_not_called()


@pytest.mark.parametrize('question',['', ' ', None, 'x\x00y'])
def test_invalid_question_never_opens_resources(answering,question):
    with pytest.raises(TomteInvalidInputError): answering.client.answer_question('user-a',question)
    answering.context.assert_not_called()


def test_invalid_user_never_opens_resources(answering):
    with pytest.raises(TomteInvalidIdError): answering.client.answer_question('','Question')
    answering.context.assert_not_called()
