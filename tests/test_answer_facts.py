"""Batched fact retrieval: API integration, isolation and evidence preservation."""
from copy import deepcopy
import json
from unittest.mock import Mock
from uuid import uuid4
import pytest

from test_answer_question import answering, decision, plan, model_inputs
from test_relevant_entities_api import retrieval
from memory_fixtures import simple_fact
from tomte.engine.core.answering import FactBatchJudgment, CategorizedFact, AnswerReaderUnavailable
from tomte.engine.core.config import settings
from tomte.engine.exceptions import TomteSecurityViolationError


def packet(name, text, *, history_index=-1, attribute='breed'):
    return {'entity': {'entity_id':str(uuid4()),'entity_name':name,'entity_type':'dog',
                       'entity_domain':'relationships','aliases':[]},
            'fact': {k:v for k,v in simple_fact(attribute,text).items() if k in ('fact_id','attribute_name','value')},
            'value_kind':'current' if history_index==-1 else 'history','history_index':history_index}


def response(packets, total=None):
    return {'facts':[{'tenant_id':'tenant-a','user_id':'user-a','packet':p} for p in packets],
            'total_facts':len(packets) if total is None else total,'oversized_facts':0}


def judgment(count=4, *, start=1, keep=None, category='related_only', possible=(),
             kind='collection', satisfied=False, satisfied_by=None):
    handles = [f'F{i:03d}' for i in range(start, start + count)]
    keep = set(handles if keep is None else keep)
    groups = {'found': [], 'possible': [], 'related': [], 'unrelated': []}
    for handle in handles:
        selected = 'possible' if handle in possible else 'found' if handle in keep else category
        groups['related' if selected == 'related_only' else selected].append(int(handle[1:]))
    return FactBatchJudgment(kind=kind, enough=[int(h[1:]) for h in (satisfied_by or [])] if satisfied else [], **groups)


def request(text='Dog breeds'):
    return plan(text)


@pytest.fixture
def fact_answering(answering):
    r=answering
    r.db.supports_answer_fact_search=True
    r.db.supports_catalogue_navigation=True
    r.packets=[packet(n,f'The user owns {n}, a {b}.') for n,b in
               [('Panda','Lab mix'),('Pepper','Lab mix'),('Pixie','Chihuahua mix'),('Precious','Chihuahua mix')]]
    owner={'tenant_id':'tenant-a','user_id':'user-a'}
    r.db.get_catalogue_domains=Mock(return_value={'rows':[
        {**owner,'entity_domain':'relationships'}], 'is_partial':False})
    r.db.get_catalogue_entities=Mock(return_value={'rows':[
        {**owner, **p['entity']} for p in r.packets], 'is_partial':False})
    def facts(tenant,user,ids,limit):
        return {'rows':[{**owner,'entity_id':p['entity']['entity_id'],
            'fact_id':p['fact']['fact_id'],'attribute_name':p['fact']['attribute_name'],
            'field_names':list(p['fact']['value'])} for p in r.packets if p['entity']['entity_id'] in ids],
            'is_partial':False}
    r.db.get_catalogue_facts=Mock(side_effect=facts)
    r.db.get_answer_facts=Mock(return_value=response(r.packets))
    r.model.parse_structured.side_effect=[request(),judgment(),decision('Four breeds.')]
    return r


def test_four_short_facts_use_one_batch_and_no_catalogue(fact_answering):
    r=fact_answering
    assert r.client.answer_question('user-a','What are my dogs breeds?')=='Four breeds.'
    body=model_inputs(r)[-1]['memory']
    assert all(p['entity']['entity_name'] in body for p in r.packets)
    assert r.model.parse_structured.call_count==3
    r.db.get_answer_facts.assert_called_once_with('tenant-a', 'user-a', [1., 0.], [], 100,
                                               settings.answer_retrieval.lane_bytes)
    r.db.get_catalogue_entities.assert_not_called()
    r.db.get_catalogue_facts.assert_not_called()
    r.db.get_catalogue_domains.assert_not_called()
    assert r.model.parse_structured.call_args_list[1].kwargs['schema'] is FactBatchJudgment
    assert body.count('The user owns Panda, a Lab mix.')==1
    assert 'Doctor at Harbour Hospital' not in body


def test_answer_keeps_information_but_omits_storage_ids_and_search_counters(fact_answering):
    r = fact_answering
    r.client.answer_question('user-a', 'What are my dogs breeds?')
    body = model_inputs(r)[-1]['memory']
    for p in r.packets:
        assert p['entity']['entity_id'] not in body
        assert p['fact']['fact_id'] not in body
        assert p['entity']['entity_name'] in body
        assert all(v['text'] in body for v in p['fact']['value'].values())
    assert 'I took Scout to the vet this morning.' in body
    assert 'FACT SEARCH COVERAGE' not in body
    assert 'history_index' not in body


def test_retrieve_memories_returns_evidence_without_an_answer_call(fact_answering):
    r = fact_answering
    r.model.parse_structured.side_effect = [judgment()]
    memory = r.client.retrieve_memories('user-a', 'Names and breeds of my dogs')
    assert all(p['entity']['entity_name'] in memory for p in r.packets)
    assert 'I took Scout to the vet this morning.' in memory
    assert r.model.parse_structured.call_count == 1
    assert r.db.get_answer_facts.call_count == 1
    assert model_inputs(r)[0]['information_needed'] == 'Names and breeds of my dogs'
    assert r.context.call_count == 1 and r.lifecycle == ['open', 'closed']
    r.usage.scope.assert_called_once_with('tenant-a', 'user-a', 'foreground')


def test_reader_removes_unrelated_facts_even_from_small_batches(fact_answering):
    r=fact_answering
    large=[packet('Other dog '+str(i),'Unrelated details '+('x'*800)) for i in range(5)]
    r.db.get_answer_facts.return_value=response(r.packets+large)
    r.model.parse_structured.side_effect=[request(),
        judgment(9,keep=['F001','F002','F003','F004']),decision('Four breeds.')]
    r.client.answer_question('user-a','Dog breeds?')
    inputs=model_inputs(r)
    assert all(p['entity']['entity_name'] in inputs[-1]['memory'] for p in r.packets)
    assert 'Other dog' not in inputs[-1]['memory']
    assert r.model.parse_structured.call_args_list[1].kwargs['schema'] is FactBatchJudgment


@pytest.mark.parametrize('reader',[judgment(1,start=999),AnswerReaderUnavailable('offline')])
def test_reader_failure_does_not_return_unclassified_evidence(fact_answering,reader):
    r=fact_answering
    r.model.parse_structured.side_effect=[request(),reader,decision('Four breeds.')]
    r.client.answer_question('user-a','Dog breeds?')
    body=model_inputs(r)[-1]['memory']
    assert all(p['entity']['entity_name'] not in body for p in r.packets)
    assert 'Some candidates could not be classified' in body


def test_catalogue_failure_cannot_block_fact_search(fact_answering):
    r=fact_answering
    r.db.get_catalogue_entities.side_effect=AssertionError('Catalogue must not be called')
    r.db.get_catalogue_facts.side_effect=AssertionError('Catalogue must not be called')
    r.client.answer_question('user-a','Dog breeds?')
    assert r.db.get_answer_facts.call_count==1 and r.db.get_answer_facts.call_args.args[3]==[]


def test_foreign_packet_is_rejected_before_reader_or_answer(fact_answering):
    r=fact_answering
    rows=response(r.packets);rows['facts'][0]['user_id']='other-user'
    r.db.get_answer_facts.return_value=rows
    with pytest.raises(TomteSecurityViolationError):r.client.answer_question('user-a','Dog breeds?')
    assert r.model.parse_structured.call_count==1


def test_history_and_same_attribute_instances_stay_distinct(fact_answering):
    r=fact_answering
    one=packet('Car','Blue now.',attribute='color');two=deepcopy(one)
    two.update(history_index=0,value_kind='history');two['fact']['value']['detail']['text']='Silver before repaint.'
    three=deepcopy(one);three['fact']['fact_id']=str(uuid4())
    r.db.get_answer_facts.return_value=response([one,two,three])
    r.model.parse_structured.side_effect=[request(),judgment(3),decision()]
    r.client.answer_question('user-a','What changed?')
    body=model_inputs(r)[-1]['memory']
    assert body.count('Blue now.')==2 and body.count('Silver before repaint.')==1
    assert body.count('S1 Car (dog)')==1
    assert 'M1 color' in body and 'M2 color' in body
    assert one['entity']['entity_id'] not in body
    assert 'Earlier value:' in body and '_recorded_at' not in body


def test_follow_up_uses_fresh_batch_handles_with_same_pending_snapshot(fact_answering):
    r=fact_answering
    r.model.parse_structured.side_effect=[request('first'),
        judgment(),
        decision(action='retrieve',missing='Names and breeds of the dogs'), plan('Names and breeds of the dogs'),
        judgment(),decision('Known now.')]
    assert r.client.answer_question('user-a','Dog breeds?')=='Known now.'
    assert [c.args[3] for c in r.db.get_answer_facts.call_args_list]==[[],[]]
    r.cache.get_pending_search_snapshot.assert_called_once()
    r.db.get_catalogue_domains.assert_not_called()
    r.db.get_catalogue_entities.assert_not_called()
    inputs=model_inputs(r)
    assert inputs[4]['information_needed']=='Names and breeds of the dogs'
    assert 'question' not in inputs[4] and 'memory' not in inputs[4]
    assert inputs[1]['previously_kept']==inputs[4]['previously_kept']==[]
    assert inputs[1]['candidates'][0]['handle']==inputs[4]['candidates'][0]['handle']=='F001'
    assert all('catalogue' not in p for p in inputs if 'memory' in p)


def test_retrieval_uses_reader_but_never_calls_answer_model(fact_answering):
    r=fact_answering
    r.model.parse_structured.side_effect=[judgment(keep=['F001'])]
    memory=r.client.retrieve_memories('user-a','Dog breed information')
    assert r.packets[0]['fact']['fact_id'] in memory
    assert all(p['fact']['fact_id'] not in memory for p in r.packets[1:])
    assert [c.kwargs['schema'] for c in r.model.parse_structured.call_args_list]==[FactBatchJudgment]


def test_retrieval_does_not_expose_foreign_fact(fact_answering):
    r=fact_answering
    rows=response(r.packets);rows['facts'][0]['tenant_id']='other-tenant'
    r.db.get_answer_facts.return_value=rows
    r.model.parse_structured.side_effect=[judgment()]
    with pytest.raises(TomteSecurityViolationError):
        r.client.retrieve_memories('user-a','Dog details')
    assert r.model.parse_structured.call_count==0


def test_partial_and_oversized_coverage_reaches_answer(fact_answering):
    r=fact_answering
    rows=response(r.packets,total=200);rows['oversized_facts']=3
    r.db.get_answer_facts.return_value=rows
    r.client.answer_question('user-a','Dog breeds?')
    body=model_inputs(r)[-1]['memory']
    assert 'Stored-memory search was partial' in body
    assert 'is_partial' not in body and 'oversized_facts' not in body


def test_possible_only_results_reach_answer_without_becoming_confirmed(fact_answering):
    r = fact_answering
    r.model.parse_structured.side_effect = [request(), judgment(keep=[], possible=['F002']),
                                          decision('Possibly, but not confirmed.')]
    assert r.client.answer_question('user-a', 'Dog question') == 'Possibly, but not confirmed.'
    body = model_inputs(r)[-1]['memory']
    assert 'Possible answer candidate; Current:' in body
    assert r.packets[1]['entity']['entity_name'] in body
    assert r.packets[0]['entity']['entity_name'] not in body
