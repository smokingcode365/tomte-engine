"""Embedded facts and complete group updates; obsolete graph edges are rejected."""
from copy import deepcopy
import json
from uuid import UUID
from unittest.mock import Mock
import pytest
from memory_fixtures import simple_entity, simple_fact, field_entries, fact_text, value, prepare_write
from tomte.engine.core.extraction import ExtractionMemorySchema, resolve_extraction
from tomte.engine.core.resolution import ResolutionResponse
from tomte.engine.workflows.consolidation import ConsolidationWorkflow
from resolution_helpers import mutation_response
from test_postgres_transactions import local_db, memory_update
from test_consolidation_mechanics import mechanics, queued_message, memory_rows, processed_count

SOURCE=dict(item_id='source',timestamp=1791547200,role='user',content='Bob works for Google. Both my cars are now red.')


def entity(handle,name,text='Works at Google'):
    return dict(entity_id=handle,entity_name=name,entity_type='person',entity_domain='relationships',aliases=[],
        facts=[dict(attribute_name='employment',value=value(text),importance=3,source_chat_id='source',
                    evidence_context='Current user confirms employment.')])


def record(name):
    payload=simple_entity(name)
    return dict(id=payload['entity_id'],**{k:payload[k] for k in ('entity_id','entity_name','entity_type','entity_domain')},payload=payload)


def decision(handle,kind,rows=()):
    return dict(extracted_entity_id=handle,kind=kind,reasoning='synthetic',explicitly_new=kind=='NEW',
        new_entity_evidence=SOURCE['content'] if kind=='NEW' else None,
        targets=[dict(db_id=r['id'],resolution_confidence=1,evidence='EXPLICIT') for r in rows])


def workflow(decisions):
    llm=Mock()
    def parse(system_prompt,user_text,schema):
        if schema is ResolutionResponse:return schema.model_validate(dict(resolutions=decisions))
        return mutation_response(schema,user_text)
    llm.parse_structured.side_effect=parse
    return ConsolidationWorkflow(Mock(),llm)


def resolve(w,entities,rows=()):
    return w._reconcile_with_golden_records('t','u',entities,list(rows),current_batch=[SOURCE])


def plan(w,ops):
    return w._plan_fact_mutations(ops,dict(CURRENT_BATCH=[SOURCE],RECENT_CONTEXT_REFERENCE_ONLY=[]),'batch')


def test_bob_employer_needs_no_organization_endpoint_and_context_is_transient():
    w=workflow([decision('tmp_bob','NEW')])
    result=plan(w,resolve(w,[entity('tmp_bob','Bob')]))
    op,=result;fact,=op['payload']['facts']
    assert str(UUID(op['entity_id']))==op['entity_id']==op['reserved_id']
    assert fact_text(fact)=='Works at Google'
    assert not {'related_entity_ids','related_entity_uuids','source_chat_id','evidence_context'} & fact.keys()
    sent=json.loads(w.llm.parse_structured.call_args.kwargs['user_text'])
    assert sent['TARGETS'][0]['incoming_semantics'][0]['evidence_context']


def test_two_handles_share_one_target_and_deduplicate_identical_proposals():
    bob=record('Bob')
    w=workflow([decision('tmp_bob_1','MATCH_ONE',[bob]),decision('tmp_bob_2','MATCH_ONE',[bob])])
    ops=resolve(w,[entity('tmp_bob_1','Bob'),entity('tmp_bob_2','Bob')],[bob])
    assert len(ops)==1 and len(ops[0]['extracted_facts'])==1
    result=plan(w,ops)
    assert len(result)==1 and len(result[0]['payload']['facts'])==1
    assert result[0]['entity_id']==bob['id']


def test_unresolved_other_entity_does_not_drop_independent_embedded_fact():
    w=workflow([decision('tmp_other','UNRESOLVED'),decision('tmp_bob','NEW')])
    result=plan(w,resolve(w,[entity('tmp_other','David'),entity('tmp_bob','Bob')]))
    assert len(result)==1 and result[0]['entity_name']=='Bob'
    assert fact_text(result[0]['payload']['facts'][0])=='Works at Google'


def group_setup():
    a,b,c=record('Tesla'),record('Honda'),record('BMW')
    a['payload']['facts']=[simple_fact('employment','Previous synthetic value')]
    d=decision('tmp_group','MATCH_MANY',[a,b])
    d.update(applies_equally_to_all=True,group_source_chat_id='source',group_evidence_quote='Both my cars are now red.')
    w=workflow([d]);ops=resolve(w,[entity('tmp_group','Both cars','red')],[a,b,c])
    return w,ops,a,b,c,d


def test_group_uses_each_targets_fact_uuid_and_leaves_third_entity_untouched():
    w,ops,a,b,c,d=group_setup();result=plan(w,ops)
    assert {op['db_id'] for op in result}=={a['id'],b['id']}
    af,=result[0]['payload']['facts'];bf,=result[1]['payload']['facts']
    assert af['fact_id']==a['payload']['facts'][0]['fact_id'] and bf['fact_id']!=af['fact_id']
    assert fact_text(af)==fact_text(bf)=='red'
    assert af['history_values'][0]['value']==a['payload']['facts'][0]['value'] and bf['history_values']==[]
    d['targets'][1]['resolution_confidence']=.5
    assert resolve(workflow([d]),[entity('tmp_group','Both cars','red')],[a,b,c])==[]


@pytest.mark.parametrize('defect',['abstain','changed_value','missing_proposal','mention_only'])
def test_incomplete_group_plan_cannot_partially_write(defect):
    w,ops,a,b,c,d=group_setup()
    def incomplete(system_prompt,user_text,schema):
        response=mutation_response(schema,user_text)
        if defect=='abstain':response.plans[1].operations=[]
        elif defect=='changed_value':response.plans[1].operations[0].fields=field_entries('different')
        elif defect=='missing_proposal':response.plans[1].operations[0].incoming_fact_key=None
        else:
            from tomte.engine.core.assertions import FactMutation
            response.plans[0].operations=[FactMutation(operation='MENTION',target_fact_id=a['payload']['facts'][0]['fact_id'],
                source_chat_ids=['source'],incoming_fact_key=response.plans[0].operations[0].incoming_fact_key)]
        return response
    w.llm.parse_structured.side_effect=incomplete
    assert plan(w,ops)==[]


@pytest.mark.parametrize('retired',['related_entity_ids','related_entity_uuids'])
def test_wire_rejects_retired_edges_and_validates_group_handles(retired):
    wire=entity('tmp_bob','Bob');wire['facts'][0]['value']=field_entries('Works at Google')
    parsed=ExtractionMemorySchema.model_validate({'relationships':[wire]})
    assert resolve_extraction(parsed,[SOURCE]).relationships[0].facts[0].evidence_context
    wire['facts'][0][retired]=['tmp_missing']
    rejected=ExtractionMemorySchema.model_validate({'relationships':[wire]})
    assert rejected.relationships[0].facts == [] and rejected._rejected
    del wire['facts'][0][retired]
    wire['facts'][0]['group_participant_handles']=['tmp_missing']
    rejected=resolve_extraction(ExtractionMemorySchema.model_validate({'relationships':[wire]}),[SOURCE])
    assert rejected.relationships[0].facts == []


def test_abstained_empty_entity_is_not_invented_as_an_endpoint():
    w=workflow([decision('tmp_other','NEW'),decision('tmp_bob','NEW')])
    other=entity('tmp_other','Google');other['facts']=[]
    result=plan(w,resolve(w,[other,entity('tmp_bob','Bob')]))
    assert len(result)==1 and result[0]['entity_name']=='Bob'
    assert plan(workflow([]),[])==[]


def test_postgres_rejects_retired_edges_and_rolls_back_batch_marker(local_db):
    db,t,u=local_db.db,local_db.tenant,local_db.users[0]
    a,b=memory_update(t,u),memory_update(t,u)
    with db.transaction():
        assert db.claim_consolidation_batch(t,u,'independent')
        assert db.commit_graph_updates([a,b])
    assert len(db.get_user_graph_skeleton(t,u))==2
    for field in ('related_entity_ids','related_entity_uuids'):
        invalid=memory_update(t,local_db.users[1]);invalid['payload']['facts'][0][field]=[a['reserved_id']]
        with pytest.raises(ValueError):
            with db.transaction():
                db.claim_consolidation_batch(t,local_db.users[1],'bad')
                db.commit_graph_updates([invalid])
        assert not db.is_consolidation_processed(t,local_db.users[1],'bad')
        assert db.get_user_graph_skeleton(t,local_db.users[1])==[]


def test_legacy_stored_edge_is_rejected_at_read_boundary(local_db):
    db,t,u=local_db.db,local_db.tenant,local_db.users[0]
    op=memory_update(t,u);assert db.commit_graph_updates([op])
    with db.transaction():
        db.conn.execute("UPDATE tomte_memory_engine SET payload=jsonb_set(payload,'{facts,0,related_entity_ids}',%s::jsonb) WHERE tenant_id=%s AND user_id=%s",(json.dumps(['tmp_old']),t,u))
    with pytest.raises(ValueError,match='invalid stored entity'):db.get_user_graph_skeleton(t,u)


def test_embedded_fact_frozen_batch_replay_and_exact_redis_ack(mechanics,monkeypatch):
    case=mechanics;current=queued_message(case)[0];current['vector']=case.llm.vector();calls=[]
    def parse(system_prompt,user_text,schema,**kwargs):
        calls.append(schema.__name__)
        if schema.__name__ in {'TerminologyAlignmentResponse','FieldAlignmentResponse','ContextualActionResponse'}:return schema()
        if schema is ExtractionMemorySchema:
            e=entity('tmp_bob','Bob');e['facts'][0].update(value=field_entries('Works at Google'),source_chat_id=current['item_id'])
            return schema.model_validate({'relationships':[e]})
        if schema is ResolutionResponse:
            d=decision('tmp_bob','NEW');d['new_entity_evidence']=current['content']
            return schema.model_validate({'resolutions':[d]})
        return mutation_response(schema,user_text)
    monkeypatch.setattr(case.llm,'parse_structured',parse);monkeypatch.setattr(case.llm,'is_junk',lambda text:False)
    monkeypatch.setattr(case.llm,'embed_batch',lambda texts:[case.llm.vector() for _ in texts])
    w=ConsolidationWorkflow(case.db,case.llm)
    monkeypatch.setattr(w.dict_service,'apply_feedback',Mock(side_effect=RuntimeError('optional')))
    assert w.run_consolidation_cycle(case.tenant,case.user,[current])
    rows=memory_rows(case);assert len(rows)==1 and processed_count(case)==1
    assert fact_text(rows[0]['payload']['facts'][0])=='Works at Google'
    before=list(calls);assert w.run_consolidation_cycle(case.tenant,case.user,[current])
    assert calls==before and memory_rows(case)==rows
    assert case.cache.clear_session_4_consolidation(case.tenant,case.user,case.session,[current])
    assert not case.cache.pop_session_chat_log_4_consolidation(case.tenant,case.user,case.session)
