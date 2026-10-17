"""Connected identities: deterministic contracts, never model-quality claims."""
from copy import deepcopy
import json
from uuid import UUID, uuid4
from unittest.mock import Mock
import pytest
from tomte.engine.core.extraction import ExtractionMemorySchema, resolve_extraction
from tomte.engine.core.resolution import ResolutionResponse
from tomte.engine.workflows.consolidation import ConsolidationWorkflow
from resolution_helpers import mutation_response

SOURCE = dict(item_id="source", timestamp=1791547200, role="user", content="Bob works for Google. Both my cars are now red.")

def entity(handle, name, refs=()):
    return dict(entity_id=handle, entity_name=name, entity_type="person", entity_domain="relationships", aliases=[],
        facts=[dict(attribute_name="employment", value=name+" employed", confidence=1, source_chat_id="source",
                    last_mentioned_at="2026-10-09T12:00:00Z", related_entity_ids=list(refs),
                    evidence_context="Bob is the person discussed; current user confirms employment.")])

def record(name):
    e=entity("legacy_"+name,name); e['facts']=[]
    return dict(id=str(uuid4()), **{k:v for k,v in e.items() if k!='facts'}, payload=e)

def decision(handle, kind, rows=()):
    return dict(extracted_entity_id=handle,kind=kind,reasoning="synthetic",explicitly_new=kind=="NEW",
        new_entity_evidence=SOURCE['content'] if kind=="NEW" else None,
        targets=[dict(db_id=r['id'],resolution_confidence=1,evidence="EXPLICIT") for r in rows])

def workflow(decisions):
    llm=Mock()
    def parse(system_prompt,user_text,schema):
        if schema is ResolutionResponse:return schema.model_validate(dict(resolutions=decisions))
        return mutation_response(schema,user_text)
    llm.parse_structured.side_effect=parse
    return ConsolidationWorkflow(Mock(),llm)

def resolve(w, entities, rows=()):
    return w._reconcile_with_golden_records('t','u',entities,list(rows),current_batch=[SOURCE])

def plan(w,ops):
    return w._plan_fact_mutations(ops,dict(CURRENT_BATCH=[SOURCE],RECENT_CONTEXT_REFERENCE_ONLY=[]),'batch')

def test_new_forward_relationship_uses_python_uuid_and_preserves_context():
    w=workflow([decision('tmp_org_1','NEW'),decision('tmp_per_1','NEW')])
    ops=resolve(w,[entity('tmp_org_1','Google',['tmp_per_1']),entity('tmp_per_1','Bob')])
    assert ops[0]['extracted_facts'][0]['related_entity_uuids']==[ops[1]['reserved_id']]
    assert all(UUID(op['reserved_id']) and op['entity_id'].startswith('entity_') for op in ops)
    result=plan(w,ops)
    assert result[0]['payload']['facts'][0]['related_entity_uuids']==[result[1]['reserved_id']]
    assert result[0]['payload']['facts'][0]['related_entity_ids']==[]
    sent=json.loads(w.llm.parse_structured.call_args.kwargs['user_text'])
    assert sent['TARGETS'][0]['incoming_semantics'][0]['evidence_context']
    assert 'evidence_context' not in result[0]['payload']['facts'][0]
    assert len(w.llm.parse_structured.call_args_list)==2

def test_existing_endpoint_and_two_handles_share_one_target():
    bob=record('Bob')
    w=workflow([decision('tmp_per_1','MATCH_ONE',[bob]),decision('tmp_per_2','MATCH_ONE',[bob]),decision('tmp_org_1','NEW')])
    ops=resolve(w,[entity('tmp_per_1','Bob'),entity('tmp_per_2','Bob'),entity('tmp_org_1','Google',['tmp_per_2'])],[bob])
    assert len(ops)==2 and len(ops[0]['extracted_facts'])==1
    assert ops[1]['extracted_facts'][0]['related_entity_uuids']==[bob['id']]
    assert len(plan(w,ops))==2

def test_unresolved_endpoint_drops_only_dependent_fact():
    w=workflow([decision('tmp_per_1','UNRESOLVED'),decision('tmp_org_1','NEW')])
    org=entity('tmp_org_1','Google',['tmp_per_1'])
    independent=deepcopy(org['facts'][0]);independent.update(attribute_name='city',value='London',related_entity_ids=[])
    org['facts'].append(independent)
    ops=resolve(w,[entity('tmp_per_1','David'),org])
    assert [f['value'] for f in ops[0]['extracted_facts']]==['London']
    assert [f['value'] for f in plan(w,ops)[0]['payload']['facts']]==['London']

def test_group_is_two_independent_plans_and_third_entity_untouched():
    a,b,c=record('Tesla'),record('Honda'),record('BMW')
    d=decision('tmp_group_1','MATCH_MANY',[a,b]);d.update(applies_equally_to_all=True,group_source_chat_id='source',group_evidence_quote='Both my cars are now red.')
    w=workflow([d]); ops=resolve(w,[entity('tmp_group_1','Both cars')],[a,b,c]); result=plan(w,ops)
    assert {op['db_id'] for op in result}=={a['id'],b['id']}
    assert len({op['payload']['facts'][0]['fact_id'] for op in result})==2
    assert all(op['payload']['facts'][0]['source_chat_id']=='source' for op in result)
    d['targets'][1]['resolution_confidence']=.5
    assert resolve(workflow([d]),[entity('tmp_group_1','Both cars')],[a,b,c])==[]

def test_wire_tmp_prefix_and_optional_context_survive_source_resolution():
    e=entity('tmp_per_1','Bob');e['facts'][0].pop('last_mentioned_at')
    wire=ExtractionMemorySchema.model_validate(dict(relationships=[e]))
    assert resolve_extraction(wire,[SOURCE]).relationships[0].facts[0].evidence_context
    e['entity_id']='per_1'
    with pytest.raises(ValueError):ExtractionMemorySchema.model_validate(dict(relationships=[e]))


from test_postgres_transactions import local_db, memory_update
from test_consolidation_mechanics import mechanics, queued_message, memory_rows, processed_count

def test_postgres_forward_links_scope_and_atomic_rollback(local_db):
    db,t,u=local_db.db,local_db.tenant,local_db.users[0]
    a,b=memory_update(t,u),memory_update(t,u)
    a['reserved_id'],b['reserved_id']=str(uuid4()),str(uuid4())
    a['payload']['facts']=[dict(value='linked',related_entity_uuids=[b['reserved_id']])]
    with db.transaction():
        assert db.claim_consolidation_batch(t,u,'forward')
        assert db.commit_graph_updates([a,b])
    rows=db.get_user_graph_skeleton(t,u)
    assert {str(r['id']) for r in rows}=={a['reserved_id'],b['reserved_id']}
    assert a['entity_id']!=b['entity_id']  # logical collision never changes the UUID link
    for endpoint in [str(uuid4()), a['reserved_id']]:
        foreign=memory_update(t,local_db.users[1])
        foreign['payload']['facts']=[dict(value='invalid',related_entity_uuids=[endpoint])]
        with pytest.raises(ValueError,match='endpoint'):
            with db.transaction():
                assert db.claim_consolidation_batch(t,local_db.users[1],'bad')
                db.commit_graph_updates([foreign])
        assert not db.is_consolidation_processed(t,local_db.users[1],'bad')
        assert db.get_user_graph_skeleton(t,local_db.users[1])==[]

def test_postgres_legacy_links_readable_but_new_opaque_links_rejected(local_db):
    db,t,u=local_db.db,local_db.tenant,local_db.users[0]
    op=memory_update(t,u);assert db.commit_graph_updates([op])
    # Emulate an existing historical JSONB document, without migrating it.
    with db.transaction():
        db.conn.execute("UPDATE tomte_memory_engine SET payload=jsonb_set(payload,'{facts}',%s::jsonb) WHERE tenant_id=%s AND user_id=%s",
            (json.dumps([dict(attribute_name='friend',value='Bob',source_chat_id='old',related_entity_ids=['per_01'])]),t,u))
    row=db.get_user_graph_skeleton(t,u)[0]
    assert row['payload']['facts'][0]['related_entity_ids']==['per_01']
    update=dict(op,mode='MUTATE',db_id=str(row['id']),payload=deepcopy(row['payload']),_expected_payload=deepcopy(row['payload']))
    assert db.commit_graph_updates([update])
    fresh=memory_update(t,u);fresh['payload']['facts']=[dict(value='Bob',related_entity_ids=['tmp_per_1'])]
    with pytest.raises(ValueError,match='legacy'):
        with db.transaction():db.commit_graph_updates([fresh])

def test_connected_frozen_batch_replay_and_exact_redis_ack(mechanics,monkeypatch):
    case=mechanics
    monkeypatch.setattr('test_consolidation_mechanics.MESSAGE', 'Bob works for Google.')
    current=queued_message(case)[0]
    current['vector']=case.llm.vector()
    calls=[]
    def parse(system_prompt,user_text,schema,**kwargs):
        calls.append(schema.__name__)
        data=json.loads(user_text)
        if schema.__name__=='TerminologyAlignmentResponse':return schema()
        if schema is ExtractionMemorySchema:
            entities=[entity('tmp_org_1','Google',['tmp_per_1']),entity('tmp_per_1','Bob')]
            for e in entities:
                for fact in e['facts']:
                    fact.pop('last_mentioned_at');fact['source_chat_id']=current['item_id']
            return schema.model_validate(dict(relationships=entities))
        if schema is ResolutionResponse:
            ds=[decision('tmp_org_1','NEW'),decision('tmp_per_1','NEW')]
            for d in ds:d['new_entity_evidence']=current['content']
            return schema.model_validate(dict(resolutions=ds))
        return mutation_response(schema,user_text)
    monkeypatch.setattr(case.llm,'parse_structured',parse)
    monkeypatch.setattr(case.llm,'is_junk',lambda text:False)
    monkeypatch.setattr(case.llm,'embed_batch',lambda texts:[case.llm.vector() for _ in texts])
    w=ConsolidationWorkflow(case.db,case.llm)
    monkeypatch.setattr(w.dict_service,'apply_feedback',Mock(side_effect=RuntimeError('optional')))
    assert w.run_consolidation_cycle(case.tenant,case.user,[current])
    rows=memory_rows(case);assert len(rows)==2 and processed_count(case)==1
    byname={r['entity_name']:r for r in case.db.get_user_graph_skeleton(case.tenant,case.user)}
    assert byname['Google']['payload']['facts'][0]['related_entity_uuids']==[str(byname['Bob']['id'])]
    before=list(calls)
    assert w.run_consolidation_cycle(case.tenant,case.user,[current])
    assert calls==before and memory_rows(case)==rows
    assert case.cache.clear_session_4_consolidation(case.tenant,case.user,case.session,[current])
    assert not case.cache.pop_session_chat_log_4_consolidation(case.tenant,case.user,case.session)


def test_linked_new_endpoint_needs_no_redundant_name_fact():
    bob=entity('tmp_per_1','Bob');bob['facts']=[]
    w=workflow([decision('tmp_org_1','NEW'),decision('tmp_per_1','NEW')])
    ops=plan(w,resolve(w,[entity('tmp_org_1','Google',['tmp_per_1']),bob]))
    assert len(ops)==2 and ops[1]['payload']['facts']==[]
    assert ops[0]['payload']['facts'][0]['related_entity_uuids']==[ops[1]['reserved_id']]
    assert plan(workflow([]),[])==[]

def test_group_planner_uses_each_prior_state_and_abstains_without_partial_write():
    from tomte.engine.core.assertions import MutationPlanResponse
    a,b,c=record('Tesla'),record('Honda'),record('BMW')
    old=entity('tmp_x','Both')['facts'][0];old['value']='old insurer';a['payload']['facts']=[old]
    d=decision('tmp_group','MATCH_MANY',[a,b]);d.update(applies_equally_to_all=True,group_source_chat_id='source',group_evidence_quote='Both my cars are now red.')
    w=workflow([d]);ops=resolve(w,[entity('tmp_group','Both')],[a,b,c]);result=plan(w,ops)
    af=result[0]['payload']['facts'];bf=result[1]['payload']['facts']
    assert af[0]['assertion_state']=='HISTORICAL' and af[1]['supersedes_fact_id']==af[0]['fact_id']
    assert len(bf)==1 and bf[0]['supersedes_fact_id'] is None
    def omit(system_prompt,user_text,schema):
        response=mutation_response(schema,user_text)
        response.plans[1].operations=[]
        return response
    w.llm.parse_structured.side_effect=omit
    assert plan(w,ops)==[]

def test_missing_template_cannot_turn_relationship_into_plain_assertion():
    from tomte.engine.core.assertions import MutationPlanResponse
    w=workflow([decision('tmp_org_1','NEW'),decision('tmp_per_1','NEW')])
    ops=resolve(w,[entity('tmp_org_1','Google',['tmp_per_1']),entity('tmp_per_1','Bob')])
    def mismatched(system_prompt,user_text,schema):
        response=mutation_response(schema,user_text)
        response.plans[0].operations[0].new_value='unassociated paraphrase'
        return response
    w.llm.parse_structured.side_effect=mismatched
    result=plan(w,ops)
    assert len(result)==1 and result[0]['entity_name']=='Bob'

def test_new_endpoint_abstention_drops_dependent_update():
    w=workflow([decision('tmp_org_1','NEW'),decision('tmp_per_1','NEW')])
    ops=resolve(w,[entity('tmp_org_1','Google',['tmp_per_1']),entity('tmp_per_1','Bob')])
    def omit(system_prompt,user_text,schema):
        response=mutation_response(schema,user_text);response.plans[1].operations=[];return response
    w.llm.parse_structured.side_effect=omit
    assert plan(w,ops)==[]


def test_host_context_distinguishes_canonical_and_legacy_links():
    from tomte.engine.utils.cognitive import CognitiveProcessor
    from demo.app import active_entity
    row=record('Google');row['payload']['facts']=entity('tmp_org','Google')['facts']
    target=str(uuid4());row['payload']['facts'][0].update(related_entity_uuids=[target],related_entity_ids=['per_old'])
    flat=dict(row['payload'],id=row['id'])
    text=CognitiveProcessor().format_entities([flat])
    assert 'related UUIDs: '+target in text and 'legacy relationship annotations: unvalidated' in text
    assert 'per_old' not in text and 'ROW UUID: '+row['id'] in text
    assert active_entity(row)['facts'][0]['related_entity_uuids']==[target]


def test_postgres_cross_tenant_link_and_reserved_uuid_collision_roll_back(local_db):
    db,t,u=local_db.db,local_db.tenant,local_db.users[0]
    base=memory_update(t,u);base['reserved_id']=str(uuid4());assert db.commit_graph_updates([base])
    foreign=memory_update(t+'-foreign',u)
    foreign['payload']['facts']=[dict(value='bad',related_entity_uuids=[base['reserved_id']])]
    with pytest.raises(ValueError,match='endpoint'):
        with db.transaction():
            db.claim_consolidation_batch(t+'-foreign',u,'foreign')
            db.commit_graph_updates([foreign])
    assert not db.is_consolidation_processed(t+'-foreign',u,'foreign')
    assert db.get_user_graph_skeleton(t+'-foreign',u)==[]
    import psycopg
    collision=memory_update(t,u);collision['reserved_id']=base['reserved_id']
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db.transaction():
            db.claim_consolidation_batch(t,u,'collision')
            db.commit_graph_updates([collision])
    assert not db.is_consolidation_processed(t,u,'collision')
    assert len(db.get_user_graph_skeleton(t,u))==1
