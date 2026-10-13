"""Contextual routing through real local services; model semantics are deterministic fixtures."""
from copy import deepcopy
import json
from uuid import uuid4
from types import SimpleNamespace
import pytest
from tomte.engine.core.schema import Entity,Fact
from tomte.engine.core.extraction import ExtractionMemorySchema
from tomte.engine.core.assertions import adapt_assertions,project_facts
from tomte.engine.core.contextual import ContextualActionResponse
from tomte.engine.core.enums import ChatSenderRole
from tomte.engine.workflows.chat_hook import ChatHookWorkflow
from tomte.engine.workflows.consolidation import ConsolidationWorkflow
from test_consolidation_mechanics import mechanics,memory_rows,processed_count
from tomte.engine.utils.keys import generate_common_key

@pytest.fixture
def contextual_case(mechanics,monkeypatch):
    case=mechanics;case.action_type='CORRECTION';case.operation='CORRECT';case.replacement=None
    case.no_action=False;case.unresolved=False;case.invalid_source=False;case.invalid_target=False;case.no_mutation=False;case.calls=[]
    monkeypatch.setattr(case.llm,'is_junk',lambda text:False)
    monkeypatch.setattr(case.llm,'embed_text',lambda text:case.llm.vector())
    monkeypatch.setattr(case.llm,'embed_batch',lambda texts:[case.llm.vector() for _ in texts])
    def model(system_prompt,user_text,schema,**kwargs):
        data=json.loads(user_text);case.calls.append((schema.__name__,data))
        if schema is ExtractionMemorySchema:return schema()
        if schema is ContextualActionResponse:
            if case.no_action:return schema(actions=[])
            target=data['CANDIDATES'][0];fact=target['assertions'][0]
            return schema.model_validate({'actions':[{'action_type':case.action_type,'reference_hint':target['entity_name'],
                'replacement_text':None,'source_chat_ids':['foreign' if case.invalid_source else data['CURRENT_BATCH'][0]['item_id']],
                'interpretation_confidence':0.97,
                'target':None if case.unresolved else {'db_id':str(uuid4()) if case.invalid_target else target['db_id'],'resolution_confidence':0.98,'evidence':'RESOLVED_REFERENCE'},
                'target_fact_id':None if case.unresolved else fact['fact_id']}]})
        assert schema.__name__=='MutationPlanResponse'
        target=data['TARGETS'][0];fact=target['assertions'][0]
        return schema.model_validate({'plans':[{'target_key':target['target_key'],'operations':[] if case.no_mutation else [{
            'operation':case.operation,'target_fact_id':fact['fact_id'],'new_value':case.replacement,
            'source_chat_ids':[data['CURRENT_BATCH'][0]['item_id']],'interpretation_confidence':0.98,'evidence_type':'RESOLVED_REFERENCE'}]}]})
    monkeypatch.setattr(case.llm,'parse_structured',model)
    return case

def seed(case,attribute='ownership_status',value='sold',trip=False,extra=()):
    entity=Entity(entity_id='seed_01',entity_name='Japan trip' if trip else 'Tesla Model Y',
        entity_type='travel_plan' if trip else 'vehicle',entity_domain='events_and_logistics' if trip else 'assets_and_possessions',
        aliases=['Japan'] if trip else ['Tesla'],facts=[Fact(attribute_name=attribute,value=value,confidence=1,
            source_chat_id='synthetic-seed',last_mentioned_at='2026-10-09T00:00:00Z') for attribute,value in [(attribute,value),*extra]]).model_dump(mode='json')
    entity['facts']=adapt_assertions(entity['facts'],case.tenant+case.user)
    assert case.db.commit_graph_updates([{'mode':'INSERT','tenant_id':case.tenant,'user_id':case.user,
        **{k:entity[k] for k in ('entity_id','entity_name','entity_type','entity_domain')},'payload':entity,'embedding':case.llm.vector()}])
    return memory_rows(case)[0]

def queue(case,current,reference):
    case.cache.append_chat_history(case.tenant,case.session,reference,'assistant',case.user)
    ChatHookWorkflow().on_chat(case.tenant,case.session,current,ChatSenderRole.USER,case.user)
    return case.cache.pop_session_chat_log_4_consolidation(case.tenant,case.user,case.session)

@pytest.mark.parametrize('action,operation,attribute,value,current,reference,replacement,state',[
 ('CORRECTION','CORRECT','ownership_status','sold',"No, that's wrong. You misunderstood me.",'I know you sold your Tesla.',None,'CORRECTED_INVALID'),
 ('CORRECTION','CORRECT','ownership_status','sold','No, I still own it.','I know you sold your Tesla.','owned','CORRECTED_INVALID'),
 ('RETRACTION','RETRACT','trip_plan','Japan',"That's not happening anymore.","You're planning a Japan trip.",None,'RETRACTED'),
 ('CONFIRMATION','REINFORCE','paint_color','red',"Yes, that's still true.",'Your Tesla is red.',None,'ACTIVE'),
])
def test_contextual_action_reuses_mutation_and_replay(contextual_case,action,operation,attribute,value,current,reference,replacement,state):
    case=contextual_case;case.action_type=action;case.operation=operation;case.replacement=replacement
    original=seed(case,attribute,value,trip=attribute=='trip_plan')
    frozen=queue(case,current,reference)
    workflow=ConsolidationWorkflow();assert workflow.run_consolidation_cycle(case.tenant,case.user,frozen)
    row,=memory_rows(case);facts=row['payload']['facts'];old=next(f for f in facts if f['fact_id']==original['payload']['facts'][0]['fact_id'])
    assert row['id']==original['id'] and row['payload']['entity_name']==original['payload']['entity_name']
    assert old['value']==value and old['assertion_state']==state
    if replacement:
        new,=project_facts(facts);assert new['value']==replacement and new['corrects_fact_id']==old['fact_id']
    else:assert len(facts)==1
    if operation=='REINFORCE':assert len(old['evidence'])==2 and old['mention_count']==2
    if operation=='CORRECT':assert old not in project_facts(facts,'history')
    before=deepcopy(memory_rows(case));calls=len(case.calls)
    assert workflow.run_consolidation_cycle(case.tenant,case.user,frozen)
    assert memory_rows(case)==before and processed_count(case)==1 and len(case.calls)==calls
    assert case.cache.clear_session_4_consolidation(case.tenant,case.user,case.session,frozen)

@pytest.mark.parametrize('action,current,reference',[
 ('CLARIFICATION','I meant the other car.','Your blue Tesla...'),
 ('CORRECTION',"No, that's wrong.",'Your Tesla is sold and its paint is red.'),
])
def test_unresolved_context_completes_without_mutating(contextual_case,action,current,reference):
    case=contextual_case;case.action_type=action;case.unresolved=True
    seed(case,extra=[('paint_color','red')] if action=='CORRECTION' else []);before=memory_rows(case);frozen=queue(case,current,reference)
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant,case.user,frozen) is True
    assert memory_rows(case)==before and processed_count(case)==1
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant,case.user,case.session)==frozen
    assert not any(name=='MutationPlanResponse' for name,_ in case.calls)
    assert case.cache.clear_session_4_consolidation(case.tenant,case.user,case.session,frozen)

def test_no_contextual_action_can_finish_empty_batch(contextual_case):
    case=contextual_case;case.no_action=True
    seed(case);before=memory_rows(case);frozen=queue(case,'Something non-actionable.','Your Tesla is sold.')
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant,case.user,frozen)
    assert memory_rows(case)==before and processed_count(case)==1
    assert case.cache.clear_session_4_consolidation(case.tenant,case.user,case.session,frozen)

@pytest.mark.parametrize('defect',['invalid_source','invalid_target'])
def test_invalid_contextual_plan_cannot_claim_or_acknowledge(contextual_case,defect):
    case=contextual_case;setattr(case,defect,True);seed(case);before=memory_rows(case)
    frozen=queue(case,"No, that's wrong.",'Your Tesla is sold.')
    with pytest.raises(ValueError):ConsolidationWorkflow().run_consolidation_cycle(case.tenant,case.user,frozen)
    assert memory_rows(case)==before and processed_count(case)==0
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant,case.user,case.session)==frozen

def test_contextual_candidate_and_reference_isolation(contextual_case,monkeypatch):
    case=contextual_case;seed(case);keys=[];foreign=[]
    try:
        for tenant,user in [(case.tenant+'-other',case.user),(case.tenant,case.user+'-other')]:
            data={**case.__dict__,'tenant':tenant,'user':user};seed(SimpleNamespace(**data));foreign.append((tenant,user))
        for tenant,user,session in [(case.tenant+'-other',case.user,case.session),(case.tenant,case.user+'-other',case.session),(case.tenant,case.user,case.session+'-other')]:
            keys.append(generate_common_key(tenant,session,user));case.cache.append_chat_history(tenant,session,'Foreign claim','assistant',user)
        candidate=case.db.get_resolution_candidates;scopes=[]
        def checked(tenant,user,*args,**kwargs):
            scopes.append((tenant,user));return candidate(tenant,user,*args,**kwargs)
        monkeypatch.setattr(case.db,'get_resolution_candidates',checked)
        frozen=queue(case,"No, that's wrong.",'Your Tesla is sold.')
        assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant,case.user,frozen)
        _,data=next((name,data) for name,data in case.calls if name=='ContextualActionResponse')
        assert scopes==[(case.tenant,case.user)]
        assert data['REFERENCE_ONLY_CONTEXT']==[{'role':'assistant','content':'Your Tesla is sold.'}]
        assert len(data['CANDIDATES'])==1
    finally:
        case.cache.client.delete(*keys)
        for tenant,user in foreign:case.db.delete_user_memory(tenant,user)



@pytest.mark.parametrize("current", ["It is not happening anymore.", "我不再继续了"])
def test_contextual_reference_without_name_still_requires_mutation_stage(contextual_case,current):
    case=contextual_case;case.action_type="RETRACTION";case.operation="RETRACT"
    original=seed(case,"trip_plan","Japan",trip=True)
    frozen=queue(case,current,"We discussed that plan.")
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant,case.user,frozen)
    stored,=memory_rows(case)
    assert any(name=="MutationPlanResponse" for name,_ in case.calls)
    assert stored["id"]==original["id"] and stored["payload"]["aliases"]==original["payload"]["aliases"]
    old,=stored["payload"]["facts"]
    assert old["fact_id"]==original["payload"]["facts"][0]["fact_id"] and old["assertion_state"]=="RETRACTED"



@pytest.mark.parametrize("defect", [None, "foreign_source", "assistant_source", "foreign_fact"])
def test_contextual_nonliteral_reference_keeps_structural_boundaries(monkeypatch, defect):
    from unittest.mock import Mock
    from tomte.engine import Tomte
    payload=Entity(entity_id="canonical",entity_name="Marathon training",entity_type="marathon_training",
        entity_domain="activities_and_skills",facts=[Fact(attribute_name="training_status",value="training",
        confidence=1,source_chat_id="old-source",last_mentioned_at="2026-10-09T00:00:00Z")]).model_dump(mode="json")
    key=str(uuid4());payload["facts"]=adapt_assertions(payload["facts"],key)
    row={"id":key,**{k:payload[k] for k in ("entity_id","entity_name","entity_type","entity_domain")},"payload":payload}
    current=[{"item_id":"u1","timestamp":1791547200,"role":"assistant" if defect=="assistant_source" else "user","content":"我不再继续了"}]
    calls=[]
    def model(system_prompt,user_text,schema):
        calls.append(schema.__name__)
        if schema is ContextualActionResponse:
            return schema.model_validate({"actions":[{"action_type":"RETRACTION","reference_hint":"the activity",
                "source_chat_ids":["foreign" if defect=="foreign_source" else "u1"],"interpretation_confidence":1,
                "target":{"db_id":key,"resolution_confidence":.98,"evidence":"RESOLVED_REFERENCE"},
                "target_fact_id":"foreign-fact" if defect=="foreign_fact" else payload["facts"][0]["fact_id"]}]})
        target,=json.loads(user_text)["TARGETS"]
        return schema.model_validate({"plans":[{"target_key":target["target_key"],"operations":[{
            "operation":"RETRACT","target_fact_id":payload["facts"][0]["fact_id"],"source_chat_ids":["u1"],
            "interpretation_confidence":1,"evidence_type":"USER_EXPLICIT"}]}]})
    db=Mock();db.get_resolution_candidates.return_value=[row]
    monkeypatch.setattr(Tomte,"_db",db);monkeypatch.setattr(Tomte,"_llm",SimpleNamespace(parse_structured=model))
    monkeypatch.setattr("tomte.engine.workflows.consolidation.get_dictionary_service",lambda:Mock())
    workflow=ConsolidationWorkflow();data={"CURRENT_BATCH":current,"RECENT_CONTEXT_REFERENCE_ONLY":[]}
    if defect in {"foreign_source","foreign_fact"}:
        with pytest.raises(ValueError):workflow._resolve_contextual_actions("tenant","user",data,[1.0])
        assert calls==["ContextualActionResponse"]
    else:
        prepared=workflow._resolve_contextual_actions("tenant","user",data,[1.0])
        if defect=="assistant_source":assert prepared==[]
        else:
            planned=workflow._plan_fact_mutations(prepared,data,"contextual-unit-batch")
            assert calls==["ContextualActionResponse","MutationPlanResponse"]
            old,=planned[0]["payload"]["facts"]
            assert old["fact_id"]==payload["facts"][0]["fact_id"] and old["assertion_state"]=="RETRACTED"
