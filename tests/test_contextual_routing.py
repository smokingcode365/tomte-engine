"""Contextual routing through real local services; model semantics are deterministic fixtures."""
from copy import deepcopy
import json
from uuid import uuid4
from types import SimpleNamespace
import pytest
from tomte.engine.core.extraction import ExtractionMemorySchema
from memory_fixtures import simple_entity, simple_fact, prepare_write, field_entries, fact_text
from tomte.engine.core.contextual import ContextualActionResponse
from tomte.engine.core.enums import ChatSenderRole
from tomte.engine.workflows.chat_hook import ChatHookWorkflow
from tomte.engine.workflows.consolidation import ConsolidationWorkflow
from test_consolidation_mechanics import mechanics,memory_rows,processed_count
from tomte.engine.utils.keys import generate_common_key

@pytest.fixture
def contextual_case(mechanics,monkeypatch):
    case=mechanics;case.action_type='CORRECTION';case.operation='UPDATE';case.replacement='Previous claim disputed';case.change_type='correction'
    case.no_action=False;case.unresolved=False;case.invalid_source=False;case.invalid_target=False;case.no_mutation=False;case.calls=[]
    monkeypatch.setattr(case.llm,'is_junk',lambda text:False)
    monkeypatch.setattr(case.llm,'embed_text',lambda text:case.llm.vector())
    monkeypatch.setattr(case.llm,'embed_batch',lambda texts:[case.llm.vector() for _ in texts])
    def model(system_prompt,user_text,schema,**kwargs):
        data=json.loads(user_text);case.calls.append((schema.__name__,data))
        if schema is ExtractionMemorySchema:return schema()
        if schema is ContextualActionResponse:
            if case.no_action:return schema(actions=[])
            target=data['CANDIDATES'][0];fact=target['facts'][0]
            return schema.model_validate({'actions':[{'action_type':case.action_type,'reference_hint':target['entity_name'],
                'evidence_quote':data['CURRENT_BATCH'][0]['content'],'source_chat_ids':['foreign' if case.invalid_source else data['CURRENT_BATCH'][0]['item_id']],
                'target':None if case.unresolved else {'db_id':str(uuid4()) if case.invalid_target else target['db_id'],'resolution_confidence':0.98,'evidence':'RESOLVED_REFERENCE'},
                'target_fact_id':None if case.unresolved else fact['fact_id']}]})
        assert schema.__name__=='MutationPlanResponse'
        target=data['TARGETS'][0];fact=target['facts'][0]
        return schema.model_validate({'plans':[{'target_key':target['target_key'],'operations':[] if case.no_mutation else [{
            'operation':case.operation,'target_fact_id':fact['fact_id'],
            'fields':field_entries(case.replacement) if case.operation=='UPDATE' else [],
            'change_type':case.change_type if case.operation=='UPDATE' else 'enrichment',
            'source_chat_ids':[data['CURRENT_BATCH'][0]['item_id']]}]}]})
    monkeypatch.setattr(case.llm,'parse_structured',model)
    return case

def seed(case,attribute='ownership_status',value='sold',trip=False,extra=()):
    entity=simple_entity('Japan trip' if trip else 'Tesla Model Y',
        kind='travel_plan' if trip else 'vehicle',domain='events_and_logistics' if trip else 'assets_and_possessions',
        aliases=['Japan'] if trip else ['Tesla'],facts=[simple_fact(attr,text) for attr,text in [(attribute,value),*extra]])
    assert case.db.commit_graph_updates([prepare_write(case.tenant,case.user,entity,vector=case.llm.vector())])
    return memory_rows(case)[0]

def queue(case,current,reference):
    case.cache.append_chat_history(case.tenant,case.session,reference,'assistant',case.user)
    ChatHookWorkflow().on_chat(case.tenant,case.session,current,ChatSenderRole.USER,case.user)
    return case.cache.pop_session_chat_log_4_consolidation(case.tenant,case.user,case.session)

@pytest.mark.parametrize('action,operation,attribute,value,current,reference,replacement,change_type',[
 ('CORRECTION','UPDATE','ownership_status','sold',"No, that's wrong. You misunderstood me.",'I know you sold your Tesla.','Previous claim disputed','correction'),
 ('CORRECTION','UPDATE','ownership_status','sold','No, I still own it.','I know you sold your Tesla.','owned','correction'),
 ('END','UPDATE','trip_plan','Japan',"That's not happening anymore.","You're planning a Japan trip.",'No longer planned','real_change'),
 ('CONFIRMATION','MENTION','paint_color','red',"Yes, that's still true.",'Your Tesla is red.',None,'enrichment'),
])
def test_contextual_action_reuses_mutation_and_replay(contextual_case,action,operation,attribute,value,current,reference,replacement,change_type):
    case=contextual_case;case.action_type=action;case.operation=operation;case.replacement=replacement;case.change_type=change_type
    original=seed(case,attribute,value,trip=attribute=='trip_plan')
    frozen=queue(case,current,reference)
    workflow=ConsolidationWorkflow();assert workflow.run_consolidation_cycle(case.tenant,case.user,frozen)
    row,=memory_rows(case);fact,=row['payload']['facts'];old=original['payload']['facts'][0]
    assert row['id']==original['id'] and row['payload']['entity_name']==original['payload']['entity_name']
    assert fact['fact_id']==old['fact_id'] and fact['_mention_count']==2
    assert fact_text(fact)==(replacement if operation=='UPDATE' else value)
    if change_type=='real_change':assert fact['history_values'][0]['value']==old['value']
    else:assert fact['history_values']==[]
    if operation=='MENTION':assert fact['value']==old['value'] and row['vector']==original['vector']
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
def test_invalid_contextual_plan_completes_without_mutating(contextual_case,defect):
    case=contextual_case;setattr(case,defect,True);seed(case);before=memory_rows(case)
    frozen=queue(case,"No, that's wrong.",'Your Tesla is sold.')
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant,case.user,frozen)
    assert memory_rows(case)==before and processed_count(case)==1
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
    case=contextual_case;case.action_type="END";case.operation="UPDATE";case.replacement="No longer planned";case.change_type="real_change"
    original=seed(case,"trip_plan","Japan",trip=True)
    frozen=queue(case,current,"We discussed that plan.")
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant,case.user,frozen)
    stored,=memory_rows(case)
    assert any(name=="MutationPlanResponse" for name,_ in case.calls)
    assert stored["id"]==original["id"] and stored["payload"]["aliases"]==original["payload"]["aliases"]
    old,=stored["payload"]["facts"]
    assert old["fact_id"]==original["payload"]["facts"][0]["fact_id"] and fact_text(old)=="No longer planned"



@pytest.mark.parametrize("defect", [None, "foreign_source", "assistant_source", "foreign_fact"])
def test_contextual_nonliteral_reference_keeps_structural_boundaries(monkeypatch, defect):
    from unittest.mock import Mock
    from tomte.engine import Tomte
    key=str(uuid4())
    payload=simple_entity('Marathon training',entity_id=key,kind='marathon_training',domain='activities_and_skills',
        facts=[simple_fact('training_status','training')])
    row={"id":key,**{k:payload[k] for k in ("entity_id","entity_name","entity_type","entity_domain")},"payload":payload}
    current=[{"item_id":"u1","timestamp":1791547200,"role":"assistant" if defect=="assistant_source" else "user","content":"我不再继续了"}]
    calls=[]
    def model(system_prompt,user_text,schema):
        calls.append(schema.__name__)
        if schema is ContextualActionResponse:
            return schema.model_validate({"actions":[{"action_type":"END","reference_hint":"the activity","evidence_quote":"我不再继续了",
                "source_chat_ids":["foreign" if defect=="foreign_source" else "u1"],
                "target":{"db_id":key,"resolution_confidence":.98,"evidence":"RESOLVED_REFERENCE"},
                "target_fact_id":"foreign-fact" if defect=="foreign_fact" else payload["facts"][0]["fact_id"]}]})
        target,=json.loads(user_text)["TARGETS"]
        return schema.model_validate({"plans":[{"target_key":target["target_key"],"operations":[{
            "operation":"UPDATE","fields":field_entries("No longer planned"),"change_type":"real_change","target_fact_id":payload["facts"][0]["fact_id"],"source_chat_ids":["u1"]}]}]})
    db=Mock();db.get_resolution_candidates.return_value=[row]
    monkeypatch.setattr(Tomte,"_db",db);monkeypatch.setattr(Tomte,"_llm",SimpleNamespace(parse_structured=model))
    monkeypatch.setattr("tomte.engine.workflows.consolidation.get_dictionary_service",lambda:Mock())
    workflow=ConsolidationWorkflow();data={"CURRENT_BATCH":current,"RECENT_CONTEXT_REFERENCE_ONLY":[]}
    if defect in {"foreign_source","foreign_fact"}:
        assert workflow._resolve_contextual_actions("tenant","user",data,[1.0]) == []
        assert calls==["ContextualActionResponse"]
    else:
        prepared=workflow._resolve_contextual_actions("tenant","user",data,[1.0])
        if defect=="assistant_source":assert prepared==[]
        else:
            planned=workflow._plan_fact_mutations(prepared,data,"contextual-unit-batch")
            assert calls==["ContextualActionResponse","MutationPlanResponse"]
            old,=planned[0]["payload"]["facts"]
            assert old["fact_id"]==payload["facts"][0]["fact_id"] and fact_text(old)=="No longer planned"
