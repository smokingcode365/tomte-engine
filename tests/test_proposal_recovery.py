"""Recovery preserves independent supported memory without guessing semantics."""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from tomte.engine.core.assertions import FactMutation, MutationPlanResponse, apply_mutations


CHAT = [{"item_id": "m1", "role": "user", "timestamp": 1793613600,
         "content": "Bob works at Cedar Labs and enjoys cycling."}]
SCOPE = "d5bcc612-2b8a-4bca-bc1f-9362d58f1456"


def add(**changes):
    return {"operation": "ADD", "attribute_name": "employment", "fields": [
        {"field_name": "employer_name", "text": "Cedar Labs", "fact_confidence": "high"}],
        "source_chat_ids": ["m1"], **changes}


@pytest.mark.parametrize("mode,change", [("replace_value", "enrichment"),
                                        ("update_fields", "real_change")])
def test_add_ignores_update_only_controls_without_changing_content(mode, change):
    operation = add(update_mode=mode, change_type=change)
    original = deepcopy(operation)
    result = apply_mutations([], [operation], CHAT, SCOPE, "batch")
    assert result[0]["value"]["employer_name"]["text"] == "Cedar Labs"
    assert result[0]["history_values"] == []
    assert operation == original


def test_exact_repetition_is_harmless_but_conflicting_fields_are_not():
    operation = add(source_chat_ids=["m1", "m1"])
    operation["fields"] *= 2
    fact, = apply_mutations([], [operation], CHAT, SCOPE, "batch")
    assert fact["_mention_count"] == 1 and len(fact["value"]) == 1
    operation["fields"][1] = {**operation["fields"][1], "text": "Different Labs"}
    with pytest.raises(ValidationError, match="Conflicting"):
        FactMutation.model_validate(operation)


def test_invalid_operation_keeps_valid_sibling_and_original_position():
    bad = add()
    del bad["fields"][0]["fact_confidence"]
    raw = {"plans": [{"target_key": "target_0", "operations": [bad, add()]}]}
    result = MutationPlanResponse.model_validate(raw)
    assert len(result.plans[0].operations) == 1
    assert result.original_index("plans", 0, "operations", 0) == 1
    assert result._rejected == [("plans", 0, "operations", 0)]
    assert "_raw" not in result.model_dump()
    assert raw["plans"][0]["operations"][0] == bad


def test_repair_cannot_rewrite_valid_sibling_or_change_rejected_target():
    bad = add(target_fact_id=SCOPE)
    first = MutationPlanResponse.model_validate({"plans": [{"target_key": "target_0", "operations": [add(), bad]}]})
    other = add(); other["fields"][0]["text"] = "Invented rewrite"
    repaired = MutationPlanResponse.model_validate({"plans": [{"target_key": "target_0", "operations": [other, add()]}]})
    result = first.merge_repair(repaired)
    assert result.plans[0].operations[0].fields[0].text == "Cedar Labs"
    assert len(result.plans[0].operations) == 1
    assert result._rejected


def test_one_model_repair_preserves_siblings_and_accepts_corrected_slot():
    from pydantic_ai import Agent
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from tomte.engine.llm.recovery import install_recovery, run_with_recovery
    calls = []
    def respond(messages, info):
        bad = add()
        valid = add()
        if not calls:
            del bad["fields"][0]["fact_confidence"]
        else:
            valid["fields"][0]["text"] = "Unwanted rewrite"
        calls.append(1)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
            {"plans": [{"target_key": "target_0", "operations": [valid, bad]}]})])
    agent = install_recovery(Agent(FunctionModel(respond), output_type=MutationPlanResponse, retries=1), MutationPlanResponse)
    result = run_with_recovery(agent, MutationPlanResponse, "Synthetic input").output
    assert len(calls) == 2
    assert [op.fields[0].text for op in result.plans[0].operations] == ["Cedar Labs", "Cedar Labs"]
    assert result._repair_attempted and not result._rejected


def prepared(response, facts=()):
    from types import SimpleNamespace
    from tomte.engine.workflows.consolidation import ConsolidationWorkflow
    workflow = object.__new__(ConsolidationWorkflow)
    workflow.llm = SimpleNamespace(parse_structured=lambda **kw: MutationPlanResponse.model_validate(response))
    entity = dict(entity_id='tmp_bob', entity_name='Bob', entity_domain='relationships', entity_type='person', aliases=[], facts=[])
    record = None
    if facts:
        payload = {**entity, 'entity_id': SCOPE, 'facts': deepcopy(list(facts))}
        record = dict(id=SCOPE, payload=payload, **{k: payload[k] for k in ('entity_name','entity_domain','entity_type')})
    op = workflow._prepare_operation('tenant', 'user', entity, record)
    op.update(entity_id=SCOPE)
    if not record:
        op['reserved_id'] = SCOPE
    op['payload']['entity_id'] = SCOPE
    return workflow, op


def plan(workflow, operations):
    return workflow._plan_fact_mutations(operations, dict(CURRENT_BATCH=CHAT, RECENT_CONTEXT_REFERENCE_ONLY=[]), 'batch')


def test_bad_target_does_not_drop_independent_add_or_renumber_its_uuid():
    from tomte.engine.core.assertions import stable_id
    response = {'plans': [{'target_key': 'target_0', 'operations': [
        dict(operation='MENTION', target_fact_id='076d32af-8a28-459a-9a12-9d950c46b453', source_chat_ids=['m1']), add()]}]}
    workflow, op = prepared(response)
    result, = plan(workflow, [op])
    fact, = result['payload']['facts']
    assert fact['fact_id'] == stable_id('tomte-fact-v2', SCOPE, 'batch', 1)
    assert fact['value']['employer_name']['text'] == 'Cedar Labs'


def test_failed_same_fact_unit_preserves_old_value_and_other_new_fact():
    old, = apply_mutations([], [add()], CHAT, SCOPE, 'seed')
    update = dict(operation='UPDATE', target_fact_id=old['fact_id'], fields=[
        dict(field_name='employer_name',text='Pine Labs',fact_confidence='high')],change_type='real_change', source_chat_ids=['m1'])
    bad = dict(operation='UPDATE', target_fact_id=old['fact_id'], remove_fields=['absent'], source_chat_ids=['m1'])
    hobby = add(attribute_name='hobby',fields=[dict(field_name='activity',text='Cycling',fact_confidence='high')])
    workflow, op = prepared({'plans': [dict(target_key='target_0',operations=[update,bad,hobby])]}, [old])
    result, = plan(workflow, [op])
    assert result['payload']['facts'][0] == old
    assert result['payload']['facts'][1]['attribute_name'] == 'hobby'


def test_unaccepted_group_keeps_independent_fact_on_same_entity():
    hobby = add(attribute_name='hobby',incoming_fact_key='fact_1',fields=[dict(field_name='activity',text='Cycling',fact_confidence='high')])
    workflow, op = prepared({'plans': [dict(target_key='target_0',operations=[hobby])]})
    op['extracted_facts'] = [
        dict(attribute_name='visit',value={'place':dict(text='London',fact_confidence='high')},source_chat_id='m1',incoming_fact_key='fact_0',_group_key='shared',_group_targets=[SCOPE,'076d32af-8a28-459a-9a12-9d950c46b453']),
        dict(attribute_name='hobby',value={'activity':dict(text='Cycling',fact_confidence='high')},source_chat_id='m1',incoming_fact_key='fact_1')]
    result, = plan(workflow, [op])
    assert [f['attribute_name'] for f in result['payload']['facts']] == ['hobby']


@pytest.mark.parametrize('second', ['still_invalid', 'malformed'])
def test_failed_repair_keeps_valid_sibling_and_never_calls_a_third_time(second):
    from pydantic_ai import Agent
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from tomte.engine.llm.recovery import install_recovery, run_with_recovery
    calls = []
    def respond(messages, info):
        bad = add(); del bad['fields'][0]['fact_confidence']
        raw = {'plans': [dict(target_key='target_0', operations=[add(), bad])]}
        if calls and second == 'malformed': raw = '{'
        calls.append(1)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, raw)])
    agent = install_recovery(Agent(FunctionModel(respond), output_type=MutationPlanResponse, retries=1), MutationPlanResponse)
    run = run_with_recovery(agent, MutationPlanResponse, 'Synthetic')
    result = run.output
    assert sum(isinstance(m, ModelResponse) for m in run.all_messages()) == 2
    assert len(calls) == 2
    assert len(result.plans[0].operations) == 1
    assert result.plans[0].operations[0].fields[0].text == 'Cedar Labs'
    assert result._rejected and result._repair_attempted


def test_bad_replacement_drops_whole_fact_unit_without_deleting_omitted_fields():
    old, = apply_mutations([], [add(fields=[
        dict(field_name='employer_name',text='Cedar Labs',fact_confidence='high'),
        dict(field_name='job_title',text='Researcher',fact_confidence='high')])], CHAT, SCOPE, 'seed')
    bad = dict(operation='UPDATE',target_fact_id=old['fact_id'],update_mode='replace_value',change_type='real_change',source_chat_ids=['m1'],
        fields=[dict(field_name='employer_name',text='Pine Labs',fact_confidence='high'),dict(field_name='job_title',text='Engineer')])
    mention = dict(operation='MENTION',target_fact_id=old['fact_id'],source_chat_ids=['m1'])
    workflow, op = prepared({'plans':[dict(target_key='target_0',operations=[bad,mention,add(attribute_name='other')])]},[old])
    result, = plan(workflow,[op])
    assert result['payload']['facts'][0] == old
    assert len(result['payload']['facts']) == 2
    assert result['payload']['facts'][1]['attribute_name'] == 'other'


@pytest.mark.parametrize('bad_source', ['unknown', 'assistant'])
def test_extraction_rejects_only_unsupported_fact_without_guessing_source(bad_source):
    from tomte.engine.core.extraction import ExtractionMemorySchema, resolve_extraction
    wire = dict(entity_id='tmp_bob',entity_name='Bob',entity_domain='relationships',entity_type='person',facts=[
        dict(attribute_name='employment',value=add()['fields'],source_chat_id=bad_source),
        dict(attribute_name='hobby',value=[dict(field_name='activity',text='Cycling',fact_confidence='high')],source_chat_id='m1')])
    chat = CHAT + [{**CHAT[0],'item_id':'assistant','role':'assistant'}]
    response = resolve_extraction(ExtractionMemorySchema.model_validate({'relationships':[wire]}),chat)
    assert [f.attribute_name for f in response.relationships[0].facts] == ['hobby']


def test_dangling_group_and_duplicate_handles_preserve_unrelated_extraction():
    from tomte.engine.core.extraction import ExtractionMemorySchema, resolve_extraction
    def entity(handle):
        return dict(entity_id=handle,entity_name=handle,entity_domain='relationships',entity_type='person',facts=[
            dict(attribute_name='employment',value=add()['fields'],source_chat_id='m1'),
            dict(attribute_name='visit',value=add()['fields'],source_chat_id='m1',group_participant_handles=['tmp_missing'])])
    response = resolve_extraction(ExtractionMemorySchema.model_validate({'relationships':[entity('tmp_duplicate'),entity('tmp_duplicate'),entity('tmp_bob')]}),CHAT)
    assert [e.entity_id for e in response.relationships] == ['tmp_bob']
    assert [f.attribute_name for f in response.relationships[0].facts] == ['employment']


@pytest.mark.parametrize('kind', ['MATCH_MANY','UNRESOLVED'])
def test_ambiguous_identity_never_retains_alias_authorization(kind):
    from tomte.engine.core.resolution import EntityResolution
    targets = [dict(db_id=uid,resolution_confidence=1,evidence='EXPLICIT') for uid in (SCOPE,'076d32af-8a28-459a-9a12-9d950c46b453')] if kind == 'MATCH_MANY' else []
    result = EntityResolution.model_validate(dict(extracted_entity_id='tmp_bob',kind=kind,targets=targets,reasoning='synthetic',accepted_aliases=['Group'],alias_evidence=[dict(alias='Group',source_chat_id='m1',evidence_quote='Bob')]))
    assert result.accepted_aliases == [] and result.alias_evidence == []


def test_rejected_group_rebuilds_payload_without_its_values_or_mentions():
    other = '076d32af-8a28-459a-9a12-9d950c46b453'
    shared = add(attribute_name='visit',incoming_fact_key='fact_0',fields=[dict(field_name='place',text='London',fact_confidence='high')])
    hobby = add(attribute_name='hobby',incoming_fact_key='fact_1',fields=[dict(field_name='activity',text='Cycling',fact_confidence='high')])
    workflow, op = prepared({'plans':[dict(target_key='target_0',operations=[shared,hobby])]})
    op['extracted_facts'] = [dict(attribute_name='visit',value={'place':dict(text='London',fact_confidence='high')},source_chat_id='m1',incoming_fact_key='fact_0',_group_key='shared',_group_targets=[SCOPE,other]),
        dict(attribute_name='hobby',value={'activity':dict(text='Cycling',fact_confidence='high')},source_chat_id='m1',incoming_fact_key='fact_1')]
    result, = plan(workflow,[op])
    fact, = result['payload']['facts']
    assert fact['attribute_name'] == 'hobby' and fact['_mention_count'] == 1
    assert workflow.last_mutation_recovery['skipped_groups'] == 1
    assert op['payload']['facts'] == []


def test_explicit_empty_plan_is_not_repaired_or_forced_into_a_write():
    from pydantic_ai import Agent
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from tomte.engine.llm.recovery import install_recovery, run_with_recovery
    calls=[]
    def respond(messages,info):
        calls.append(1)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,{'plans':[]})])
    agent=install_recovery(Agent(FunctionModel(respond),output_type=MutationPlanResponse,retries=1),MutationPlanResponse)
    response=run_with_recovery(agent,MutationPlanResponse,'Synthetic').output
    assert response.plans == [] and len(calls) == 1


@pytest.mark.parametrize('error', ['capacity','cancel','transport'])
def test_proposal_stage_does_not_swallow_infrastructure_or_control_errors(error):
    from tomte.engine.workers.control import WorkStopped
    from tomte.engine.core.capacity import CapacityExceeded
    kinds={'capacity':CapacityExceeded,'cancel':WorkStopped,'transport':TimeoutError}
    workflow,_=prepared({'plans':[]})
    def fail(**kwargs): raise kinds[error]('synthetic')
    workflow.llm.parse_structured=fail
    with pytest.raises(kinds[error]):
        workflow._parse_proposals(system_prompt='synthetic',user_text='synthetic',schema=MutationPlanResponse)


def test_malformed_duplicate_extraction_handle_does_not_authorize_its_sibling():
    from tomte.engine.core.extraction import ExtractionMemorySchema, resolve_extraction
    good = dict(entity_id='tmp_bob',entity_name='Bob',entity_type='person',entity_domain='relationships',facts=[])
    malformed = {**good,'entity_name':None}
    response = ExtractionMemorySchema.model_validate({'relationships':[malformed,good]})
    assert resolve_extraction(response, CHAT).relationships == []


def test_malformed_duplicate_identity_decision_does_not_authorize_its_sibling():
    from tomte.engine.core.resolution import ResolutionResponse
    valid = dict(extracted_entity_id='tmp_bob',kind='NEW',explicitly_new=True,new_entity_evidence='Bob',reasoning='Synthetic')
    response = ResolutionResponse.model_validate({'resolutions':[{**valid,'kind':'invalid'},valid]})
    workflow, _ = prepared({'plans':[]})
    workflow.llm.parse_structured = lambda **kw: response
    entity = dict(entity_id='tmp_bob',entity_name='Bob',entity_type='person',entity_domain='relationships',facts=[],aliases=[])
    assert workflow._reconcile_with_golden_records('tenant','user',[entity],[],current_batch=CHAT) == []


def test_cascading_group_rejection_rebuilds_from_original_facts():
    other = '076d32af-8a28-459a-9a12-9d950c46b453'
    third = '4d88c23c-1d4b-4977-9f49-3c7f46f72e83'
    old, = apply_mutations([], [add()], CHAT, SCOPE, 'seed')
    def fields(key,text): return [dict(field_name=key,text=text,fact_confidence='high')]
    def incoming(key,group,targets,name,text):
        return dict(incoming_fact_key=key,_group_key=group,_group_targets=targets,attribute_name='employment',
                    value={name:dict(text=text,fact_confidence='high')},source_chat_id='m1')
    first = dict(operation='UPDATE',target_fact_id=old['fact_id'],fields=fields('employer_name','Pine Labs'),
                 change_type='real_change',source_chat_ids=['m1'],incoming_fact_key='fact_0')
    second = {**first,'fields':fields('job_title','Researcher'),'incoming_fact_key':'fact_1'}
    shared = add(incoming_fact_key='fact_0',fields=fields('job_title','Researcher'))
    hobby = add(attribute_name='hobby',incoming_fact_key='fact_1',fields=fields('activity','Cycling'))
    workflow, a = prepared({'plans':[dict(target_key='target_0',operations=[first,second]),
                                    dict(target_key='target_1',operations=[shared,hobby])]},[old])
    _, c = prepared({'plans':[]})
    c.update(entity_id=third,reserved_id=third); c['payload']['entity_id']=third
    a['extracted_facts']=[incoming('fact_0','g1',[SCOPE,other],'employer_name','Pine Labs'),
                          incoming('fact_1','g2',[SCOPE,third],'job_title','Researcher')]
    c['extracted_facts']=[incoming('fact_0','g2',[SCOPE,third],'job_title','Researcher'),
                          dict(attribute_name='hobby',incoming_fact_key='fact_1',source_chat_id='m1',value={'activity':dict(text='Cycling',fact_confidence='high')})]
    result, = plan(workflow,[a,c])
    assert result['entity_id'] == third
    fact, = result['payload']['facts']
    assert fact['attribute_name'] == 'hobby' and fact['_mention_count'] == 1
    assert a['payload']['facts'] == [old]
    assert workflow.last_mutation_recovery['skipped_groups'] == 2


def test_recovered_pipeline_commits_once_and_replay_makes_no_model_calls():
    from test_memory_pipeline import MemoryDB, LLM, chat
    from tomte.engine.workflows.consolidation import ConsolidationWorkflow
    class RecoveringLLM(LLM):
        def parse_structured(self, system_prompt, user_text, schema):
            result = super().parse_structured(system_prompt,user_text,schema)
            if schema is MutationPlanResponse:
                data=result.model_dump(mode='json')
                data['plans'][0]['operations'].insert(0,dict(operation='UPDATE',target_fact_id='unknown',
                    fields=add()['fields'],source_chat_ids=['m1']))
                for operation in data['plans'][0]['operations'][1:]:
                    operation.update(update_mode='replace_value',change_type='real_change')
                return schema.model_validate(data)
            return result
    db,llm=MemoryDB(),RecoveringLLM()
    workflow=ConsolidationWorkflow(db,llm)
    assert workflow.run_consolidation_cycle('tenant','user',chat())
    before=deepcopy(db.rows); calls=len(llm.calls)
    assert len(before)==1 and len(before[0]['payload']['facts'])==2
    assert all(f['_mention_count']==1 and f['history_values']==[] for f in before[0]['payload']['facts'])
    assert workflow.run_consolidation_cycle('tenant','user',chat())
    assert db.rows==before and len(db.processed)==1 and len(llm.calls)==calls


def test_corrupt_stored_fact_still_aborts_before_planning():
    old, = apply_mutations([], [add()], CHAT, SCOPE, 'seed')
    old['value']['employer_name']['text'] = None
    workflow, op = prepared({'plans':[]},[old])
    with pytest.raises(ValidationError):
        plan(workflow,[op])


@pytest.mark.parametrize('provider_name',['openai','anthropic','gemini'])
@pytest.mark.parametrize('stage',['extraction','mutation'])
def test_native_provider_recovery_routes_keep_one_repair_and_valid_siblings(provider_name,stage):
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.messages import ModelResponse,ToolCallPart
    from tomte.engine.core.extraction import ExtractionMemorySchema
    from tomte.engine.providers.openai_provider import OpenAILLMProvider
    from tomte.engine.providers.anthropic_provider import AnthropicLLMProvider
    from tomte.engine.providers.gemini_provider import GeminiLLMProvider
    cls={'openai':OpenAILLMProvider,'anthropic':AnthropicLLMProvider,'gemini':GeminiLLMProvider}[provider_name]
    provider=cls.__new__(cls)
    calls=[]
    def respond(messages,info):
        good,bad=add(),add()
        if not calls: del bad['fields'][0]['fact_confidence']
        else: good['fields'][0]['text']='Unwanted rewrite'
        if stage=='mutation': raw={'plans':[dict(target_key='target_0',operations=[good,bad])]}
        else:
            raw={'relationships':[dict(entity_id='tmp_bob',entity_name='Bob',entity_domain='relationships',entity_type='person',facts=[
                dict(attribute_name='employment',value=op['fields'],source_chat_id='m1') for op in (good,bad)])]}
        calls.append(1)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,raw)])
    provider.strong_model=provider.cheap_model=FunctionModel(respond)
    provider.api_key='synthetic'; provider.agent_retries=0
    schema=ExtractionMemorySchema if stage=='extraction' else MutationPlanResponse
    result=provider.parse_structured(None if stage=='extraction' else 'Synthetic', 'Synthetic', schema)
    assert len(calls)==2 and result._repair_attempted and not result._rejected
    values=[f.value[0].text for f in result.relationships[0].facts] if stage=='extraction' else [op.fields[0].text for op in result.plans[0].operations]
    assert values==['Cedar Labs','Cedar Labs']


def test_parallel_runs_on_shared_agent_keep_recovery_state_separate():
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from pydantic_ai import Agent
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.messages import ModelResponse,ToolCallPart,UserPromptPart
    from tomte.engine.llm.recovery import install_recovery,run_with_recovery
    barrier=Barrier(2)
    calls={'A':0,'B':0}
    def respond(messages,info):
        name=next(p.content for m in messages for p in m.parts if isinstance(p,UserPromptPart))
        calls[name]+=1
        good,bad=add(),add()
        good['fields'][0]['text']=name
        if calls[name]==1:
            barrier.wait(timeout=10)
            del bad['fields'][0]['fact_confidence']
        else: good['fields'][0]['text']='Unwanted rewrite'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,{'plans':[dict(target_key='target_0',operations=[good,bad])]})])
    agent=install_recovery(Agent(FunctionModel(respond),output_type=MutationPlanResponse,retries=1),MutationPlanResponse)
    def run(name):
        loop=asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        try: return run_with_recovery(agent,MutationPlanResponse,name).output
        finally: loop.close(); asyncio.set_event_loop(None)
    with ThreadPoolExecutor(max_workers=2) as workers:
        results=list(workers.map(run,['A','B']))
    assert [r.plans[0].operations[0].fields[0].text for r in results]==['A','B']
    assert calls=={'A':2,'B':2}


def test_unusable_envelope_abstains_after_one_native_repair():
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.messages import ModelResponse,ToolCallPart
    from tomte.engine.providers.openai_provider import OpenAILLMProvider
    calls=[]
    def respond(messages,info):
        calls.append(1)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,{'plans':None})])
    provider=OpenAILLMProvider.__new__(OpenAILLMProvider)
    provider.strong_model=FunctionModel(respond); provider.api_key='synthetic'
    workflow,op=prepared({'plans':[]}); workflow.llm=provider
    assert plan(workflow,[op])==[] and len(calls)==2
    assert workflow.last_proposal_recovery['MutationPlanResponse']['unusable_response'] is True


def test_malformed_plan_key_does_not_abort_independent_target():
    workflow, op = prepared({'plans':[dict(target_key=['bad'],operations=[]),dict(target_key='target_0',operations=[add()])]})
    result, = plan(workflow,[op])
    assert result['payload']['facts'][0]['value']['employer_name']['text']=='Cedar Labs'


@pytest.mark.parametrize('kind',[[],{}])
def test_nonscalar_identity_kind_is_rejected_as_one_item(kind):
    from tomte.engine.core.resolution import ResolutionResponse
    result=ResolutionResponse.model_validate({'resolutions':[
        dict(extracted_entity_id='tmp_bad',kind=kind,reasoning='Synthetic'),
        dict(extracted_entity_id='tmp_good',kind='UNRESOLVED',reasoning='Synthetic')]})
    assert [r.extracted_entity_id for r in result.resolutions]==['tmp_good']
    assert result._rejected==[('resolutions',0)]
