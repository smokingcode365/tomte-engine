"""Native Pydantic-AI retries and metadata-only logging, without external APIs."""
import json
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from pydantic_ai import capture_run_messages
from pydantic_ai.messages import ModelRequest,ModelResponse,RetryPromptPart,ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.exceptions import UnexpectedModelBehavior
from tomte.engine.core.assertions import MutationPlanResponse
from tomte.engine.providers import openai_provider
from test_openai_provider import provider

SECRET='synthetic-secret-do-not-log'
PRIVATE='synthetic-private-conversation-do-not-log'

def plan(value=None,target='existing-fact'):
    return {'plans':[{'target_key':'target_0','operations':[{
        'operation':'REINFORCE','target_fact_id':target,'new_value':value,
        'source_chat_ids':['current-source'],'interpretation_confidence':0.97,
        'evidence_type':'USER_EXPLICIT'}]}]}

def model_for(outputs,calls):
    def model(messages,info):
        calls.append(messages)
        data=outputs[min(len(calls)-1,len(outputs)-1)]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,data)])
    return FunctionModel(model)

def test_native_validation_retry_is_logged_and_repaired_output_is_preserved(provider,monkeypatch):
    calls=[];debug=Mock();monkeypatch.setattr(openai_provider.tomtelog,'debug',debug)
    provider.strong_model=model_for([plan(SECRET),plan()],calls)
    result=provider.parse_structured('Synthetic mutation instructions',PRIVATE,MutationPlanResponse)
    assert result.model_dump()==MutationPlanResponse.model_validate(plan()).model_dump()
    assert len(calls)==2
    metadata=json.loads(debug.call_args.args[0])
    assert metadata['stage']=='fact_mutation' and metadata['schema']=='MutationPlanResponse'
    assert metadata['attempt']==1 and metadata['retry_count']==1
    assert metadata['reason']=='pydantic_validation'
    assert metadata['errors']==[{'type':'value_error','location':['plans',0,'operations',0]}]
    logged=str(debug.call_args_list)
    assert SECRET not in logged and PRIVATE not in logged and provider.api_key not in logged

def test_one_request_success_adds_no_retry_and_preserves_normal_output(provider,monkeypatch):
    calls=[];debug=Mock();monkeypatch.setattr(openai_provider.tomtelog,'debug',debug)
    provider.strong_model=model_for([plan()],calls)
    result=provider.parse_structured('Synthetic instructions',PRIVATE,MutationPlanResponse)
    assert result.model_dump()==MutationPlanResponse.model_validate(plan()).model_dump()
    assert len(calls)==1;debug.assert_not_called()

def test_retry_metadata_excludes_arbitrary_message_input_context_and_field_keys(monkeypatch):
    debug=Mock();monkeypatch.setattr(openai_provider.tomtelog,'debug',debug)
    messages=[ModelResponse(parts=[]),ModelRequest(parts=[
        RetryPromptPart(content=[{'type':SECRET,'loc':('plans',SECRET),'msg':SECRET,'input':PRIVATE,'ctx':{'secret':SECRET}}]),
        RetryPromptPart(content=PRIVATE)])]
    openai_provider._log_structured_retries(SimpleNamespace(all_messages=lambda:messages),MutationPlanResponse,'fact_mutation')
    assert debug.call_count==2
    first=json.loads(debug.call_args_list[0].args[0]);assert first['errors']==[{'type':'custom_error','location':['plans','<field>']}]
    assert json.loads(debug.call_args_list[1].args[0])['reason']=='agent_retry'
    assert SECRET not in str(debug.call_args_list) and PRIVATE not in str(debug.call_args_list)

def test_exhausted_native_retry_still_propagates_without_changing_retry_limit(provider,monkeypatch):
    calls=[];provider.strong_model=model_for([plan(target=None)],calls)
    debug=Mock();monkeypatch.setattr(openai_provider.tomtelog,'debug',debug)
    with pytest.raises(UnexpectedModelBehavior):
        provider.parse_structured('Synthetic instructions',PRIVATE,MutationPlanResponse)
    assert len(calls)==2  # Existing Pydantic-AI default: initial request plus one retry.

    metadata=json.loads(debug.call_args.args[0])
    assert metadata['event']=='structured_output_failure'
    assert metadata['reason']=='pydantic_validation'
    assert metadata['attempt']==2 and metadata['retry_count']==1
    assert metadata['errors']==[{'type':'value_error','location':['plans',0,'operations',0]}]
    assert PRIVATE not in str(debug.call_args_list) and provider.api_key not in str(debug.call_args_list)


def test_failure_diagnostics_do_not_attribute_an_enclosing_runs_history(provider,monkeypatch):
    debug=Mock();monkeypatch.setattr(openai_provider.tomtelog,'debug',debug)
    with capture_run_messages():
        provider.strong_model=model_for([plan()],[])
        provider.parse_structured('Synthetic instructions',PRIVATE,MutationPlanResponse)
        provider.strong_model=model_for([plan(SECRET)],[])
        with pytest.raises(UnexpectedModelBehavior):
            provider.parse_structured('Synthetic instructions',PRIVATE,MutationPlanResponse)
    metadata=json.loads(debug.call_args.args[0])
    assert metadata['event']=='structured_output_failure'
    assert metadata['attempt'] is None and metadata['retry_count'] is None
    assert metadata['reason']=='pydantic_validation'
    assert SECRET not in str(debug.call_args_list) and PRIVATE not in str(debug.call_args_list)


@pytest.mark.parametrize('exhausted',[False,True])
def test_diagnostic_logger_failure_does_not_change_model_results_or_exceptions(provider,monkeypatch,exhausted):
    monkeypatch.setattr(openai_provider.tomtelog,'debug',Mock(side_effect=RuntimeError('Synthetic logger failure')))
    outputs=[plan(target=None)] if exhausted else [plan(SECRET),plan()]
    calls=[];provider.strong_model=model_for(outputs,calls)
    if exhausted:
        with pytest.raises(UnexpectedModelBehavior,match='Exceeded maximum retries'):
            provider.parse_structured('Synthetic instructions',PRIVATE,MutationPlanResponse)
    else:
        result=provider.parse_structured('Synthetic instructions',PRIVATE,MutationPlanResponse)
        assert result.model_dump()==MutationPlanResponse.model_validate(plan()).model_dump()
    assert len(calls)==2
