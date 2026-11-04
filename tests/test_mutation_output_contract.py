"""Strict mutation output and transient evidence; synthetic SDK responses only."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from openai import AsyncOpenAI
from pydantic import ValidationError
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.exceptions import UnexpectedModelBehavior
import pytest

from tomte.engine.core.assertions import FactMutation, MutationPlanResponse
from tomte.engine.llm.prompts.fact_mutation import MUTATION_SYSTEM_PROMPT
from tomte.engine.workflows.consolidation import ConsolidationWorkflow
from tomte.engine.core.config import settings
from tomte.engine.core.resolution import ResolutionResponse
from tomte.engine.dictionary.alignment import TerminologyAlignmentResponse, FieldAlignmentResponse
from tomte.engine.core.contextual import ContextualActionResponse
from tomte.engine.providers import diagnostics, openai_provider
from test_openai_provider import provider


def proposal(confidence="high"):
    return MutationPlanResponse.model_validate({"plans": [{"target_key": "target_0", "operations": [{
        "operation": "ADD", "attribute_name": "employment",
        "fields": [{"field_name": "job_title", "text": "cartographer", "fact_confidence": confidence}],
        "source_chat_ids": ["fictional-user-001"]}]}]}).model_dump(mode="json")


def test_actual_mutation_request_is_strict_and_preserves_nullable_fields(provider, monkeypatch):
    requests = []
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json={"id": "resp_synthetic", "object": "response", "created_at": 1,
            "model": "gpt-6-luna", "status": "completed", "output": [{"type": "function_call",
            "id": "fc_synthetic", "call_id": "call_synthetic", "status": "completed",
            "name": body["tools"][0]["name"], "arguments": json.dumps(proposal())}],
            "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30,
                      "output_tokens_details": {"reasoning_tokens": 0}}})
    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    sdk = AsyncOpenAI(api_key="synthetic-not-a-key", http_client=http, max_retries=0)
    provider.strong_model = OpenAIResponsesModel("gpt-6-luna", provider=OpenAIProvider(openai_client=sdk))
    try:
        result = provider.parse_structured(MUTATION_SYSTEM_PROMPT, "Fictional user evidence", MutationPlanResponse)
    finally:
        asyncio.get_event_loop().run_until_complete(sdk.close())
    assert result.model_dump(mode="json") == proposal() and len(requests) == 1
    tool = requests[0]["tools"][0]
    assert tool["strict"] is True
    schema = tool["parameters"]
    for obj in [schema, *schema["$defs"].values()]:
        if obj.get("type") == "object":
            assert set(obj["required"]) == set(obj["properties"])
            assert obj["additionalProperties"] is False
    properties = schema["$defs"]["FactMutation"]["properties"]
    assert "evidence_type" not in properties and "interpretation_confidence" not in properties
    for name in ("target_fact_id", "attribute_name", "incoming_fact_key"):
        assert {a["type"] for a in properties[name]["anyOf"]} == {"string", "null"}
        assert "default" not in properties[name]
    assert schema["$defs"]["WireValueField"]["properties"]["fact_confidence"]["enum"] == ["high", "medium", "low"]


@pytest.mark.parametrize("invalid", ["missing", None, 0.9, "certain", "unrecognized"])
def test_missing_or_invalid_item_confidence_rejected_locally(invalid):
    data = proposal()
    item = data["plans"][0]["operations"][0]["fields"][0]
    if invalid == "missing":
        del item["fact_confidence"]
    else:
        item["fact_confidence"] = invalid
    with pytest.raises(ValidationError) as error:
        FactMutation.model_validate(data["plans"][0]["operations"][0])
    assert error.value.errors()[0]["loc"] == ("fields", 0, "fact_confidence")
    response = MutationPlanResponse.model_validate(data)
    assert response.plans[0].operations == []
    assert response._rejected == [("plans", 0, "operations", 0)]


@pytest.mark.parametrize("confidence", ["high", "medium", "low"])
def test_valid_confidence_survives_preparation_without_durable_source(confidence):
    workflow = ConsolidationWorkflow.__new__(ConsolidationWorkflow)
    workflow.llm = SimpleNamespace(parse_structured=lambda **kwargs: MutationPlanResponse.model_validate(proposal(confidence)))
    entity = {"entity_id": "tmp_person", "entity_name": "Ari", "entity_type": "person",
              "entity_domain": "identity_and_demographics", "aliases": [], "facts": []}
    op = workflow._prepare_operation("fictional-tenant", "fictional-user", entity, None)
    items = [{"item_id": "fictional-user-001", "role": "user", "timestamp": 1735808400,
              "content": "I work as a cartographer."}]
    result = workflow._plan_fact_mutations([op], {"CURRENT_BATCH": items, "RECENT_CONTEXT_REFERENCE_ONLY": []}, "fictional-batch")
    fact = result[0]["payload"]["facts"][0]
    assert fact["value"]["job_title"]["fact_confidence"] == confidence
    assert "source_chat_id" not in fact and "evidence" not in fact
    assert op["payload"]["facts"] == []


def test_prompt_defines_separate_semantics_and_stable_fact_identity():
    for keyword in ("real_change", "correction", "enrichment", "MENTION", "APPEND_HISTORY", "replace_value", "fact_confidence"):
        assert keyword in MUTATION_SYSTEM_PROMPT
    assert "Existing history cannot be corrected" in MUTATION_SYSTEM_PROMPT
    assert "permanent UUIDs" in MUTATION_SYSTEM_PROMPT


@pytest.mark.parametrize("schema", [ResolutionResponse, TerminologyAlignmentResponse, FieldAlignmentResponse, ContextualActionResponse])
def test_nonmutation_agent_output_declaration_remains_typed(provider, monkeypatch, schema):
    expected = schema.model_validate({"resolutions": []} if schema is ResolutionResponse else {})
    agent = Mock()
    agent.run_sync.return_value = SimpleNamespace(output=expected)
    factory = Mock(return_value=agent)
    monkeypatch.setattr(openai_provider, "Agent", factory)
    assert provider.parse_structured("Synthetic instructions", "Synthetic input", schema) is expected
    assert factory.call_args.kwargs["output_type"] is schema


def test_missing_confidence_reaches_safe_diagnostics(provider, monkeypatch):
    monkeypatch.setattr(settings, "provider_diagnostics_enabled", True)
    log = Mock()
    monkeypatch.setattr(diagnostics.logger, "log", log)
    calls = []
    def respond(messages, info):
        calls.append(1)
        data = proposal()
        del data["plans"][0]["operations"][0]["fields"][0]["fact_confidence"]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, data)])
    provider.strong_model = FunctionModel(respond)
    response = provider.parse_structured(MUTATION_SYSTEM_PROMPT, "Synthetic input", MutationPlanResponse)
    assert response.plans[0].operations == []
    assert response.recovery_summary() == {"rejected_items": 1, "repair_attempted": True}
    assert "cartographer" not in repr(log.call_args_list)
    assert len(calls) == 2
