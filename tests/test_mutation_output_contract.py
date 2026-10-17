"""Mutation output protocol/provenance checks; synthetic SDK responses, no AI quality claims."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
from openai import AsyncOpenAI
from pydantic import ValidationError
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.exceptions import UnexpectedModelBehavior
from unittest.mock import Mock
import pytest

from tomte.engine.core.assertions import FactMutation, MutationPlanResponse
from tomte.engine.llm.prompts.fact_mutation import MUTATION_SYSTEM_PROMPT
from tomte.engine.workflows.consolidation import ConsolidationWorkflow
from tomte.engine.core.config import settings
from tomte.engine.core.resolution import ResolutionResponse
from tomte.engine.dictionary.alignment import TerminologyAlignmentResponse
from tomte.engine.core.contextual import ContextualActionResponse
from tomte.engine.providers import diagnostics, openai_provider
from test_openai_provider import provider

LABELS = ["USER_EXPLICIT", "RESOLVED_REFERENCE", "GROUP_EXPLICIT", "USER_IMPLIED", "DERIVED"]


def proposal(label="USER_EXPLICIT"):
    return {"plans":[{"target_key":"target_0","operations":[{
        "operation":"ADD", "target_fact_id":None, "attribute_name":"job_title",
        "new_value":"cartographer", "incoming_fact_key":None, "source_chat_ids":["fictional-user-001"],
        "interpretation_confidence":1.0, "evidence_type":label}]}]}


def test_actual_mutation_request_is_strict_and_preserves_nullable_fields(provider, monkeypatch):
    async def forbid(*args, **kwargs):
        raise AssertionError("Real HTTP forbidden")
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbid)
    requests = []
    def respond(request):
        body = json.loads(request.content); requests.append(body)
        return httpx.Response(200, json={"id":"resp_synthetic", "object":"response", "created_at":1,
            "model":"gpt-6-luna", "status":"completed", "output":[{"type":"function_call",
            "id":"fc_synthetic", "call_id":"call_synthetic", "status":"completed",
            "name":body["tools"][0]["name"], "arguments":json.dumps(proposal())}],
            "usage":{"input_tokens":20, "output_tokens":10, "total_tokens":30,
                     "output_tokens_details":{"reasoning_tokens":0}}})
    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    sdk = AsyncOpenAI(api_key="synthetic-not-a-key", http_client=http, max_retries=0)
    provider.strong_model = OpenAIResponsesModel("gpt-6-luna", provider=OpenAIProvider(openai_client=sdk))
    try:
        result = provider.parse_structured(MUTATION_SYSTEM_PROMPT, "Fictional user evidence", MutationPlanResponse)
    finally:
        asyncio.get_event_loop().run_until_complete(sdk.close())
    assert result.model_dump() == proposal() and len(requests) == 1
    tool = requests[0]["tools"][0]
    assert tool["strict"] is True
    schema = tool["parameters"]
    for obj in [schema, *schema["$defs"].values()]:
        if obj.get("type") == "object":
            assert set(obj["required"]) == set(obj["properties"])
            assert obj["additionalProperties"] is False
    fields = schema["$defs"]["FactMutation"]["properties"]
    assert fields["evidence_type"]["enum"] == LABELS
    assert fields["evidence_type"]["type"] == "string"
    for name in ("target_fact_id", "attribute_name", "new_value"):
        assert {a["type"] for a in fields[name]["anyOf"]} == {"string", "null"}
        assert "default" not in fields[name]


@pytest.mark.parametrize("invalid", ["missing", None, "LEGACY", "EXPLICIT", "unrecognized"])
def test_missing_or_invalid_category_is_rejected_locally(invalid):
    data = proposal(); op = data["plans"][0]["operations"][0]
    if invalid == "missing": del op["evidence_type"]
    else: op["evidence_type"] = invalid
    with pytest.raises(ValidationError) as error:
        MutationPlanResponse.model_validate(data)
    assert error.value.errors()[0]["loc"] == ("plans", 0, "operations", 0, "evidence_type")
    assert FactMutation.model_fields["evidence_type"].is_required()


@pytest.mark.parametrize("label", LABELS)
def test_valid_category_survives_real_mutation_preparation(label):
    workflow = ConsolidationWorkflow.__new__(ConsolidationWorkflow)
    workflow.llm = SimpleNamespace(parse_structured=lambda **kwargs:MutationPlanResponse.model_validate(proposal(label)))
    entity = {"entity_id":"per_01", "entity_name":"Ari", "entity_type":"person",
              "entity_domain":"identity_and_demographics", "facts":[]}
    op = dict(entity, tenant_id="fictional-tenant", user_id="fictional-user", payload=deepcopy(entity),
              extracted_facts=[{"attribute_name":"job_title", "value":"cartographer", "confidence":1.0}])
    items = [{"item_id":"fictional-user-001", "role":"user", "timestamp":1735808400,
              "content":"I work as a cartographer."}]
    result = workflow._plan_fact_mutations([op], {"CURRENT_BATCH":items,"RECENT_CONTEXT_REFERENCE_ONLY":[]}, "fictional-batch")
    stored = json.loads(json.dumps(result[0]["payload"]))["facts"][0]
    assert stored["origin"] == stored["evidence"][0]["evidence_type"] == label
    assert stored["evidence"][0]["source_chat_id"] == "fictional-user-001"
    assert op["payload"]["facts"] == []


def test_prompt_and_field_define_required_categories_and_example():
    field = FactMutation.model_fields["evidence_type"]
    for label in LABELS:
        assert label in MUTATION_SYSTEM_PROMPT and label in field.description
    assert "LEGACY" in MUTATION_SYSTEM_PROMPT and "role" in MUTATION_SYSTEM_PROMPT
    # Validate the actual full example rather than just finding field-name substrings.
    example = MUTATION_SYSTEM_PROMPT.split("COMPLETE OUTPUT EXAMPLE\n", 1)[1].splitlines()[0]
    assert MutationPlanResponse.model_validate_json(example).model_dump() == proposal()


@pytest.mark.parametrize("schema", [ResolutionResponse, TerminologyAlignmentResponse, ContextualActionResponse])
def test_nonmutation_agent_output_declaration_is_unchanged(provider, monkeypatch, schema):
    expected = schema.model_validate({"resolutions":[]} if schema is ResolutionResponse else {})
    agent = Mock(); agent.run_sync.return_value = SimpleNamespace(output=expected)
    factory = Mock(return_value=agent); monkeypatch.setattr(openai_provider, "Agent", factory)
    assert provider.parse_structured("Synthetic instructions", "Synthetic input", schema) is expected
    assert factory.call_args.kwargs["output_type"] is schema


def test_missing_category_still_reaches_safe_diagnostics(provider, monkeypatch):
    monkeypatch.setattr(settings, "provider_diagnostics_enabled", True)
    log = Mock(); monkeypatch.setattr(diagnostics.logger, "log", log)
    calls = []
    def respond(messages, info):
        calls.append(1)
        data = proposal(); del data["plans"][0]["operations"][0]["evidence_type"]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, data)])
    provider.strong_model = FunctionModel(respond)
    with pytest.raises(UnexpectedModelBehavior):
        provider.parse_structured(MUTATION_SYSTEM_PROMPT, "Synthetic input", MutationPlanResponse)
    event = json.loads(log.call_args.args[1])
    assert event["category"] == "schema_validation"
    assert event["validation_errors"] == [{"code":"missing", "path":["plans",0,"operations",0,"evidence_type"]}]
    assert len(calls) == 2  # Existing native initial request plus one output retry.
