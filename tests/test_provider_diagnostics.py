"""Offline metadata diagnostics through the real mutation parser and native retries."""
import json
from unittest.mock import Mock

import httpx
import pytest
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai import capture_run_messages
from pydantic_ai.messages import TextPart
from types import SimpleNamespace
from pydantic_ai.usage import RequestUsage

from tomte.engine.core.assertions import MutationPlanResponse
from tomte.engine.core.config import TomteSettings, settings
from tomte.engine.providers import diagnostics
from test_openai_provider import provider
from test_provider_retry_diagnostics import plan, model_for

PRIVATE = "synthetic-private-value"
SECRET = "sk-synthetic-credential-no-real-key"


class GateStopped(RuntimeError):
    pass


@pytest.fixture
def enabled(monkeypatch, caplog):
    monkeypatch.setattr(settings, "provider_diagnostics_enabled", True)
    caplog.set_level("INFO", logger=diagnostics.logger.name)
    return caplog


def events(caplog):
    return [json.loads(r.message) for r in caplog.records if r.name == diagnostics.logger.name]


def test_mutation_validation_survives_gate_masking(provider, enabled):
    calls = []
    def respond(messages, info):
        calls.append(messages)
        if len(calls) > 1:
            raise GateStopped("Automatic retry or split refused")
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, plan(SECRET))],
            usage=RequestUsage(input_tokens=17, output_tokens=9),
            provider_response_id="resp_" + SECRET, finish_reason="tool_call")
    provider.strong_model = FunctionModel(respond)
    # Same outer masking behavior as a bounded evaluation provider operation.
    with pytest.raises(GateStopped, match="Provider operation stopped"):
        try:
            provider.parse_structured(SECRET, PRIVATE, MutationPlanResponse)
        except BaseException:
            raise GateStopped("Provider operation stopped") from None
    event = events(enabled)[0]
    assert event["category"] == "schema_validation"
    assert event["validation_errors"] == [{"code":"value_error", "path":["plans",0,"operations",0], "constraint":"replacement_forbidden"}]
    assert event["schema"] == "MutationPlanResponse" and event["stage"] == "fact_mutation"
    assert event["stop_reason"] == "retry_refused"
    assert event["transport_succeeded"] is True and event["tool_output_count"] == 1
    assert event["usage"] == {"input_tokens":17,"output_tokens":9}
    assert len(calls) == 2  # Native retry reached the refusal, not another successful transport.
    assert SECRET not in enabled.text and PRIVATE not in enabled.text


@pytest.mark.parametrize("active", [False, True])
def test_switch_preserves_success_and_native_retry_count(provider, monkeypatch, caplog, active):
    monkeypatch.setattr(settings, "provider_diagnostics_enabled", active)
    caplog.set_level("INFO", logger=diagnostics.logger.name)
    calls = []; provider.strong_model = model_for([plan(SECRET), plan()], calls)
    result = provider.parse_structured(SECRET, PRIVATE, MutationPlanResponse)
    assert result.model_dump() == MutationPlanResponse.model_validate(plan()).model_dump()
    assert len(calls) == 2
    assert len(events(caplog)) == int(active)
    if active:
        assert events(caplog)[0]["outcome"] == "success"
        assert events(caplog)[0]["retry_prompts"] == 1


@pytest.mark.parametrize("active", [False, True])
def test_switch_preserves_failure_and_native_retry_count(provider, monkeypatch, caplog, active):
    monkeypatch.setattr(settings, "provider_diagnostics_enabled", active)
    caplog.set_level("INFO", logger=diagnostics.logger.name)
    calls = []; provider.strong_model = model_for([plan(SECRET)], calls)
    with pytest.raises(UnexpectedModelBehavior):
        provider.parse_structured(SECRET, PRIVATE, MutationPlanResponse)
    assert len(calls) == 2 and len(events(caplog)) == int(active)


@pytest.mark.parametrize("error, category", [
    (httpx.ConnectError(SECRET), "transport_network"),
    (httpx.ReadTimeout(SECRET), "timeout"),
    (ModelHTTPError(429, SECRET, body={"secret":PRIVATE}), "http_rejection"),
    (GateStopped("Spend reservation ceiling reached"), "budget_stop"),
    (GateStopped("Experiment deadline reached"), "deadline_stop"),
    (GateStopped(SECRET), "gate_stop_unknown"),
    (UsageLimitExceeded(SECRET), "usage_limit_stop"),
    (RuntimeError(SECRET), "engine_or_host_logic_error"),
])
def test_failure_categories_exception_identity_and_redaction(provider, enabled, error, category):
    provider.embedder.embed_query_sync.side_effect = error
    with pytest.raises(type(error)) as raised:
        provider.embed_text(PRIVATE)
    assert raised.value is error
    provider.embedder.embed_query_sync.assert_called_once_with(PRIVATE)
    event = events(enabled)[0]
    assert event["category"] == category and event["duration_ms"] >= 0
    assert event["transport_succeeded"] is (True if category == "http_rejection" else None)
    assert SECRET not in enabled.text and PRIVATE not in enabled.text


def test_untrusted_validation_keys_inputs_context_messages_are_excluded():
    errors = [{"type":SECRET,"loc":("plans",SECRET,0),"msg":SECRET,"input":PRIVATE,"ctx":{"error":ValueError(SECRET)}}]
    assert diagnostics._errors(errors, MutationPlanResponse) == [{"code":"custom_error","path":["plans","<field>",0]}]


def test_logger_failure_preserves_original_exception(provider, enabled, monkeypatch):
    error = RuntimeError(SECRET)
    provider.embedder.embed_query_sync.side_effect = error
    monkeypatch.setattr(diagnostics.logger, "log", Mock(side_effect=ValueError(SECRET)))
    with pytest.raises(RuntimeError) as raised:
        provider.embed_text(PRIVATE)
    assert raised.value is error


def test_default_is_off_and_disabled_does_not_capture(provider, monkeypatch):
    assert TomteSettings.model_fields["provider_diagnostics_enabled"].default is False
    monkeypatch.setenv("PROVIDER_DIAGNOSTICS_ENABLED", "true")
    assert TomteSettings(_env_file=None).provider_diagnostics_enabled is True
    monkeypatch.setattr(settings, "provider_diagnostics_enabled", False)
    capture = Mock(side_effect=AssertionError("must not capture"))
    monkeypatch.setattr(diagnostics, "capture_run_messages", capture)
    provider.embedder.embed_query_sync.return_value.embeddings = [[1.0]]
    assert provider.embed_text(PRIVATE) == [1.0]
    capture.assert_not_called()


def test_embedding_usage_and_no_vectors_are_emitted(provider, enabled):
    provider.embedder.embed_query_sync.return_value = SimpleNamespace(embeddings=[[0.123456789]],
        usage=RequestUsage(input_tokens=11))
    assert provider.embed_text(PRIVATE) == [0.123456789]
    event = events(enabled)[0]
    assert event["usage"] == {"input_tokens":11,"output_tokens":0}
    assert "0.123456789" not in enabled.text


def test_malformed_json_and_multiple_output_metadata(provider, enabled):
    calls = []
    def respond(messages, info):
        calls.append(1)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{invalid json'),
                                   ToolCallPart(info.output_tools[0].name, '{invalid json')],
                             finish_reason="length", provider_details={"status":"incomplete","secret":SECRET})
    provider.strong_model = FunctionModel(respond)
    with pytest.raises(UnexpectedModelBehavior):
        provider.parse_structured(SECRET, PRIVATE, MutationPlanResponse)
    event = events(enabled)[0]
    assert event["category"] == "malformed_response"
    assert "IncompleteToolCall" in event["exception_chain"]
    assert event["tool_output_count"] == 2 and event["response_status"] == "incomplete"
    assert event["finish_reason"] == "length"
    assert SECRET not in enabled.text


def test_zero_tool_outputs_and_nested_history_not_misattributed(provider, enabled):
    def respond(messages, info):
        return ModelResponse(parts=[TextPart(PRIVATE)])
    provider.strong_model = FunctionModel(respond)
    with pytest.raises(UnexpectedModelBehavior):
        provider.parse_structured(SECRET, PRIVATE, MutationPlanResponse)
    assert events(enabled)[0]["tool_output_count"] == 0
    enabled.clear()
    with capture_run_messages():
        provider.strong_model = model_for([plan()], [])
        provider.parse_structured(SECRET, PRIVATE, MutationPlanResponse)
        provider.strong_model = model_for([plan(SECRET)], [])
        with pytest.raises(UnexpectedModelBehavior):
            provider.parse_structured(SECRET, PRIVATE, MutationPlanResponse)
    assert events(enabled)[-1]["history_available"] is False
    assert events(enabled)[-1]["response_count"] is None


def test_provider_http_ids_and_body_are_not_logged(provider, enabled):
    from openai import RateLimitError
    response = httpx.Response(429, request=httpx.Request("POST", "https://example.invalid/"),
                              headers={"x-request-id":SECRET})
    error = RateLimitError(SECRET, response=response, body={"error":{"message":PRIVATE}})
    provider.embedder.embed_query_sync.side_effect = error
    with pytest.raises(RateLimitError):
        provider.embed_text(PRIVATE)
    event = events(enabled)[0]
    assert event["http_status"] == 429 and event["category"] == "http_rejection"
    assert event["provider_request_id_fingerprint"] == diagnostics._fingerprint(SECRET)
    assert SECRET not in enabled.text and PRIVATE not in enabled.text


@pytest.mark.parametrize("reason, category", [
    ("Spend reservation ceiling reached", "budget_stop"),
    ("Experiment deadline reached", "deadline_stop"),
])
def test_wrapped_gate_stop_is_not_misclassified_as_network(provider, enabled, reason, category):
    from openai import APIConnectionError
    error = APIConnectionError(request=httpx.Request("POST", "https://example.invalid/"))
    error.__cause__ = GateStopped(reason)
    provider.embedder.embed_query_sync.side_effect = error
    with pytest.raises(APIConnectionError) as raised:
        provider.embed_text(PRIVATE)
    assert raised.value is error
    assert events(enabled)[0]["category"] == category
