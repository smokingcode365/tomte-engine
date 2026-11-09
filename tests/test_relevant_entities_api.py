"""Read-only SDK retrieval; no model HTTP, database or Redis required."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from memory_fixtures import STAMP, simple_entity, simple_fact
from tomte.engine import Tomte
from tomte.engine.clients.chat_client import TomteChatClient
from tomte.engine.core.config import settings
from tomte.engine.core.memory import public_entity, public_value
from tomte.engine.exceptions import (
    TomteInvalidIdError, TomteInvalidInputError, TomteSecurityViolationError,
    TomteSessionStateUnavailable,
)


@pytest.fixture
def retrieval(monkeypatch):
    payload = simple_entity("My hospital job", facts=[simple_fact("employment", "Doctor at Harbour Hospital")])
    row = {**payload, "id": payload["entity_id"], "tenant_id": "tenant-a", "user_id": "user-a",
           "relevance_score": 0.8, "private_metadata": "not public"}
    db = Mock(spec=["get_raw_semantic_context", "get_history_candidates"])
    db.get_raw_semantic_context.return_value = [row]
    db.get_history_candidates.return_value = []
    model = Mock(spec=["embed_text", "parse_structured", "is_junk"])
    model.embed_text.return_value = [1.0, 0.0]
    model.parse_structured.side_effect = AssertionError("No generative LLM calls allowed")
    model.is_junk.side_effect = AssertionError("No junk-filter calls allowed")
    lifecycle = []

    @contextmanager
    def operation(control):
        lifecycle.append("open")
        try:
            yield db, model
        finally:
            lifecycle.append("closed")

    context = Mock(side_effect=operation)
    usage = SimpleNamespace(scope=Mock(side_effect=lambda *args: nullcontext()))
    monkeypatch.setattr(Tomte, "operation_context", context)
    monkeypatch.setattr(Tomte, "_usage", usage)
    monkeypatch.setattr(Tomte, "get_cache", Mock(side_effect=AssertionError("Read API must not use Redis")))
    return SimpleNamespace(client=TomteChatClient("tenant-a"), db=db, model=model,
                           row=row, context=context, lifecycle=lifecycle, usage=usage)


def test_chat_statement_returns_public_entities_without_ingestion(retrieval):
    r = retrieval
    before = deepcopy(r.row)
    message = "I changed my job last month"
    result = r.client.get_relevant_entities("user-a", message, limit=3)

    assert result == [public_entity(before)]
    r.model.embed_text.assert_called_once_with(message)
    assert [call[0] for call in r.model.mock_calls] == ["embed_text"]
    r.db.get_raw_semantic_context.assert_called_once_with("tenant-a", "user-a", [1.0, 0.0], limit=3)
    assert [call[0] for call in r.db.mock_calls] == ["get_raw_semantic_context"]
    r.usage.scope.assert_called_once_with("tenant-a", "user-a", "foreground")
    assert r.lifecycle == ["open", "closed"]
    assert r.row == before
    result[0]["facts"][0]["value"]["detail"]["text"] = "caller edit"
    assert r.row == before


def test_large_history_uses_index_order_without_llm_or_storage_changes(retrieval, monkeypatch):
    r = retrieval
    fact = r.row["facts"][0]
    fact["history_values"] = [
        {"_archived_at": STAMP, "value": simple_fact("employment", text)["value"]}
        for text in ("Earlier work at Cedar Labs", "Later work at Pine Labs", "Former hospital job")
    ]
    budget = len(json.dumps({"value": public_value(fact["history_values"][0]["value"])}, ensure_ascii=False).encode("utf-8"))
    monkeypatch.setattr(settings, "memory_history_bytes", budget)
    r.db.get_history_candidates.return_value = [
        {"entity_uuid": r.row["entity_id"], "fact_id": fact["fact_id"], "history_index": 0}
    ]
    before = deepcopy(r.row)

    result = r.client.get_relevant_entities("user-a", "Cedar Labs reminds me of my old job")

    assert result == [public_entity(before, {fact["fact_id"]: [0]})]
    assert result[0]["facts"][0]["history_is_partial"] is True
    r.db.get_history_candidates.assert_called_once_with(
        "tenant-a", "user-a", [r.row["entity_id"]], [1.0, 0.0], settings.memory_history_candidate_limit)
    assert [call[0] for call in r.model.mock_calls] == ["embed_text"]
    assert r.row == before


def test_thought_returns_first_mention_and_qualified_text_without_generation(retrieval):
    r = retrieval
    r.row.update(entity_domain="thoughts_and_plans", entity_name="Car purchase intention",
                 entity_type="purchase_intention", first_mentioned_at="2026-01-05T10:00:00Z")
    r.row["facts"] = [simple_fact("intention", "As of 5 January 2026, considering buying a Tesla.")]
    before = deepcopy(r.row)
    result = r.client.get_relevant_entities("user-a", "I am thinking about that car again")
    assert result == [public_entity(before)]
    assert result[0]["first_mentioned_at"] == "2026-01-05T10:00:00Z"
    assert result[0]["facts"][0]["value"]["detail"]["text"] == before["facts"][0]["value"]["detail"]["text"]
    assert "_recorded_at" not in json.dumps(result)
    assert [call[0] for call in r.model.mock_calls] == ["embed_text"]
    assert r.row == before


@pytest.mark.parametrize("message", ["I changed my job last month", "我上个月换了工作。", "Where did I work?"])
def test_arbitrary_text_keeps_provider_rank_and_current_values(retrieval, message):
    r = retrieval
    second = {**simple_entity("My former job", facts=[simple_fact("employment", "Part-time work")]),
              "tenant_id": "tenant-a", "user_id": "user-a"}
    r.db.get_raw_semantic_context.return_value = [second, r.row]

    assert r.client.get_relevant_entities("user-a", message) == [public_entity(second), public_entity(r.row)]
    r.model.embed_text.assert_called_once_with(message)
    r.db.get_raw_semantic_context.assert_called_once_with(
        "tenant-a", "user-a", [1.0, 0.0], limit=settings.semantic_retrieval_limit)


def test_empty_user_result_is_empty_without_background_or_history(retrieval, monkeypatch):
    r = retrieval
    r.db.get_raw_semantic_context.return_value = []
    monkeypatch.setattr(Tomte, "_usage", None)
    assert r.client.get_relevant_entities("user-a", "my work") == []
    r.db.get_history_candidates.assert_not_called()
    assert r.lifecycle == ["open", "closed"]


@pytest.mark.parametrize("change", [
    {"tenant_id": "another-tenant"}, {"user_id": "another-user"}, {"tenant_id": None}, {"user_id": None},
])
def test_foreign_or_unscoped_rows_raise_before_public_projection(retrieval, change):
    r = retrieval
    r.db.get_raw_semantic_context.return_value = [{**r.row, **change}]
    with pytest.raises(TomteSecurityViolationError, match="namespace mismatch"):
        r.client.get_relevant_entities("user-a", "my work")
    r.db.get_history_candidates.assert_not_called()
    assert r.lifecycle == ["open", "closed"]


@pytest.mark.parametrize("tenant,user", [("", "user-a"), ("tenant:a", "user-a"), ("tenant-a", ""), ("tenant-a", None)])
def test_invalid_scope_rejected_before_any_provider_call(retrieval, tenant, user):
    with pytest.raises(TomteInvalidIdError):
        TomteChatClient(tenant).get_relevant_entities(user, "my work")
    retrieval.context.assert_not_called()
    assert retrieval.model.mock_calls == retrieval.db.mock_calls == []


@pytest.mark.parametrize("kwargs", [
    {"message": ""}, {"message": "   "}, {"message": None}, {"message": 10},
    {"message": "x" * 11}, {"limit": 0}, {"limit": -1}, {"limit": True}, {"limit": 1.5}, {"limit": "3"},
])
def test_invalid_input_rejected_before_any_provider_call(retrieval, monkeypatch, kwargs):
    monkeypatch.setattr(settings, "max_input_length", 10)
    with pytest.raises(TomteInvalidInputError):
        retrieval.client.get_relevant_entities(**{"user_id": "user-a", "message": "my work", **kwargs})
    retrieval.context.assert_not_called()
    assert retrieval.model.mock_calls == retrieval.db.mock_calls == []


@pytest.mark.parametrize("stage", ["embedding", "retrieval"])
def test_provider_failure_propagates_and_releases_resources(retrieval, stage):
    r = retrieval
    error = TomteSessionStateUnavailable("Synthetic provider failure")
    method = r.model.embed_text if stage == "embedding" else r.db.get_raw_semantic_context
    method.side_effect = error
    with pytest.raises(TomteSessionStateUnavailable) as caught:
        r.client.get_relevant_entities("user-a", "my work")
    assert caught.value is error
    assert r.lifecycle == ["open", "closed"]
    r.db.get_history_candidates.assert_not_called()
