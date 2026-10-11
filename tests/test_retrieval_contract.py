from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tomte.engine import Tomte
from tomte.engine.core.enums import ChatSenderRole, MemoryDomain
from tomte.engine.providers.postgres_provider import PostgresProvider
from tomte.engine.providers import postgres_provider
from tomte.engine.utils.cognitive import CognitiveProcessor
from tomte.engine.workflows import chat_hook


@pytest.fixture
def entity():
    return {
        "entity_id": "pref_01",
        "entity_name": "Programming preference",
        "entity_domain": MemoryDomain.IDENTITY.value,
        "entity_type": "preference",
        "aliases": ["my favorite language"],
        "facts": [{"attribute_name": "favorite_language", "value": "Python", "importance": 4}],
        "metadata": {"source": "synthetic", "confidence": 1.0},
    }


@pytest.fixture
def provider(monkeypatch):
    # Exercise real provider methods without constructing a database connection.
    db = PostgresProvider.__new__(PostgresProvider)
    db.conn = object()
    # These unit tests isolate normalization; real tests verify transaction behavior.
    monkeypatch.setattr(db, "transaction", nullcontext)
    return db


RETRIEVALS = [
    ("get_important_entities", "get_entities_by_priority", (MemoryDomain.IDENTITY, 3)),
    ("get_recent_entities", "get_recent_entities", (3,)),
    ("get_frequent_entities", "get_frequent_entities", (3,)),
    ("semantic_retrieval", "get_semantic_context", ([1.0, 0.0], 3)),
]


@pytest.mark.parametrize("method,operation,extra", RETRIEVALS)
@pytest.mark.parametrize("wrapped", [False, True], ids=["flat", "postgres-row"])
def test_retrieval_preserves_entity_and_context(monkeypatch, provider, entity, method, operation, extra, wrapped):
    expected = {**entity, "relevance_score": 0.9, "retrieval_metadata": {"rank": 1}}
    row = {"payload": deepcopy(entity), "relevance_score": 0.9, "retrieval_metadata": {"rank": 1}} if wrapped else deepcopy(expected)
    original = deepcopy(row)
    fetch = Mock(return_value=[row])
    monkeypatch.setattr(postgres_provider.operations, operation, fetch)

    retrieved = getattr(provider, method)("synthetic-tenant", "synthetic-user", *extra)

    fetch.assert_called_once_with(provider.conn, "synthetic-tenant", "synthetic-user", *extra)
    adapted=retrieved[0]["facts"][0]
    from uuid import UUID
    assert UUID(adapted["fact_id"]).version==5 and adapted["assertion_state"]=="ACTIVE"
    assert {k:adapted[k] for k in entity["facts"][0]}==entity["facts"][0]
    assert adapted["origin"]=="LEGACY" and adapted["evidence"]==[]
    expected["facts"]=deepcopy(retrieved[0]["facts"])
    assert retrieved == [expected]
    assert row == original  # Normalization does not mutate the database result.
    cog = CognitiveProcessor()
    merged = cog.merge_entities(retrieved)
    assert merged == [expected]
    assert cog.format_entities(merged) == cog.format_entities([expected])
    assert "favorite_language: Python" in cog.format_entities(merged)


@pytest.mark.parametrize("row", [None, {"payload": None}, {"payload": []}, {"payload": {}}, {"entity_id": ""}, {"payload": {"entity_id": 42}}])
def test_malformed_retrieval_raises_without_exposing_payload(monkeypatch, provider, row):
    monkeypatch.setattr(postgres_provider.operations, "get_recent_entities", lambda *args: [row])
    with pytest.raises(ValueError, match="Invalid retrieved memory row at index 0"):
        provider.get_recent_entities("synthetic-tenant", "synthetic-user")


def test_row_metadata_precedence_and_empty_results(monkeypatch, provider, entity):
    monkeypatch.setattr(postgres_provider.operations, "get_semantic_context", lambda *args: [
        {"payload": {**entity, "relevance_score": 0.1}, "relevance_score": 0.9}
    ])
    assert provider.semantic_retrieval("synthetic-tenant", "synthetic-user", [1.0])[0]["relevance_score"] == 0.9
    monkeypatch.setattr(postgres_provider.operations, "get_recent_entities", lambda *args: [])
    assert provider.get_recent_entities("synthetic-tenant", "synthetic-user") == []


def test_registered_chat_context_equivalent_for_flat_and_postgres_rows(monkeypatch, provider, entity):
    monkeypatch.setattr(provider, "find_session_entity_mentions", Mock(return_value=[]))
    monkeypatch.setattr(provider, "discover_session_entities", Mock(return_value=[]))
    heartbeat = Mock()
    monkeypatch.setattr(provider, "upsert_user_activity", heartbeat)
    security = SimpleNamespace(sanitize_input=lambda text: text, check_for_injection=lambda text: True)
    monkeypatch.setattr(chat_hook, "get_security_guard", lambda: security)
    results = []
    for wrapped in (False, True):
        for _, operation, _ in RETRIEVALS:
            row = {"payload": deepcopy(entity)} if wrapped else deepcopy(entity)
            if operation == "get_semantic_context":
                row["relevance_score"] = 0.9
            monkeypatch.setattr(postgres_provider.operations, operation, Mock(return_value=[row]))
        cache = SimpleNamespace(
            promote_session_entities=Mock(),
            get_session_entity_ids=Mock(return_value=[]),
            get_persona_entities=Mock(return_value=[]),
            set_persona_entities=Mock(),
            get_chat_history=Mock(return_value=[]),
            append_chat_history=Mock(),
            enqueue_for_consolidation=Mock(),
        )
        llm = SimpleNamespace(embed_text=Mock(return_value=[1.0, 0.0]))
        monkeypatch.setattr(Tomte, "_db", provider)
        monkeypatch.setattr(Tomte, "_cache", cache)
        monkeypatch.setattr(Tomte, "_llm", llm)

        result = chat_hook.ChatHookWorkflow().on_chat(
            "synthetic-tenant", "synthetic-session", "What language do I like?", ChatSenderRole.USER, "synthetic-user"
        )

        results.append(result)
        llm.embed_text.assert_called_once_with("What language do I like?")
        adapted=cache.set_persona_entities.call_args.args[2][0]
        assert adapted["facts"][0]["assertion_state"]=="ACTIVE" and adapted["facts"][0]["fact_id"]
        assert {k:adapted[k] for k in entity if k!="facts"}=={k:entity[k] for k in entity if k!="facts"}
        assert {k:adapted["facts"][0][k] for k in entity["facts"][0]}==entity["facts"][0]
        cache.set_persona_entities.assert_called_once_with("synthetic-tenant", "synthetic-user", [adapted])
        cache.append_chat_history.assert_called_once()
        cache.enqueue_for_consolidation.assert_called_once()
    assert results[0] == results[1]
    assert "favorite_language: Python" in results[1]["context_text"]
    assert results[1]["context_text"].count("[[ENTITY: pref_01") == 1
