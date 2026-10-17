from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tomte.engine import Tomte
from tomte.engine.providers.postgres_provider import PostgresProvider
from tomte.engine.providers.openai_provider import OpenAILLMProvider
from tomte.engine.runtime import owned_operation
from tomte.engine.clients.chat_client import TomteChatClient


def test_database_only_operation_does_not_construct_sdk(monkeypatch):
    db = object.__new__(PostgresProvider)
    child = SimpleNamespace(close=Mock())
    monkeypatch.setattr(db, 'fork', lambda: child)
    constructor = Mock(side_effect=ValueError('invalid synthetic credentials'))
    monkeypatch.setattr('openai.AsyncOpenAI', constructor)
    with owned_operation(db, object.__new__(OpenAILLMProvider)) as (owned, llm):
        assert owned is child
        constructor.assert_not_called()
        with pytest.raises(ValueError, match='invalid synthetic credentials'):
            llm.is_junk('hello')
    child.close.assert_called_once()


@pytest.mark.parametrize('assistant,user', [(False,None),(True,None),(True,'u')])
def test_cache_only_chat_never_opens_database_or_models(monkeypatch, assistant, user):
    cache = Mock()
    cache.get_chat_history.return_value = []
    monkeypatch.setattr(Tomte, '_cache', cache)
    monkeypatch.setattr(Tomte, '_db', object())
    monkeypatch.setattr(Tomte, '_llm', object())
    monkeypatch.setattr(Tomte, '_closing', False)
    monkeypatch.setattr(Tomte, '_active_operations', 0)
    factory = Mock(side_effect=AssertionError('No DB or LLM needed'))
    monkeypatch.setattr(Tomte, '_operation_factory', factory)
    client = TomteChatClient('t')
    method = client.log_assistant_response if assistant else client.send_user_message
    result = method('s', 'A harmless synthetic message', user)
    assert isinstance(result, dict) and not result.get('error')
    factory.assert_not_called()
    cache.append_chat_history.assert_called_once()
    assert Tomte._active_operations == 0


def test_committed_batch_acks_without_sdk_configuration(monkeypatch):
    from test_managed_worker import Harness
    from tomte.engine.workflows.consolidation import ConsolidationWorkflow
    h = Harness()
    h.cache.add()
    items = h.cache.queues[('t', 'u', 's')]
    h.markers.add(('t', 'u', ConsolidationWorkflow._batch_id('t', 'u', items)))
    template = object.__new__(PostgresProvider)
    child = SimpleNamespace(user_guard=lambda *args:h.guard(),
                            is_consolidation_processed=h.processed, close=Mock())
    monkeypatch.setattr(template, 'fork', lambda:child)
    constructor = Mock(side_effect=ValueError('missing synthetic credentials'))
    monkeypatch.setattr('openai.AsyncOpenAI', constructor)
    h.worker.factory = lambda control:owned_operation(template, object.__new__(OpenAILLMProvider), control)
    try:
        h.step()
        assert not h.calls and h.cache.terminal[-1][3] == 'committed'
        constructor.assert_not_called()
        child.close.assert_called_once()
    finally:
        assert h.worker.stop()


def test_partial_sdk_construction_closes_loop_and_client(monkeypatch):
    import asyncio
    from tomte.engine.runtime import LazyOpenAI
    from unittest.mock import AsyncMock
    clients = []
    def make_client(**kwargs):
        client = SimpleNamespace(close=AsyncMock())
        clients.append(client)
        return client
    monkeypatch.setattr('openai.AsyncOpenAI', make_client)
    monkeypatch.setattr('pydantic_ai.providers.openai.OpenAIProvider',
                        Mock(side_effect=ValueError('synthetic model initialization failure')))
    lazy = LazyOpenAI()
    for _ in range(2):
        with pytest.raises(ValueError, match='synthetic model initialization'):
            lazy.is_junk('hello')
        assert lazy.loop is None and lazy.client is None
    assert len(clients) == 2
    assert all(c.close.await_count == 1 for c in clients)
