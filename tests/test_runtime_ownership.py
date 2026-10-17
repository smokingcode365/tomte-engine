from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from tomte.engine import Tomte
from tomte.engine.dictionary.service import DictionaryService, InternalMergeResponse
from pydantic import ValidationError


@pytest.mark.parametrize('groups',[
    [{'words':['a','a'],'confidence_score':1}],
    [{'words':['a','b'],'confidence_score':1},{'words':['b','c'],'confidence_score':1}],
    [{'words':['a','b'],'confidence_score':float('nan')}],
    [{'words':['a','b'],'confidence_score':1.01}],
])
def test_invalid_dictionary_groups_are_rejected(groups):
    with pytest.raises(ValidationError): InternalMergeResponse(groups=groups)


def test_unoffered_dictionary_word_aborts_before_write():
    db=Mock()
    llm=Mock()
    llm.parse_structured.return_value=InternalMergeResponse(groups=[{'words':['a','unoffered'],'confidence_score':1}])
    service=DictionaryService(db,llm)
    with pytest.raises(ValueError,match='Unoffered'):
        service._analyze_internal_synonyms([{'keyword':'a','keyword_type':'ATTRIBUTE','hits':1},
                                           {'keyword':'b','keyword_type':'ATTRIBUTE','hits':2}])
    assert db.mock_calls==[]


def test_reinitialization_rejects_different_dependencies_and_shutdown_waits(monkeypatch):
    for name in ('_db','_cache','_llm','_worker','_usage','_operation_factory'):
        monkeypatch.setattr(Tomte,name,None)
    monkeypatch.setattr(Tomte,'_closing',False)
    monkeypatch.setattr(Tomte,'_active_operations',0)
    closed=[]
    @contextmanager
    def factory(control):
        yield SimpleNamespace(),SimpleNamespace()
        closed.append(True)
    db,cache,llm=SimpleNamespace(),SimpleNamespace(),SimpleNamespace()
    Tomte.init(db,cache,llm,start_daemon_worker=False,operation_factory=factory)
    worker=Tomte.get_worker()
    Tomte.init(db,cache,llm,start_daemon_worker=False,operation_factory=factory)
    assert Tomte.get_worker() is worker
    with pytest.raises(RuntimeError):
        Tomte.init(SimpleNamespace(),cache,llm,start_daemon_worker=False)
    with Tomte.operation_context():
        assert Tomte.shutdown(0) is False
    assert closed==[True] and Tomte.shutdown()


def test_native_operation_owns_distinct_sdk_clients_loops_and_db(monkeypatch):
    import asyncio
    from tomte.engine.providers.postgres_provider import PostgresProvider
    from tomte.engine.providers.openai_provider import OpenAILLMProvider
    from tomte.engine.runtime import owned_operation
    db=object.__new__(PostgresProvider)
    owned=[]
    def fork():
        child=SimpleNamespace(close=Mock()); owned.append(child); return child
    monkeypatch.setattr(db,'fork',fork)
    monkeypatch.setenv('OPENAI_API_KEY','synthetic-not-a-key')
    llm=object.__new__(OpenAILLMProvider)
    clients=[]; loops=[]
    for _ in range(2):
        with owned_operation(db,llm) as (child,controlled):
            clients.append(controlled.provider.strong_model.client)
            loops.append(asyncio.get_event_loop())
            assert child is owned[-1]
            assert controlled.provider.agent_retries==0
    assert clients[0] is not clients[1] and all(c.is_closed() for c in clients)
    assert loops[0] is not loops[1] and all(loop.is_closed() for loop in loops)
    assert all(child.close.call_count==1 for child in owned)


def test_late_shutdown_cannot_clear_reinitialized_runtime(monkeypatch):
    # Simulate another shutdown completing/reinitializing while this caller drains.
    newer=SimpleNamespace()
    def stop(timeout):
        Tomte._worker=newer
        return True
    old=SimpleNamespace(stop=stop)
    monkeypatch.setattr(Tomte,'_worker',old)
    monkeypatch.setattr(Tomte,'_usage',None)
    monkeypatch.setattr(Tomte,'_active_operations',0)
    monkeypatch.setattr(Tomte,'_closing',False)
    assert Tomte.shutdown() is False
    assert Tomte._worker is newer


def test_inspecting_legacy_workflow_does_not_capture_global_resources(monkeypatch):
    from tomte.engine.clients import chat_client
    monkeypatch.setattr(chat_client,'ChatHookWorkflow',Mock(return_value=object()))
    client=chat_client.TomteChatClient('synthetic')
    assert client.workflow is not None
    assert client._workflow is None
