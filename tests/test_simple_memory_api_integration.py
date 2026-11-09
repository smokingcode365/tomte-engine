"""Opt-in local Redis/PostgreSQL contracts; synthetic namespaces, no model HTTP."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest
from test_postgres_transactions import local_db
from test_managed_integration import real_worker
from test_consolidation_mechanics import mechanics, MESSAGE
from test_unresolved_memories import pending_payload
from memory_fixtures import simple_entity, simple_fact, prepare_write
from tomte.engine import Tomte
from tomte.engine.clients.chat_client import TomteChatClient
from tomte.engine.core.config import settings
from tomte.engine.core.memory import public_entity
from tomte.engine.utils.keys import (
    generate_common_key, generate_consolidation_key, generate_consolidation_batch_key,
)
from tomte.engine.workflows.consolidation import ConsolidationWorkflow


@pytest.fixture
def live_api(real_worker, monkeypatch):
    case = real_worker
    monkeypatch.setattr(settings, 'isolate_chat_sessions', True)
    class NoModel:
        def __getattr__(self, name):
            raise AssertionError('Unexpected foreground model access: ' + name)
    @contextmanager
    def context(control=None, *, cache_only=False):
        db = None if cache_only else case.db.fork()
        try:
            yield db, NoModel()
        finally:
            if db is not None: db.close()
    monkeypatch.setattr(Tomte, 'operation_context', context)
    monkeypatch.setattr(Tomte, 'get_cache', lambda: case.cache)
    client = TomteChatClient(case.t)
    def store(message, *, role='user', user=None, session='s'):
        user = user or case.users[0]
        scope = (case.t, user, session)
        if scope not in case.sessions: case.sessions.append(scope)
        return getattr(client, 'store_' + role + '_message')(user, session, message)
    return SimpleNamespace(case=case, client=client, store=store)


def test_store_and_pending_snapshot_across_queue_processing_and_commit(live_api):
    a = live_api; c = a.case; u = c.users[0]
    first = a.store('I adopted Scout.')
    second = a.store('What breed is Scout?', role='assistant')
    assert str(UUID(first['message_id'])) == first['message_id']
    frozen = c.cache.pop_session_chat_log_4_consolidation(c.t, u, 's')
    assert [m['item_id'] for m in frozen] == [first['message_id'], second['message_id']]
    assert all(m['vector'] is None for m in frozen)
    assert frozen[1]['recent_context'] == [{'role':'user','content':'I adopted Scout.'}]
    third = a.store('Scout is a collie.')
    fourth = a.store('A different session', session='other')
    a.store('Another user', user=c.users[1])
    before = c.cache.get_pending_message_batches(c.t, u)
    result = a.client.get_pending_messages(u)
    assert [m['message_id'] for m in result['messages']] == [
        first['message_id'], second['message_id'], third['message_id'], fourth['message_id']]
    assert [m['status'] for m in result['messages']] == ['processing','processing','queued','queued']
    assert c.cache.get_pending_message_batches(c.t, u) == before
    tail = a.client.get_pending_messages(u, session_id='s', limit=1)
    assert [m['message_id'] for m in tail['messages']] == [third['message_id']]
    assert tail['has_more'] is True
    batch_id = ConsolidationWorkflow._batch_id(c.t, u, frozen)
    with c.db.transaction():
        assert c.db.claim_consolidation_batch(c.t, u, batch_id)
    # PostgreSQL committed but Redis ACK has not happened yet.
    assert c.cache.client.llen(generate_consolidation_batch_key(c.t, u, 's')) == 2
    assert [m['message_id'] for m in a.client.get_pending_messages(u)['messages']] == [
        third['message_id'], fourth['message_id']]
    assert TomteChatClient(c.t + '-other').get_pending_messages(u)['messages'] == []


def test_concurrent_storage_captures_the_preceding_turn(live_api):
    a = live_api; c = a.case; u = c.users[0]
    c.sessions.append((c.t, u, 's'))
    barrier = Barrier(2)
    def store(text):
        barrier.wait(timeout=5)
        return a.client.store_user_message(u, 's', text)
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(store, ['First concurrent message', 'Second concurrent message']))
    queued = c.cache.get_pending_message_batches(c.t, u, 's')[0]['queued']
    assert {m['item_id'] for m in queued} == {r['message_id'] for r in receipts}
    assert queued[0]['recent_context'] == []
    assert queued[1]['recent_context'] == [{'role':'user','content':queued[0]['content']}]
    assert [m['content'] for m in c.cache.get_chat_history(c.t, 's', u)] == [m['content'] for m in queued]


def test_bad_queue_type_does_not_partially_append_history(live_api):
    a = live_api; c = a.case; u = c.users[0]
    queue = generate_consolidation_key(c.t, u, 's')
    c.cache.client.set(queue, 'synthetic wrong type')
    import redis
    with pytest.raises(redis.ResponseError, match='storage key type'):
        a.store('Must not be partially stored')
    assert not c.cache.client.exists(generate_common_key(c.t, 's', u))
    assert c.cache.client.get(queue) == 'synthetic wrong type'


def test_worker_embeds_queued_user_only_and_acknowledges_original_snapshot(live_api):
    a = live_api; c = a.case; u = c.users[0]
    a.store('Synthetic user message')
    a.store('Synthetic assistant reply', role='assistant')
    original = c.worker.factory
    calls = []
    @contextmanager
    def factory(control):
        with original(control) as (db, llm):
            llm.get_vector_dimension = lambda: 1536
            def embed(texts):
                calls.append(list(texts))
                return [[1.] + [0.] * 1535 for _ in texts]
            llm.embed_batch = embed
            yield db, llm
    c.worker.factory = factory
    c.step()
    assert calls == [['Synthetic user message']]
    assert len(c.calls) == 1
    assert a.client.get_pending_messages(u) == {'messages': [], 'has_more': False}
    assert c.cache.worker_state(c.t, u) is None
    assert not c.cache.client.exists(generate_consolidation_batch_key(c.t, u, 's'))


def test_embedding_failure_keeps_pending_until_worker_retry(live_api):
    a = live_api; c = a.case; u = c.users[0]
    receipt = a.store('A message to retry')
    original = c.worker.factory
    failed = []
    @contextmanager
    def factory(control):
        with original(control) as (db, llm):
            llm.get_vector_dimension = lambda: 1536
            def embed(texts):
                if not failed:
                    failed.append(True)
                    raise httpx.ReadTimeout('Synthetic embedding failure')
                return [[1.] + [0.] * 1535 for _ in texts]
            llm.embed_batch = embed
            yield db, llm
    c.worker.factory = factory
    c.step()
    assert c.cache.worker_state(c.t, u)['status'] == 'WAIT'
    assert not c.calls  # Extraction did not run with an invalid vector.
    pending = a.client.get_pending_messages(u)['messages']
    assert [m['message_id'] for m in pending] == [receipt['message_id']]
    frozen = c.cache.get_pending_message_batches(c.t, u, 's')[0]['processing']
    assert frozen[0]['vector'] is None
    assert not c.db.is_consolidation_processed(c.t, u, ConsolidationWorkflow._batch_id(c.t, u, frozen))
    c.due(); c.step()
    assert len(c.calls) == 1 and not a.client.get_pending_messages(u)['messages']


def test_storage_only_message_becomes_durable_memory(mechanics):
    c = mechanics
    receipt = TomteChatClient(c.tenant).store_user_message(c.user, c.session, MESSAGE)
    assert c.llm.calls == []
    frozen = c.cache.pop_session_chat_log_4_consolidation(c.tenant, c.user, c.session)
    assert frozen[0]['item_id'] == receipt['message_id'] and frozen[0]['vector'] is None
    assert ConsolidationWorkflow().run_consolidation_cycle(c.tenant, c.user, frozen)
    rows = c.observer.execute('SELECT payload FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s',
                             (c.tenant, c.user)).fetchall()
    assert len(rows) == 1
    assert rows[0][0]['facts'][0]['value']['detail']['text'] == 'Python'
    assert c.llm.calls[0:2] == ['embed_batch', 'extract']
    assert c.cache.pop_session_chat_log_4_consolidation(c.tenant, c.user, c.session) == frozen


def test_profile_and_frequency_queries_are_scoped_and_read_only(live_api):
    a = live_api; c = a.case; u = c.users[0]
    profiles = [simple_entity('Profile ' + str(i), domain='identity_and_demographics',
                             facts=[simple_fact('detail', str(i))]) for i in range(4)]
    frequent = simple_entity('Frequent', facts=[simple_fact('a', 'A', mentions=3),
                                               simple_fact('b', 'B', mentions=4)])
    runner_up = simple_entity('Runner up', facts=[simple_fact('a', 'A', mentions=6)])
    unresolved = pending_payload()
    unresolved['facts'][0]['_mention_count'] = 100
    entities = profiles + [frequent, runner_up, unresolved]
    assert c.db.commit_graph_updates([prepare_write(c.t, u, e) for e in entities])
    assert c.db.commit_graph_updates([prepare_write(c.t, c.users[1], simple_entity('Other user',
        domain='identity_and_demographics', facts=[simple_fact('a','A',mentions=500)]))])
    before = c.db.get_user_graph_skeleton(c.t, u)
    assert {e['entity_id'] for e in a.client.get_profile_entities(u)} == {e['entity_id'] for e in profiles}
    assert a.client.get_most_mentioned_entities(u, limit=2) == [public_entity(frequent), public_entity(runner_up)]
    assert c.db.get_user_graph_skeleton(c.t, u) == before
    other_tenant = TomteChatClient(c.t + '-other')
    assert other_tenant.get_profile_entities(u) == []
    assert other_tenant.get_most_mentioned_entities(u) == []
