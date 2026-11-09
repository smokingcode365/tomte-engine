"""Owned local Redis/SQL fixtures, deterministic vectors, no provider HTTP."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event, Lock
from types import SimpleNamespace
import json
import pytest

from test_postgres_transactions import local_db
from test_managed_integration import real_worker
from tomte.engine import Tomte
from tomte.engine.clients.chat_client import TomteChatClient
from tomte.engine.core.config import settings
from tomte.engine.providers.redis_pending import keys
from tomte.engine.workflows.pending_search import PendingMemorySearch
from tomte.engine.workflows.consolidation import ConsolidationWorkflow


@pytest.fixture
def search_case(real_worker, monkeypatch):
    c = real_worker
    docs, queries = [], []
    mutex = Lock()
    class Model:
        identity = 'synthetic-embedding-v1'
        action = None
        def get_vector_dimension(self): return 1536
        def get_embedding_cache_identity(self): return self.identity
        def embed_text(self, text):
            queries.append(text)
            return [1.] + [0.] * 1535
        def embed_batch(self, texts):
            with mutex: docs.extend(texts)
            if self.action: self.action()
            return [[1.] + [0.] * 1535 for text in texts]
    model = Model()
    @contextmanager
    def context(control=None, *, cache_only=False):
        db = None if cache_only else c.db.fork()
        try: yield db, model
        finally:
            if db is not None: db.close()
    monkeypatch.setattr(Tomte,'operation_context',context)
    monkeypatch.setattr(Tomte,'get_cache',lambda:c.cache)
    monkeypatch.setattr(Tomte,'_usage',None)
    client = TomteChatClient(c.t)
    def enqueue(text, session='s', user=None, role='user', recent_context=None):
        user = user or c.users[0]
        scope = (c.t,user,session)
        if scope not in c.sessions: c.sessions.append(scope)
        c.cache.enqueue_for_consolidation(*scope,text,role,[1.]+[0.]*1535,recent_context)
    def retrieve(user=None):
        return client.get_relevant_memories(user or c.users[0], 'Synthetic question', include_pending=True)
    return SimpleNamespace(c=c,model=model,docs=docs,queries=queries,enqueue=enqueue,retrieve=retrieve,client=client)


def test_combined_retrieval_reuses_windows_and_reads_beyond_twenty_messages(search_case):
    a=search_case; c=a.c; u=c.users[0]
    for i in range(30): a.enqueue(f'Owned synthetic message {i}')
    result=a.retrieve()
    assert result['entities']==[] and result['unresolved_memories']==[]
    assert result['pending_search']['complete'] is True
    assert result['pending_search']['messages_considered']==30
    assert any('Owned synthetic message 0' in text for text in a.docs)
    assert len(a.docs)==7
    assert 'PENDING CONVERSATION' in result['context_text']
    before=list(a.docs)
    second=a.retrieve()
    assert a.docs==before and len(a.queries)==2
    assert second['pending_message_windows']==result['pending_message_windows']
    mids=[m['message_id'] for w in second['pending_message_windows'] for m in w['messages']]
    assert len(mids)==len(set(mids))
    assert c.cache.client.hlen(keys(c.t,u,'s')[0])==7


def test_concurrent_calls_share_a_single_window_embedding(search_case,monkeypatch):
    a=search_case
    a.enqueue('Concurrent source')
    entered, busy, release=Event(),Event(),Event()
    def wait():
        entered.set()
        assert release.wait(5)
    a.model.action=wait
    claim=a.c.cache.claim_pending_window
    def tracked(*args):
        result=claim(*args)
        if result['state']=='busy': busy.set()
        return result
    monkeypatch.setattr(a.c.cache,'claim_pending_window',tracked)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(a.retrieve)
        assert entered.wait(5)
        second=pool.submit(a.retrieve)
        try: assert busy.wait(5)
        finally: release.set()
        assert first.result(10)['pending_search']['complete']
        assert second.result(10)['pending_search']['complete']
    assert len(a.docs)==1


def test_manual_ack_removes_affected_windows_and_preserves_later_ones(search_case):
    a=search_case; c=a.c; u=c.users[0]
    for i in range(6): a.enqueue(f'Message {i}')
    frozen=c.cache.pop_session_chat_log_4_consolidation(c.t,u,'s')
    for i in range(6,14): a.enqueue(f'Message {i}')
    a.retrieve()
    catalog,vectors,epoch=keys(c.t,u,'s')
    assert c.cache.client.hlen(vectors)==3
    with c.db.transaction():
        c.db.claim_consolidation_batch(c.t,u,ConsolidationWorkflow._batch_id(c.t,u,frozen))
    assert c.cache.clear_session_4_consolidation(c.t,u,'s',frozen)
    assert c.cache.client.hlen(vectors)==c.cache.client.hlen(catalog)==1
    assert c.cache.client.get(epoch) is None
    a.docs.clear()
    result=a.retrieve()
    assert len(a.docs)==1
    assert result['pending_search']['messages_considered']==8
    assert result['pending_search']['complete']
    old={m['item_id'] for m in frozen}
    assert not old & {m['message_id'] for w in result['pending_message_windows'] for m in w['messages']}


def test_worker_ack_cleans_the_same_cache(search_case):
    a=search_case; c=a.c; u=c.users[0]
    a.enqueue('Worker completion')
    a.retrieve()
    assert c.cache.client.hlen(keys(c.t,u,'s')[1])==1
    c.step()
    assert c.cache.client.hlen(keys(c.t,u,'s')[0])==0
    assert c.cache.client.hlen(keys(c.t,u,'s')[1])==0
    assert a.retrieve()['pending_message_windows']==[]


def test_completion_during_embedding_cannot_resurrect_cache(search_case):
    a=search_case; c=a.c; u=c.users[0]
    a.enqueue('Captured before completion')
    frozen=c.cache.pop_session_chat_log_4_consolidation(c.t,u,'s')
    def complete():
        with c.db.transaction():
            c.db.claim_consolidation_batch(c.t,u,ConsolidationWorkflow._batch_id(c.t,u,frozen))
        assert c.cache.clear_session_4_consolidation(c.t,u,'s',frozen)
    a.model.action=complete
    result=a.retrieve()
    assert result['pending_message_windows']  # This request retains its captured source.
    assert c.cache.client.hlen(keys(c.t,u,'s')[0])==0
    assert c.cache.client.hlen(keys(c.t,u,'s')[1])==0
    a.model.action=None
    assert a.retrieve()['pending_message_windows']==[]


def test_new_tail_and_embedding_model_use_distinct_cache_entries(search_case):
    a=search_case
    for i in range(8): a.enqueue(f'Message {i}')
    a.retrieve(); assert len(a.docs)==2
    a.enqueue('Message 8'); a.docs.clear()
    a.retrieve(); assert len(a.docs)==1
    a.model.identity='synthetic-embedding-v2'; a.docs.clear()
    a.retrieve(); assert len(a.docs)==2


def test_queue_limits_and_embedding_budget_are_explicit(search_case,monkeypatch):
    a=search_case
    for i in range(30): a.enqueue('Large synthetic message '+str(i)+'x'*200)
    monkeypatch.setattr(settings.pending_search,'max_messages',12)
    monkeypatch.setattr(settings.pending_search,'max_embedding_bytes',512)
    result=a.retrieve()
    assert not result['pending_search']['complete']
    assert result['pending_search']['messages_considered']==12
    assert 'embedding_budget' in result['pending_search']['reasons']
    assert 'snapshot_limit_or_invalid_input' in result['pending_search']['reasons']
    assert not a.docs


def test_sessions_and_users_are_never_combined_in_windows(search_case):
    a=search_case; c=a.c
    a.enqueue('Session one')
    a.enqueue('Session two',session='second')
    a.enqueue('Other user',user=c.users[1])
    result=a.retrieve()
    assert {w['session_id'] for w in result['pending_message_windows']}=={'s','second'}
    assert 'Other user' not in result['context_text']
    assert len(a.docs)==2
    a.retrieve(c.users[1]); assert len(a.docs)==3
    assert TomteChatClient(c.t+'-other').get_relevant_memories(c.users[0],'text',include_pending=True)['pending_message_windows']==[]


def test_expired_owner_cannot_publish_or_remove_replacement(search_case):
    a=search_case; c=a.c; u=c.users[0]
    a.enqueue('Lease test')
    captured=PendingMemorySearch(c.t,u,c.db,c.cache)
    w=captured.windows[0]; generation=w['generation']
    descriptor={k:w[k] for k in ('session_id','parts','closed')}
    args=(c.t,u,'s',generation,'synthetic-key',descriptor)
    assert c.cache.claim_pending_window(*args,'first')['state']=='owned'
    catalog,vectors,_=keys(c.t,u,'s')
    entry=json.loads(c.cache.client.hget(catalog,'synthetic-key'))
    entry['lease_until']=0  # Only this test's lease; simulate expiry without sleeping.
    c.cache.client.hset(catalog,'synthetic-key',json.dumps(entry))
    assert c.cache.claim_pending_window(*args,'second')['state']=='owned'
    assert not c.cache.publish_pending_window(c.t,u,'s',generation,'synthetic-key','first',[1.,0.])
    assert not c.cache.release_pending_window(c.t,u,'s','synthetic-key','first')
    assert c.cache.publish_pending_window(c.t,u,'s',generation,'synthetic-key','second',[1.,0.])
    assert c.cache.client.hlen(vectors)==1


def test_committed_pending_is_hidden_and_old_snapshot_cannot_register(search_case):
    a=search_case; c=a.c; u=c.users[0]
    a.enqueue('Commit before ACK')
    frozen=c.cache.pop_session_chat_log_4_consolidation(c.t,u,'s')
    captured=PendingMemorySearch(c.t,u,c.db,c.cache)
    with c.db.transaction():
        c.db.claim_consolidation_batch(c.t,u,ConsolidationWorkflow._batch_id(c.t,u,frozen))
    assert a.retrieve()['pending_message_windows']==[]
    assert c.cache.clear_session_4_consolidation(c.t,u,'s',frozen)
    assert captured.search([1.]+[0.]*1535,a.model)['pending_message_windows']
    assert c.cache.client.hlen(keys(c.t,u,'s')[0])==0
    assert c.cache.client.hlen(keys(c.t,u,'s')[1])==0


def test_failed_embedding_releases_claim_and_retry_succeeds(search_case):
    a=search_case; c=a.c; u=c.users[0]
    a.enqueue('Retry test')
    def fail(): raise RuntimeError('Synthetic provider failure')
    a.model.action=fail
    with pytest.raises(RuntimeError,match='Synthetic provider failure'): a.retrieve()
    assert c.cache.client.hlen(keys(c.t,u,'s')[0])==0
    a.model.action=None
    assert a.retrieve()['pending_search']['complete']


def test_worker_retry_retains_cache_and_terminal_skip_removes_it(search_case):
    import httpx
    a=search_case; c=a.c; u=c.users[0]
    a.enqueue('Retry then skip'); a.retrieve()
    def fail(): raise httpx.ReadTimeout('Synthetic transient')
    c.action[0]=fail
    c.step()
    assert c.cache.worker_state(c.t,u)['status']=='WAIT'
    assert c.cache.client.hlen(keys(c.t,u,'s')[1])==1
    c.due(); c.step(); c.due(); c.step()
    assert c.cache.worker_state(c.t,u) is None
    assert c.cache.client.hlen(keys(c.t,u,'s')[1])==0
    assert a.retrieve()['pending_message_windows']==[]


def test_cache_eviction_and_expiry_do_not_remove_pending_source(search_case,monkeypatch):
    a=search_case; c=a.c; u=c.users[0]
    monkeypatch.setattr(settings.pending_search,'cache_windows',1)
    for i in range(14): a.enqueue(f'Cache capacity message {i}')
    result=a.retrieve()
    assert result['pending_search']['complete'] and len(a.docs)==3
    catalog,vectors,_=keys(c.t,u,'s')
    assert c.cache.client.hlen(catalog)==c.cache.client.hlen(vectors)==1
    key,raw=next(iter(c.cache.client.hgetall(catalog).items()))
    entry=json.loads(raw); entry['expires_at']=0
    c.cache.client.hset(catalog,key,json.dumps(entry))
    captured=PendingMemorySearch(c.t,u,c.db,c.cache)
    assert captured.messages==14 and c.cache.client.hlen(vectors)==0
    assert captured.search([1.]+[0.]*1535,a.model)['pending_search']['complete']


def test_malformed_optional_cache_cannot_block_queue_ack(search_case):
    a=search_case; c=a.c; u=c.users[0]
    a.enqueue('Cache damage'); a.retrieve()
    frozen=c.cache.pop_session_chat_log_4_consolidation(c.t,u,'s')
    catalog,vectors,_=keys(c.t,u,'s')
    c.cache.client.hset(catalog,'broken','{"parts": false}')
    c.cache.client.delete(vectors)
    c.cache.client.set(vectors,'wrong derived key type')
    assert c.cache.clear_session_4_consolidation(c.t,u,'s',frozen)
    assert a.retrieve()['pending_message_windows']==[]
