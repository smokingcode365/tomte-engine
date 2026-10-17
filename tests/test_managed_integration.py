"""Scoped synthetic tests on existing PostgreSQL/Redis. No model HTTP."""
from contextlib import contextmanager
from threading import Barrier
from types import SimpleNamespace
from uuid import uuid4
from urllib.parse import urlsplit
import os
import json
import httpx
import pytest
from dotenv import dotenv_values
from test_postgres_transactions import local_db
from tomte.engine.providers.redis_provider import RedisCacheProvider
from tomte.engine.workers.runner import TomteWorker
from tomte.engine.workflows.consolidation import ConsolidationWorkflow
from tomte.engine.utils.keys import generate_session_ref, get_section_key_active_sessions


@pytest.fixture
def real_worker(local_db,monkeypatch):
    if os.environ.get('TOMTE_RUN_REDIS_TESTS')!='1': pytest.skip('Enable local Redis tests')
    url=os.environ.get('REDIS_URL') or dotenv_values('.env').get('REDIS_URL')
    assert urlsplit(url).hostname in ('localhost','127.0.0.1','::1')
    cache=RedisCacheProvider(url); t=local_db.tenant
    users=local_db.users; sessions=[]; connections=[]; calls=[]; action=[None]
    @contextmanager
    def factory(control):
        db=local_db.db.fork(); connections.append(db)
        def junk(text):
            calls.append(text)
            if action[0]: action[0]()
            return True
        try:
            yield db,SimpleNamespace(is_junk=junk)
        finally:
            db.close()
    # Restrict execution to this fixture; never dispatch other users' existing queues.
    monkeypatch.setattr(cache,'worker_candidates',lambda *args:list(sessions))
    worker=TomteWorker(cache=cache,operation_factory=factory,max_workers=2,idle_timeout_seconds=0)
    def enqueue(user,session='s'):
        scope=(t,user,session)
        if scope not in sessions: sessions.append(scope)
        cache.enqueue_for_consolidation(t,user,session,'Synthetic harmless chat','user',[1.]+[0.]*1535)
    def step():
        worker.poll()
        for future in list(worker.futures):
            try: future.result(timeout=10)
            except ConnectionError: pass
    def due():
        for user in users:
            key=cache.worker_state_key(t,user)
            raw=cache.client.get(key)
            if raw:
                state=json.loads(raw); state['due']=0
                cache.client.set(key,json.dumps(state))
        worker.cooldowns.clear()
    try:
        yield SimpleNamespace(db=local_db.db,cache=cache,t=t,users=users,worker=worker,
            enqueue=enqueue,step=step,due=due,connections=connections,calls=calls,action=action,sessions=sessions)
    finally:
        assert worker.stop(10)
        keys=list(cache.client.scan_iter(match=f'Tomte:*{t}*'))
        if keys: cache.client.delete(*keys)
        refs=[generate_session_ref(a,c,b) for a,b,c in sessions]
        if refs:
            cache.client.zrem(get_section_key_active_sessions(),*refs)
            cache.client.zrem('Tomte:worker:due',*refs)
        cache.client.close()


def test_real_two_workers_own_connections_and_commit_markers(real_worker):
    c=real_worker; barrier=Barrier(2)
    c.action[0]=lambda:barrier.wait(timeout=5)
    for user in c.users: c.enqueue(user)
    c.step()
    assert len(c.calls)==2 and len(c.connections)==2
    assert c.connections[0].conn is not c.connections[1].conn
    assert all(db.conn.closed for db in c.connections)
    count=c.db.conn.execute('SELECT count(*) AS n FROM tomte_processed_consolidations WHERE tenant_id=%s',(c.t,)).fetchone()['n']
    assert count==2
    assert all(c.cache.worker_state(c.t,u) is None for u in c.users)


def test_real_retry_skip_and_later_batch(real_worker):
    c=real_worker; u=c.users[0]; c.enqueue(u); c.enqueue(u,'later')
    def fail(): raise httpx.ReadTimeout('synthetic transient')
    c.action[0]=fail
    for attempt in range(1,4):
        c.step(); assert len(c.calls)==attempt
        if attempt==1:
            from tomte.engine.utils.keys import generate_consolidation_batch_key, generate_consolidation_key
            snapshot=c.cache.client.lrange(generate_consolidation_batch_key(c.t,u,'s'),0,-1)
        if attempt<3:
            state=c.cache.worker_state(c.t,u)
            assert state['status']=='WAIT' and state['attempts']==attempt
            c.step(); assert len(c.calls)==attempt
            c.due()
    assert c.cache.worker_state(c.t,u) is None
    terminals=list(c.cache.client.scan_iter(match=f'Tomte:worker:terminal:{c.t}:{u}:*'))
    assert any(json.loads(c.cache.client.get(k))['outcome']=='skipped' for k in terminals)
    assert all(c.cache.client.ttl(k)==-1 for k in terminals)
    c.cache.client.rpush(generate_consolidation_key(c.t,u,'s'),*snapshot)
    c.worker.candidate_offset=0  # Explicitly select the reintroduced snapshot.
    c.step()
    assert len(c.calls)==3
    assert all(json.loads(c.cache.client.get(k))['attempts']==3 for k in terminals)
    # Stop offering the empty completed session, just as the real candidate index does.
    c.sessions.remove((c.t,u,'s'))
    c.action[0]=None; c.step()
    assert len(c.calls)==4 and c.cache.worker_state(c.t,u) is None


def test_real_ack_failure_recovers_without_models(real_worker,monkeypatch):
    c=real_worker; u=c.users[0]; c.enqueue(u)
    original=c.cache.worker_finish; failed=[]
    def finish(*args):
        if not failed:
            failed.append(True); raise ConnectionError('synthetic failed ACK')
        return original(*args)
    monkeypatch.setattr(c.cache,'worker_finish',finish)
    c.step(); assert c.cache.worker_state(c.t,u)['status']=='ACK'
    c.step(); c.due(); c.step()
    assert len(c.calls)==1 and c.cache.worker_state(c.t,u) is None


def test_real_lease_replacement_and_database_user_fence(real_worker):
    c=real_worker; u=c.users[0]; key=f'Tomte:lock:consolidation:{c.t}:{u}'
    assert c.cache.acquire_lock(key,'old',120)
    assert c.cache.renew_lock(key,'old',120)
    c.cache.client.delete(key)  # Only this test-owned lease: simulate expiry.
    assert c.cache.acquire_lock(key,'new',120)
    assert not c.cache.renew_lock(key,'old',120)
    assert not c.cache.release_lock(key,'old')
    assert c.cache.client.get(key)=='new'
    other=c.db.fork()
    try:
        with c.db.user_guard(c.t,u) as first:
            assert first
            with other.user_guard(c.t,u) as same: assert not same
            with other.user_guard(c.t,c.users[1]) as different: assert different
            with other.user_guard(c.t+'-other',u) as tenant: assert tenant
        with other.user_guard(c.t,u) as released: assert released
    finally: other.close()


def test_real_dictionary_maintenance_excludes_feedback(local_db):
    db=local_db.db; other=db.fork()
    import psycopg
    try:
        with db.maintenance_guard() as acquired:
            assert acquired
            with other.maintenance_guard() as second: assert not second
            with pytest.raises(psycopg.errors.LockNotAvailable):
                with other.transaction(): other.dictionary_write_guard()
        with other.transaction(): other.dictionary_write_guard()
    finally: other.close()


def test_real_unknown_commit_reconciles_on_fresh_connection(real_worker,monkeypatch):
    from tomte.engine.providers.postgres_provider import PostgresProvider
    c=real_worker; u=c.users[0]; c.enqueue(u); c.enqueue(u,'later')
    unavailable=[False]
    read=PostgresProvider.is_consolidation_processed
    def marker(self,*args):
        if unavailable[0]: raise ConnectionError('synthetic database unavailable')
        return read(self,*args)
    monkeypatch.setattr(PostgresProvider,'is_consolidation_processed',marker)
    cycle=ConsolidationWorkflow.run_consolidation_cycle
    def uncertain(self,*args):
        result=cycle(self,*args)
        unavailable[0]=True
        raise ConnectionError('synthetic response lost after commit')
    monkeypatch.setattr(ConsolidationWorkflow,'run_consolidation_cycle',uncertain)
    c.step()
    assert c.cache.worker_state(c.t,u)['status']=='RECONCILE'
    c.due(); c.step()
    assert len(c.calls)==1
    unavailable[0]=False
    c.due(); c.step()
    assert len(c.calls)==1 and c.cache.worker_state(c.t,u) is None
    assert all(db.conn.closed for db in c.connections)


def test_real_shutdown_stops_before_commit_and_reports_incomplete(real_worker):
    from threading import Event
    c=real_worker; u=c.users[0]; c.enqueue(u)
    entered,release=Event(),Event()
    def pause():
        entered.set(); assert release.wait(5)
    c.action[0]=pause
    c.worker.poll(); assert entered.wait(3)
    try:
        assert c.worker.stop(.01) is False
    finally:
        release.set()
        assert c.worker.stop(10)
    count=c.db.conn.execute('SELECT count(*) AS n FROM tomte_processed_consolidations WHERE tenant_id=%s',(c.t,)).fetchone()['n']
    assert count==0 and c.cache.worker_state(c.t,u) is None


def test_real_optional_feedback_failure_and_ack_recovery_do_not_repeat_models(real_worker, monkeypatch):
    from tomte.engine.dictionary.service import DictionaryService
    c = real_worker
    user = c.users[0]
    c.enqueue(user)
    def fail_feedback(*args):
        raise ValueError('synthetic optional failure')
    monkeypatch.setattr(DictionaryService, 'apply_feedback', fail_feedback)
    finish = c.cache.worker_finish
    failed = []
    def fail_ack_once(*args):
        if not failed:
            failed.append(True)
            raise ConnectionError('synthetic lost ACK')
        return finish(*args)
    monkeypatch.setattr(c.cache, 'worker_finish', fail_ack_once)
    c.step()
    assert c.cache.worker_state(c.t,user)['status'] == 'ACK'
    c.step()
    c.due()
    c.step()
    assert len(c.calls) == 1
    assert c.cache.worker_state(c.t,user) is None


def test_real_oversized_snapshot_skips_exactly_then_same_user_advances(real_worker):
    from tomte.engine.utils.keys import generate_consolidation_key, generate_consolidation_batch_key
    c = real_worker
    user = c.users[0]
    c.enqueue(user)
    key = generate_consolidation_key(c.t,user,'s')
    item = json.loads(c.cache.client.lindex(key,0))
    item['content'] = 'x'*70000
    c.cache.client.lset(key,0,json.dumps(item))  # This fixture's one owned source only.
    c.enqueue(user,'later')
    c.step()
    assert not c.calls
    terminals = list(c.cache.client.scan_iter(match=f'Tomte:worker:terminal:{c.t}:{user}:*'))
    terminal = json.loads(c.cache.client.get(terminals[0]))
    assert terminal['outcome'] == 'skipped' and terminal['reason'] == 'capacity_exceeded'
    assert terminal['attempts'] == 1
    assert not c.cache.client.exists(generate_consolidation_batch_key(c.t,user,'s'))
    assert c.cache.client.llen(generate_consolidation_key(c.t,user,'later')) == 1
    c.sessions.remove((c.t,user,'s'))
    c.step()
    assert len(c.calls) == 1 and c.cache.worker_state(c.t,user) is None
