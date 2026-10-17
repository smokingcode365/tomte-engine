"""Deterministic scheduler/failure tests; no model or service requests."""
from contextlib import contextmanager
from copy import deepcopy
from threading import Event, Lock
from types import SimpleNamespace
from uuid import uuid4
import json
import pytest
import httpx
from tomte.engine.workers.runner import TomteWorker
from tomte.engine.workflows.consolidation import ConsolidationWorkflow


class MemoryCache:
    def __init__(self):
        self.states, self.locks, self.queues, self.frozen, self.terminal = {}, {}, {}, {}, []
        self.acquired, self.released = [], []
        self.fail_ack = False

    def add(self, user='u', session='s', tenant='t'):
        key = (tenant, user, session)
        self.queues.setdefault(key, []).append({'item_id': str(uuid4()), 'session_id': session,
            'tenant_id': tenant, 'user_id': user, 'role': 'user', 'timestamp': 1,
            'content': 'Synthetic preference', 'vector': [1.]})

    def worker_candidates(self, *args):
        return list(dict.fromkeys([*self.frozen, *self.queues]))

    def worker_state(self, t, u):
        return deepcopy(self.states.get((t,u)))

    def worker_terminal(self,t,u,b):
        return next(({'outcome':outcome} for a,c,state,outcome in self.terminal
                     if (a,c,state['batch'])==(t,u,b)),None)

    def acquire_lock(self, key, token, ttl_seconds):
        if key in self.locks:
            return False
        self.locks[key] = token
        self.acquired.append((key, token))
        return True

    def renew_lock(self, key, token, ttl):
        return self.locks.get(key) == token

    def release_lock(self, key, token):
        self.released.append((key,token))
        if self.locks.get(key) == token:
            del self.locks[key]

    def worker_save(self, t,u,token,state,due):
        if self.locks.get(f'Tomte:lock:consolidation:{t}:{u}') != token:
            return False
        self.states[(t,u)] = deepcopy(state)
        return True

    def worker_batch(self,t,u,s):
        key=(t,u,s)
        if key not in self.frozen:
            self.frozen[key] = self.queues.pop(key, [])
        return deepcopy(self.frozen[key]), [json.dumps(i) for i in self.frozen[key]]

    def worker_finish(self,t,u,token,state,raw,outcome):
        if self.fail_ack:
            self.fail_ack=False
            raise ConnectionError('synthetic ACK failure')
        if self.locks.get(f'Tomte:lock:consolidation:{t}:{u}') != token:
            return False
        self.terminal.append((t,u,deepcopy(state),outcome))
        self.states.pop((t,u),None)
        self.frozen.pop((t,u,state['session']),None)
        return True


class Harness:
    def __init__(self, behavior=None, workers=1):
        self.cache, self.markers, self.calls, self.connections = MemoryCache(), set(), [], []
        self.behavior = behavior
        self.db_unavailable = False
        self.worker = TomteWorker(idle_timeout_seconds=0, max_workers=workers,
            cache=self.cache, operation_factory=self.factory, workflow_factory=self.workflow)

    @contextmanager
    def factory(self, control):
        db=SimpleNamespace(user_guard=lambda *args:self.guard(), is_consolidation_processed=self.processed)
        self.connections.append(db)
        yield db, SimpleNamespace()

    @contextmanager
    def guard(self):
        yield True

    def processed(self,t,u,b):
        if self.db_unavailable:
            raise ConnectionError('unknown DB')
        return (t,u,b) in self.markers

    def workflow(self, **kwargs):
        def run(t,u,items):
            self.calls.append((t,u,items[0]['session_id']))
            if self.behavior:
                self.behavior(self,t,u,items)
            self.markers.add((t,u,ConsolidationWorkflow._batch_id(t,u,items)))
            return True
        return SimpleNamespace(run_consolidation_cycle=run)

    def step(self):
        self.worker.poll()
        for future in list(self.worker.futures):
            try:
                future.result(timeout=5)
            except ConnectionError:
                pass

    def due(self):
        for state in self.cache.states.values():
            state['due']=0


@pytest.fixture
def harness():
    h=Harness()
    yield h
    assert h.worker.stop()


def test_ack_recovery_does_not_repeat_model_work(harness):
    h=harness; h.cache.add(); h.cache.fail_ack=True
    h.step()
    assert h.cache.states[('t','u')]['status']=='ACK'
    h.step()
    h.worker.cooldowns.clear()
    h.step()
    assert len(h.calls)==1 and h.cache.terminal[-1][3]=='committed'
    assert h.cache.acquired[0][1]!=h.cache.acquired[1][1]
    assert h.cache.acquired==h.cache.released
    assert h.connections[0] is not h.connections[1]


def test_three_attempts_then_skip_and_next_session_advances(harness):
    h=harness; h.cache.add(); h.cache.add(session='later')
    def fail(*args): raise httpx.ReadTimeout('temporary')
    h.behavior=fail
    for attempt in range(1,4):
        h.step()
        assert len(h.calls)==attempt
        if attempt<3:
            state=h.cache.states[('t','u')]
            assert state['status']=='WAIT' and state['attempts']==attempt
            h.step(); assert len(h.calls)==attempt
            h.due()
    assert h.cache.terminal[-1][3]=='skipped'
    h.behavior=None; h.step()
    assert h.calls[-1]==('t','u','later')


def test_permanent_failure_skips_without_retry(harness):
    h=harness; h.cache.add()
    def fail(*args): raise ValueError('invalid output')
    h.behavior=fail; h.step(); h.step()
    assert len(h.calls)==1 and h.cache.terminal[-1][3]=='skipped'


def test_skipped_snapshot_reintroduced_is_not_replayed(harness):
    h=harness; h.cache.add()
    saved=deepcopy(h.cache.queues[('t','u','s')])
    def fail(*args): raise ValueError('invalid output')
    h.behavior=fail; h.step()
    h.behavior=None
    h.cache.queues[('t','u','s')]=saved
    h.step()
    assert len(h.calls)==1 and h.cache.terminal[-1][2]['reason']=='previously_skipped'


def test_retrying_user_does_not_block_another_user(harness):
    h=harness; h.cache.add(); h.cache.add('other')
    def selective_failure(h,t,u,items):
        if u=='u': raise httpx.ReadTimeout('temporary')
    h.behavior=selective_failure; h.step()
    assert h.cache.states[('t','u')]['status']=='WAIT'
    h.step()
    assert h.calls==[('t','u','s'),('t','other','s')]


def test_busy_database_owners_cannot_monopolize_submission_slots(harness):
    h=harness
    for user in ['busy0','busy1','busy2','busy3','busy4','healthy']:
        h.cache.add(user)
    @contextmanager
    def guard(t,u):
        yield u=='healthy'
    @contextmanager
    def factory(control):
        yield SimpleNamespace(user_guard=guard,is_consolidation_processed=h.processed),SimpleNamespace()
    h.worker.factory=factory
    for _ in range(6): h.step()
    assert h.calls==[('t','healthy','s')]


def test_unknown_commit_waits_for_database_then_ack_without_replay(harness):
    h=harness; h.cache.add(); h.cache.add(session='later')
    def uncertain(h,t,u,items):
        h.markers.add((t,u,ConsolidationWorkflow._batch_id(t,u,items)))
        h.db_unavailable=True
        raise httpx.ReadTimeout('commit response lost')
    h.behavior=uncertain; h.step()
    assert h.cache.states[('t','u')]['status']=='RECONCILE'
    h.due(); h.step(); assert len(h.calls)==1
    h.db_unavailable=False; h.behavior=None; h.worker.cooldowns.clear(); h.step()
    assert len(h.calls)==1 and h.cache.terminal[-1][3]=='committed'


def test_two_users_overlap_and_submitted_work_is_bounded():
    entered=[Event(),Event()]; release=Event(); lock=Lock(); count=[]
    def block(h,t,u,items):
        with lock:
            count.append(u); index=len(count)-1
        entered[index].set()
        assert release.wait(5)
    h=Harness(block,2)
    try:
        h.cache.add('a'); h.cache.add('a','other'); h.cache.add('b'); h.cache.add('c')
        h.worker.poll()
        assert all(event.wait(3) for event in entered)
        assert len(h.worker.futures)==2 and set(count)=={'a','b'}
        assert h.worker.stop(timeout=.01) is False
    finally:
        release.set()
        assert h.worker.stop(timeout=5)


def test_lease_loss_stops_later_stages(harness):
    h=harness; h.cache.add()
    entered,release=Event(),Event()
    def block(*args): entered.set(); release.wait(3)
    h.behavior=block; h.worker.poll(); assert entered.wait(2)
    work=next(iter(h.worker.futures.values()))
    h.cache.locks[work['key']]='successor'; work['renewed']=0
    h.worker.poll()
    assert work['control'].stopped.is_set()
    release.set()
    for future in h.worker.futures:
        with pytest.raises(Exception): future.result(timeout=3)
    assert h.cache.locks[work['key']]=='successor'
