"""Bounded execution with durable retry/ACK recovery; no global scheduler."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from threading import Event, Lock, Thread
from time import time, monotonic
from uuid import uuid4
from datetime import date
from tomte.engine.workers.control import WorkControl, WorkStopped, transient
from tomte.engine.workflows.consolidation import ConsolidationWorkflow


class TomteWorker:
    def __init__(self, idle_timeout_seconds=300, max_workers=1, operation_factory=None,
                 cache=None, usage=None, workflow_factory=ConsolidationWorkflow):
        from tomte.engine import Tomte
        if type(max_workers) is not int or not 1 <= max_workers <= 32:
            raise ValueError('max_workers must be between 1 and 32')
        self.idle_timeout_seconds, self.max_workers = idle_timeout_seconds, max_workers
        self.cache = cache if cache is not None else Tomte.get_cache()
        self.factory = operation_factory or Tomte.operation_context
        self.usage, self.workflow_factory = usage, workflow_factory
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='tomte-memory')
        self.futures, self.mutex = {}, Lock()
        self.poll_lock = Lock()  # Serialize polling, never needed by shutdown.
        self.stopping, self.wake = Event(), Event()
        self.thread = None
        self.last_errors = 0
        self.cooldowns = {}
        self.candidate_offset = 0
        self.maintenance_day = date.today()

    def poll(self):
        """At most N submitted jobs. Never sleeps for retries."""
        if not self.poll_lock.acquire(blocking=False):
            return
        try:
            self._poll()
        finally:
            self.poll_lock.release()

    def _poll(self):
        now = time()
        renewals = []
        with self.mutex:
            for future, work in list(self.futures.items()):
                if future.done():
                    try:
                        if future.result() is False and work['scope'] is not None:
                            self.cooldowns[work['scope']] = now + 5
                    except Exception:
                        self.last_errors += 1
                        if work['scope'] is not None:
                            self.cooldowns[work['scope']] = now + 5
                    del self.futures[future]
                elif work['key'] is not None and now - work['renewed'] >= 30:
                    renewals.append(work)
        for work in renewals:
            try:
                if not self.cache.renew_lock(work['key'], work['token'], 120):
                    work['control'].stopped.set()
            except Exception:
                work['control'].stopped.set()
            work['renewed'] = now
        with self.mutex:
            if self.stopping.is_set() or len(self.futures) >= self.max_workers:
                return
            if date.today() != self.maintenance_day:
                # Drain this process before the daily global dictionary cycle.
                if self.futures:
                    return
                control = WorkControl()
                future = self.executor.submit(self._maintenance, control)
                self.futures[future] = {'key': None, 'control': control, 'scope': None}
                self.maintenance_day = date.today()
                return
            active = {w['scope'] for w in self.futures.values()}
            self.cooldowns = {scope: due for scope, due in self.cooldowns.items() if due > now}
        candidates = self.cache.worker_candidates(now, self.idle_timeout_seconds)
        if candidates:
            offset = self.candidate_offset % len(candidates)
            candidates = candidates[offset:] + candidates[:offset]
            self.candidate_offset = (offset + self.max_workers) % len(candidates)
        for tenant, user, session in candidates:
            if self.stopping.is_set() or len(self.futures) >= self.max_workers:
                break
            if (tenant, user) in active or (tenant, user) in self.cooldowns:
                continue
            state = self.cache.worker_state(tenant, user)
            if state and (state['session'] != session or state.get('due', 0) > now):
                continue
            key, token = f'Tomte:lock:consolidation:{tenant}:{user}', uuid4().hex
            if not self.cache.acquire_lock(key, token, ttl_seconds=120):
                continue
            work = {'key': key, 'token': token, 'control': WorkControl(),
                    'scope': (tenant, user), 'renewed': now}
            admitted = False
            try:
                with self.mutex:
                    # Stop and submit share this tiny, I/O-free critical section.
                    if self.stopping.is_set():
                        return
                    future = self.executor.submit(self._attempt, tenant, user, session, work)
                    self.futures[future] = work
                    admitted = True
                    active.add((tenant, user))
            finally:
                if not admitted:
                    self.cache.release_lock(key, token)

    def _save(self, tenant, user, work, state, delay=0):
        state['due'] = time() + delay
        if not self.cache.worker_save(tenant, user, work['token'], state, state['due']):
            raise WorkStopped('lease_lost')
        if self.usage:
            self.usage.batch_state(tenant, user, state['status'])

    def _finish(self, tenant, user, work, state, raw, outcome):
        state['status'] = 'ACK' if outcome == 'committed' else 'SKIP'
        self._save(tenant, user, work, state)
        if not self.cache.worker_finish(tenant, user, work['token'], state, raw, outcome):
            raise WorkStopped('ack_not_owned')
        if self.usage:
            self.usage.batch_state(tenant, user, None)

    def _failure(self, tenant, user, work, state, raw):
        if state.get('retryable') and state['attempts'] < 3:
            state['status'] = 'WAIT'
            self._save(tenant, user, work, state, (30, 120)[max(0, state['attempts']-1)])
        else:
            self._finish(tenant, user, work, state, raw, 'skipped')

    def _attempt(self, tenant, user, session, work):
        try:
            state = self.cache.worker_state(tenant, user)
            if self.usage:
                self.usage.batch_state(tenant, user, state['status'] if state else None)
            if state and (state['session'] != session or state.get('due', 0) > time()):
                return
            with self.factory(work['control']) as (db, llm):
                with db.user_guard(tenant, user) as acquired:
                    if not acquired:
                        return False
                    items, raw = self.cache.worker_batch(tenant, user, session)
                    invalid = False
                    try:
                        batch = ConsolidationWorkflow._batch_id(tenant, user, items) if items else ''
                    except (ValueError, KeyError, TypeError, AttributeError):
                        batch, invalid = '', True
                    if not batch:
                        batch = sha256(json.dumps(raw).encode()).hexdigest()
                        invalid = bool(raw)
                    if state and state['batch'] != batch:
                        raise WorkStopped('snapshot_changed')
                    if state is None:
                        state = {'session': session, 'batch': batch, 'attempts': 0, 'status': 'NEW'}
                        self._save(tenant, user, work, state)
                    # Read the transaction marker before any model call, including
                    # recovery after an unknown commit or failed Redis ACK.
                    if db.is_consolidation_processed(tenant, user, batch):
                        self._finish(tenant, user, work, state, raw, 'committed')
                        return
                    terminal = self.cache.worker_terminal(tenant, user, batch)
                    if terminal and terminal['outcome'] == 'skipped':
                        state['reason'] = 'previously_skipped'
                        self._finish(tenant, user, work, state, raw, 'skipped')
                        return
                    if state['status'] in {'RUNNING', 'ACK', 'SKIP'}:
                        state['reason'] = state.get('reason', 'interrupted_unclassified')
                        self._finish(tenant, user, work, state, raw, 'skipped')
                        return
                    if state['status'] == 'RECONCILE':
                        self._failure(tenant, user, work, state, raw)
                        return
                    if invalid or not items:
                        state['reason'] = 'invalid_or_empty_batch'
                        self._finish(tenant, user, work, state, raw, 'skipped')
                        return
                    state.update(status='RUNNING', attempts=state['attempts']+1)
                    self._save(tenant, user, work, state)
                    try:
                        work['control'].check()
                        from contextlib import nullcontext
                        scope = self.usage.scope(tenant, user, 'consolidation', sum(len(i.get('content','')) for i in items)) if self.usage else nullcontext()
                        with scope:
                            success = self.workflow_factory(db=db, llm=llm, control=work['control']).run_consolidation_cycle(tenant, user, items)
                            if not success:
                                raise ValueError('batch_not_processed')
                    except Exception as error:
                        from tomte.engine.core.capacity import CapacityExceeded
                        retryable = transient(error)
                        state.update(status='RECONCILE', retryable=retryable,
                                     reason='capacity_exceeded' if isinstance(error, CapacityExceeded) else
                                     'transient_failure' if retryable else 'permanent_failure')
                        self._save(tenant, user, work, state, 5)
                        # A broken connection leaves RECONCILE for a fresh owned
                        # connection; an exception alone never proves rollback.
                        if db.is_consolidation_processed(tenant, user, batch):
                            self._finish(tenant, user, work, state, raw, 'committed')
                        else:
                            self._failure(tenant, user, work, state, raw)
                        return
                    self._finish(tenant, user, work, state, raw, 'committed')
        finally:
            self.cache.release_lock(work['key'], work['token'])

    def start(self):
        try:
            while not self.stopping.is_set() or self.futures:
                try:
                    self.poll()
                except Exception:
                    self.last_errors += 1
                self.wake.wait(1)
                self.wake.clear()
        finally:
            self.executor.shutdown(wait=True)

    def start_in_background(self):
        if self.thread and self.thread.is_alive():
            return self.thread
        if self.stopping.is_set():
            raise RuntimeError('A stopped worker cannot be restarted')
        self.thread = Thread(target=self.start, name='tomte-coordinator', daemon=True)
        self.thread.start()
        return self.thread

    def stop(self, timeout=5):
        deadline = monotonic() + max(0, timeout)
        if not self.mutex.acquire(timeout=max(0, deadline-monotonic())):
            self.stopping.set()
            self.wake.set()
            return False
        try:
            self.stopping.set()
            for work in self.futures.values():
                work['control'].stopped.set()
        finally:
            self.mutex.release()
        self.wake.set()
        if self.thread:
            self.thread.join(max(0, deadline-monotonic()))
            return not self.thread.is_alive()
        # A manually running poll may be blocked in Redis. Never run its I/O here.
        while not self._drained() and monotonic() < deadline:
            self.wake.wait(min(.01, max(0, deadline-monotonic())))
            self.wake.clear()
        complete = self._drained()
        self.executor.shutdown(wait=complete)
        return complete

    def _drained(self):
        with self.mutex:
            return not self.poll_lock.locked() and all(f.done() for f in self.futures)

    def _run_consolidation_job(self):
        self.poll()

    def _run_midnight_job(self):
        return self._maintenance(WorkControl())

    def _maintenance(self, control):
        from contextlib import nullcontext
        from tomte.engine.workflows.dictionary_maintenance import MaintenanceWorkflow
        scope = self.usage.scope(None, None, 'maintenance') if self.usage else nullcontext()
        with scope, self.factory(control) as (db, llm):
            return MaintenanceWorkflow(db, llm).run_midnight_cleanup()
