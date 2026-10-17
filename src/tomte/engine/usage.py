"""In-process SDK observations. Estimates are informational, never spending gates."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
from queue import Queue, Full, Empty
from threading import Lock, Thread, Event
from time import monotonic
from uuid import uuid4

_scope = ContextVar('tomte_usage_scope', default=None)
_embedding = ContextVar('tomte_embedding_result', default=None)


def embedding_result(result):
    target = _embedding.get()
    if target is not None:
        target.append(result)


def _observation(response):
    usage = getattr(response, 'usage', None)
    values = {name: getattr(usage, name, None) for name in
              ('input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens')}
    values = {name: value if type(value) is int and value >= 0 else None for name, value in values.items()}
    # SDKs commonly default missing usage to zero. Treat an all-zero record as
    # unknown unless the SDK explicitly supplies other positive usage details.
    if not any(isinstance(v, int) and v > 0 for v in values.values()):
        values = dict.fromkeys(values)
    # The installed OpenAI text adapter drops Responses cache-write metadata;
    # RequestUsage then supplies a default zero. At this public boundary zero
    # cannot be distinguished from missing data. Keep positive SDK observations,
    # but do not advertise an unobserved zero (or a price based on it) as complete.
    # EmbeddingResult is separate: embeddings do not use text prompt caching.
    from pydantic_ai.messages import ModelResponse
    if (isinstance(response, ModelResponse) and response.provider_name == 'openai'
            and values['cache_write_tokens'] == 0):
        values['cache_write_tokens'] = None
    cost = None
    try:
        price = response.cost()
        if price is not None and all(value is not None for value in values.values()):
            cost = float(price.total_price)
    except Exception:
        pass  # Unknown SDK pricing is not free usage.
    return {'model': getattr(response, 'model_name', None), **values,
            'usage_complete': all(value is not None for value in values.values()),
            'estimated_dollars': cost}


def track_provider(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        scope = _scope.get()
        texts = args[0] if args else kwargs.get('texts')
        if scope is None or (method.__name__ == 'embed_batch' and not texts):
            return method(self, *args, **kwargs)
        from pydantic_ai import capture_run_messages
        from pydantic_ai.messages import ModelResponse
        embedded, started, failed = [], monotonic(), False
        token = _embedding.set(embedded)
        with capture_run_messages() as messages:
            owned = not messages
            scope['reporter'].request_started(scope)
            try:
                return method(self, *args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                try:
                    responses = embedded or ([m for m in messages if isinstance(m, ModelResponse)] if owned else [])
                    observations = [_observation(r) for r in responses] or [_observation(None)]
                    for observation in observations:
                        event = {'id': uuid4().hex, 'operation': method.__name__,
                                 'outcome': 'failure' if failed else 'success',
                                 'response_observed': bool(responses),
                                 'duration_ms': (monotonic()-started)*1000, **observation}
                        scope['events'].append(event)
                        scope['reporter'].record(scope, event)
                except Exception:
                    scope['reporter'].observation_errors += 1
                finally:
                    scope['reporter'].request_finished(scope)
                    _embedding.reset(token)
    return run


class UsageReporter:
    def __init__(self, listener=None):
        self.lock = Lock()
        self.totals, self.profiles = {}, {}
        self.batch_states, self.in_flight = {}, {}
        self.listener = listener
        self.notifications = Queue(maxsize=256)
        self.stopping = Event()
        self.listener_errors = self.dropped_notifications = self.observation_errors = 0
        self.thread = None
        if listener is not None:
            self.thread = Thread(target=self._dispatch, name='tomte-usage', daemon=True)
            self.thread.start()

    def _dispatch(self):
        while not self.stopping.is_set() or not self.notifications.empty():
            try:
                event = self.notifications.get(timeout=.1)
            except Empty:
                continue
            try:
                self.listener(event)
            except Exception:
                self.listener_errors += 1

    def close(self, timeout=1):
        self.stopping.set()
        if self.thread:
            self.thread.join(timeout)
        return not self.thread or not self.thread.is_alive()

    def batch_state(self, tenant, user, status):
        """Last locally observed durable state; no queue scan or billing inference."""
        with self.lock:
            if status in {'RUNNING', 'RECONCILE', 'ACK'}:
                self.batch_states[(tenant, user)] = status
            else:
                self.batch_states.pop((tenant, user), None)

    def request_started(self, scope):
        key = (scope['tenant_id'], scope['user_id'])
        with self.lock:
            self.in_flight[key] = self.in_flight.get(key, 0) + 1

    def request_finished(self, scope):
        key = (scope['tenant_id'], scope['user_id'])
        with self.lock:
            remaining = self.in_flight[key] - 1
            if remaining:
                self.in_flight[key] = remaining
            else:
                del self.in_flight[key]

    @contextmanager
    def scope(self, tenant_id, user_id, kind, input_chars=0):
        scope = {'reporter': self, 'tenant_id': tenant_id, 'user_id': user_id,
                 'kind': kind, 'events': []}
        token = _scope.set(scope)
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            _scope.reset(token)
            events = scope['events']
            if (succeeded and kind == 'consolidation' and input_chars > 0 and events
                    and all(e['input_tokens'] is not None and e['outcome'] == 'success' for e in events)):
                key = (tenant_id, user_id)
                with self.lock:
                    p = self.profiles.setdefault(key, {'samples': 0, 'chars': 0, 'input_tokens': 0,
                                                       'output_tokens': 0, 'estimated_dollars': 0})
                    p['samples'] += 1
                    p['chars'] += input_chars
                    for field in ('input_tokens', 'output_tokens', 'estimated_dollars'):
                        if p[field] is None or any(e[field] is None for e in events):
                            p[field] = None
                        else:
                            p[field] += sum(e[field] for e in events)

    def record(self, scope, event):
        key = (scope['tenant_id'], scope['user_id'], event['model'], scope['kind'])
        with self.lock:
            totals = self.totals.setdefault(key, {'observations': 0, 'unknown_usage': 0,
                'unknown_pricing': 0, 'observed_responses': 0, 'input_tokens': 0,
                'output_tokens': 0, 'cache_read_tokens': 0, 'cache_write_tokens': 0, 'known_estimated_dollars': 0})
            totals['observations'] += 1
            totals['observed_responses'] += int(event['response_observed'])
            fields = ('input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens')
            totals['unknown_usage'] += int(any(event[field] is None for field in fields))
            totals['unknown_pricing'] += int(event['estimated_dollars'] is None)
            for field in fields:
                unknown = 'unknown_' + field
                totals[unknown] = totals.get(unknown, 0) + int(event[field] is None)
                totals[field] += event[field] or 0
            totals['known_estimated_dollars'] += event['estimated_dollars'] or 0
        if self.listener:
            try:
                self.notifications.put_nowait({**event, 'tenant_id': scope['tenant_id'],
                                               'user_id': scope['user_id'], 'kind': scope['kind']})
            except Full:
                self.dropped_notifications += 1

    def query(self, tenant_id=None, user_id=None, pending_chars=None):
        with self.lock:
            def matches(key):
                return ((tenant_id is None or key[0] == tenant_id) and
                        (user_id is None or key[1] == user_id))
            states = [s for key, s in self.batch_states.items() if matches(key)]
            running = states.count('RUNNING')
            reconciling = sum(s in {'RECONCILE', 'ACK'} for s in states)
            requests = sum(n for key, n in self.in_flight.items() if matches(key))
            rows = []
            for (tenant, user, model, kind), value in self.totals.items():
                if (tenant_id is None or tenant == tenant_id) and (user_id is None or user == user_id):
                    row = deepcopy(value)
                    for field in ('input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens'):
                        row['known_' + field] = row[field]
                        if row['unknown_' + field]:
                            row[field] = None
                    row['estimated_dollars'] = None if row['unknown_pricing'] else row['known_estimated_dollars']
                    row['usage_complete'] = not row['unknown_usage']
                    rows.append({'tenant_id': tenant, 'user_id': user, 'model': model, 'kind': kind, **row})
            profile = self.profiles.get((tenant_id, user_id))
            pending = None
            if pending_chars == 0:
                pending = dict.fromkeys(('input_tokens', 'output_tokens', 'estimated_dollars'), 0)
            elif pending_chars is not None and profile:
                pending = {f: None if profile[f] is None else profile[f] * pending_chars / profile['chars']
                           for f in ('input_tokens', 'output_tokens', 'estimated_dollars')}
            return {'totals': rows, 'pending_estimate': pending, 'pending_chars': pending_chars,
                    'running_batches': running, 'reconciling_batches': reconciling,
                    'in_flight_requests': requests,
                    'in_flight_usage_unknown': bool(running or reconciling or requests),
                    'estimate_samples': profile['samples'] if profile else 0,
                    'listener_errors': self.listener_errors, 'dropped_notifications': self.dropped_notifications,
                    'observation_errors': self.observation_errors}
