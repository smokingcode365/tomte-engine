"""Small cooperative control boundary; no billing or semantic policy."""
from dataclasses import dataclass, field
from threading import Event
from time import monotonic
from pydantic import ValidationError
from pydantic_ai.exceptions import UnexpectedModelBehavior


class WorkStopped(RuntimeError):
    pass


class StaleSnapshot(ValueError):
    pass


@dataclass
class WorkControl:
    timeout: float = 300
    stopped: Event = field(default_factory=Event)
    deadline: float = field(init=False)

    def __post_init__(self):
        self.deadline = monotonic() + self.timeout

    def check(self):
        if self.stopped.is_set() or monotonic() >= self.deadline:
            raise WorkStopped('operation_stopped')


def transient(error, seen=None):
    """Only known temporary errors qualify. Never classify by message text."""
    import httpx
    import psycopg
    import redis
    from tomte.engine.core.capacity import CapacityExceeded
    seen = set() if seen is None else seen
    if id(error) in seen:
        return False
    seen.add(id(error))
    if isinstance(error, (WorkStopped, CapacityExceeded, ValidationError, UnexpectedModelBehavior)):
        return False
    if isinstance(error, (StaleSnapshot, httpx.TimeoutException, httpx.NetworkError,
                          psycopg.errors.SerializationFailure, psycopg.errors.DeadlockDetected,
                          psycopg.errors.LockNotAvailable)):
        return True
    if isinstance(error, (redis.exceptions.AuthenticationError, redis.exceptions.AuthorizationError)):
        return False
    if isinstance(error, (redis.TimeoutError, redis.ConnectionError)):
        return True
    status = getattr(error, 'status_code', None)
    if status in (408, 500, 502, 503, 504):
        return True
    if status == 429:
        body = getattr(error, 'body', None)
        detail = body.get('error', body) if isinstance(body, dict) else {}
        return isinstance(detail, dict) and (detail.get('code') or detail.get('type')) in ('rate_limit_exceeded', 'rate_limit_error')
    # Explicit public SDK transport types, preserving typed causes.
    import sys
    for name in ('openai', 'anthropic'):
        kind = getattr(sys.modules.get(name), 'APIConnectionError', None)
        if kind is not None and isinstance(error, kind):
            return True
    cause = error.__cause__
    return transient(cause, seen) if cause is not None else False
