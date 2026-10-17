from threading import Event, Thread
from time import monotonic

from test_managed_worker import Harness


def test_shutdown_does_not_wait_for_blocked_candidate_scan():
    h = Harness()
    h.cache.add()
    entered, release, finished = Event(), Event(), Event()
    candidates = h.cache.worker_candidates
    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return candidates(*args)
    h.cache.worker_candidates = blocked
    h.worker.start_in_background()
    assert entered.wait(3)
    result = []
    def stop():
        started = monotonic()
        result.append((h.worker.stop(.05), monotonic() - started))
        finished.set()
    caller = Thread(target=stop)
    caller.start()
    try:
        assert finished.wait(.5), 'Shutdown waited on Redis while holding the coordinator mutex'
        assert result[0][0] is False and result[0][1] < .5
    finally:
        release.set()
        caller.join(3)
        assert h.worker.stop(3)
    assert h.calls == [] and not h.cache.acquired


def test_stop_during_lease_acquisition_releases_without_admission():
    h = Harness()
    h.cache.add()
    entered, release = Event(), Event()
    acquire = h.cache.acquire_lock
    def blocked(*args, **kwargs):
        value = acquire(*args, **kwargs)
        entered.set()
        assert release.wait(5)
        return value
    h.cache.acquire_lock = blocked
    h.worker.start_in_background()
    assert entered.wait(3)
    try:
        assert h.worker.stop(.05) is False
    finally:
        release.set()
        assert h.worker.stop(3)
    assert h.calls == [] and not h.cache.locks
