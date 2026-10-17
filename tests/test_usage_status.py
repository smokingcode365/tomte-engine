from threading import Event
from tomte.engine.usage import UsageReporter
from test_managed_worker import Harness


def test_running_batch_is_not_idle_and_status_is_user_scoped():
    entered, release = Event(), Event()
    def block(*args):
        entered.set()
        assert release.wait(5)
    h = Harness(block)
    reporter = h.worker.usage = UsageReporter()
    h.cache.add()
    h.worker.poll()
    assert entered.wait(3)
    try:
        running = reporter.query('t', 'u', 0)
        assert running['running_batches'] == 1
        assert running['in_flight_usage_unknown'] is True
        assert running['totals'] == []
        assert reporter.query('t', 'other', 0)['running_batches'] == 0
        assert reporter.query('other', 'u', 0)['in_flight_usage_unknown'] is False
    finally:
        release.set()
        for f in h.worker.futures:
            f.result(timeout=3)
        assert h.worker.stop()
    assert reporter.query('t', 'u', 0)['running_batches'] == 0
    assert reporter.query('t', 'u', 0)['in_flight_usage_unknown'] is False


def test_reconciliation_is_visible_until_marker_ack():
    from tomte.engine.workflows.consolidation import ConsolidationWorkflow
    h = Harness()
    reporter = h.worker.usage = UsageReporter()
    h.cache.add()
    def uncertain(h,t,u,items):
        h.markers.add((t,u,ConsolidationWorkflow._batch_id(t,u,items)))
        h.db_unavailable = True
        raise ConnectionError('synthetic uncertain commit')
    h.behavior = uncertain
    try:
        h.step()
        assert reporter.query('t','u',0)['reconciling_batches'] == 1
        assert reporter.query('t','u',0)['in_flight_usage_unknown'] is True
        h.db_unavailable = False
        h.due()
        h.step()
        h.worker.cooldowns.clear()  # Expire the infrastructure-failure cooldown.
        h.step()
        assert len(h.calls) == 1
        assert reporter.query('t','u',0)['reconciling_batches'] == 0
    finally:
        assert h.worker.stop()


def test_blocked_provider_has_unobserved_usage_without_fabricated_tokens():
    from concurrent.futures import ThreadPoolExecutor
    from tomte.engine.usage import track_provider
    entered, release = Event(), Event()
    reporter = UsageReporter()
    class Provider:
        @track_provider
        def is_junk(self, text):
            entered.set()
            assert release.wait(5)
            return False
    def run():
        with reporter.scope('t', 'u', 'foreground'):
            Provider().is_junk('synthetic')
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run)
        assert entered.wait(3)
        try:
            current = reporter.query('t', 'u', 0)
            assert current['in_flight_requests'] == 1
            assert current['in_flight_usage_unknown'] is True and current['totals'] == []
            assert reporter.query('t','other',0)['in_flight_requests'] == 0
        finally:
            release.set()
        future.result(timeout=3)
    assert reporter.query('t','u',0)['in_flight_requests'] == 0
    assert reporter.query('t','u',0)['totals'][0]['input_tokens'] is None
