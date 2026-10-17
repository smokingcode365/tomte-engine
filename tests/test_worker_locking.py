"""Lease regression intent retained against the managed worker implementation."""
from uuid import UUID
import pytest
from test_managed_worker import Harness


@pytest.fixture
def wired():
    h = Harness()
    h.cache.add()
    yield h
    assert h.worker.stop()


def test_same_session_attempts_use_distinct_tokens_and_release_their_own(wired):
    h = wired
    h.step()
    h.cache.add()
    h.step()
    tokens = [token for _, token in h.cache.acquired]
    assert len(tokens) == 2 and tokens[0] != tokens[1]
    assert all(UUID(token).version == 4 for token in tokens)
    assert h.cache.acquired == h.cache.released
    assert all(key == 'Tomte:lock:consolidation:t:u' for key, _ in h.cache.acquired)
    assert [result[3] for result in h.cache.terminal] == ['committed', 'committed']


@pytest.mark.parametrize('raises', [False, True])
def test_unsuccessful_cycle_releases_token_and_records_skip(wired, raises):
    h = wired
    def fail(*args):
        if raises:
            raise RuntimeError('synthetic failure')
        raise ValueError('batch_not_processed')
    h.behavior = fail
    h.step()
    assert h.cache.acquired == h.cache.released
    assert h.cache.terminal[-1][3] == 'skipped'


def test_busy_lock_is_not_released_or_processed(wired):
    h = wired
    h.cache.locks['Tomte:lock:consolidation:t:u'] = 'other-owner'
    h.step()
    assert h.cache.released == [] and h.calls == []
    assert h.cache.locks['Tomte:lock:consolidation:t:u'] == 'other-owner'

