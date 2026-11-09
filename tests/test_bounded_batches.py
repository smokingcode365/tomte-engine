"""Whole-message batching on owned Redis namespaces; no model HTTP."""
import json
import time

import pytest

from test_managed_integration import real_worker
from test_postgres_transactions import local_db
from tomte.engine.core.config import settings
from tomte.engine.core.extraction import build_extraction_input
from tomte.engine.providers.redis_provider import RedisCacheProvider
from tomte.engine.utils.keys import generate_consolidation_key, generate_session_ref, get_section_key_active_sessions


def enqueue(case, text, context=None, user=None, session="s"):
    user = user or case.users[0]
    scope = (case.t, user, session)
    if scope not in case.sessions:
        case.sessions.append(scope)
    case.cache.enqueue_for_consolidation(*scope, text, "user", [1.] + [0.] * 1535, context)
    return scope


def queued(case, scope):
    return [json.loads(raw) for raw in case.cache.client.lrange(generate_consolidation_key(*scope), 0, -1)]


def size(items):
    return len(build_extraction_input(items).encode("utf-8"))


def candidates(case, now=None, idle=300):
    found = RedisCacheProvider.worker_candidates(case.cache, now or time.time(), idle)
    return [scope for scope in found if scope[0] == case.t]


def test_long_queue_splits_in_order_with_context_and_stable_retries(real_worker):
    c = real_worker
    history = []
    for index in range(40):
        text = f"Message {index}: " + "x" * 3900
        scope = enqueue(c, text, history[-4:])
        history.append({"role": "user", "content": text})
    original = queued(c, scope)
    first = c.cache.pop_session_chat_log_4_consolidation(*scope)
    assert 0 < len(first) < len(original)
    assert size(first) <= 65536
    assert first + queued(c, scope) == original
    enqueue(c, "Arrived after freeze", history[-4:])
    assert c.cache.pop_session_chat_log_4_consolidation(*scope) == first
    seen, batches = [], 0
    batch = first
    while batch:
        assert size(batch) <= 65536
        assert json.loads(build_extraction_input(batch))["RECENT_CONTEXT_REFERENCE_ONLY"] == batch[0]["recent_context"]
        seen.extend(batch)
        batches += 1
        assert c.cache.clear_session_4_consolidation(*scope, batch)
        batch = c.cache.pop_session_chat_log_4_consolidation(*scope)
        if batch:
            assert not c.cache.clear_session_4_consolidation(*scope, first)
    assert batches >= 3
    assert seen[:-1] == original
    assert seen[-1]["content"] == "Arrived after freeze"
    assert len({item["item_id"] for item in seen}) == len(seen)


@pytest.mark.parametrize("text", ["x" * 1000, "漢🙂" * 250, 'line\n"quoted"\\' * 100])
def test_exact_wire_budget_and_active_size_trigger(real_worker, monkeypatch, text):
    c = real_worker
    scope = enqueue(c, text, [{"role": "assistant", "content": "preceding context"}])
    first = queued(c, scope)
    exact = size(first)
    monkeypatch.setattr(settings, "consolidation_batch_bytes", exact + 1)
    assert scope not in candidates(c)
    monkeypatch.setattr(settings, "consolidation_batch_bytes", exact)
    assert scope in candidates(c)
    enqueue(c, "second")
    frozen = c.cache.pop_session_chat_log_4_consolidation(*scope)
    assert frozen == first and size(frozen) == exact
    assert [i["content"] for i in queued(c, scope)] == ["second"]


def test_small_chat_waits_for_idle_and_other_sessions_stay_separate(real_worker):
    c = real_worker
    scopes = [enqueue(c, "small"), enqueue(c, "different session", session="other"),
              enqueue(c, "different user", user=c.users[1])]
    now = time.time()
    assert not candidates(c, now)
    assert set(candidates(c, now + 301)) == set(scopes)
    frozen = c.cache.pop_session_chat_log_4_consolidation(*scopes[0])
    assert len(frozen) == 1 and frozen[0]["content"] == "small"
    assert all(len(queued(c, scope)) == 1 for scope in scopes[1:])


def test_append_between_selection_and_atomic_freeze_is_preserved(real_worker, monkeypatch):
    c = real_worker
    scope = enqueue(c, "first")
    expected = queued(c, scope)
    original_eval = c.cache.client.eval
    injected = []
    def append_then_eval(script, *args):
        if "bounded-prefix-freeze" in script and not injected:
            injected.append(True)
            enqueue(c, "concurrent arrival")
        return original_eval(script, *args)
    monkeypatch.setattr(c.cache.client, "eval", append_then_eval)
    assert c.cache.pop_session_chat_log_4_consolidation(*scope) == expected
    assert [item["content"] for item in queued(c, scope)] == ["concurrent arrival"]
    assert c.cache.clear_session_4_consolidation(*scope, expected)
    assert [item["content"] for item in c.cache.pop_session_chat_log_4_consolidation(*scope)] == ["concurrent arrival"]


def test_two_freezers_get_same_atomic_snapshot(real_worker, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    c = real_worker
    scope = enqueue(c, "x" * 1000)
    enqueue(c, "y" * 1000)
    monkeypatch.setattr(settings, "consolidation_batch_bytes", size(queued(c, scope)[:1]))
    barrier = Barrier(2)
    original_eval = c.cache.client.eval
    def synchronized(script, *args):
        if "bounded-prefix-freeze" in script:
            barrier.wait(timeout=5)
        return original_eval(script, *args)
    monkeypatch.setattr(c.cache.client, "eval", synchronized)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: c.cache.pop_session_chat_log_4_consolidation(*scope), range(2)))
    assert len(results[0]) == 1 and results[0] == results[1]
    assert [item["content"] for item in queued(c, scope)] == ["y" * 1000]


def test_reference_context_shrinks_only_when_first_whole_message_needs_room(real_worker):
    c = real_worker
    context = [{"role": "user", "content": str(i) + "🙂" * 3999} for i in range(4)]
    scope = enqueue(c, "🙂" * 4000, context)
    before = queued(c, scope)[0]
    frozen = c.cache.pop_session_chat_log_4_consolidation(*scope)
    assert size(frozen) <= 65536
    assert frozen[0]["recent_context"] == context[1:]
    assert {k: v for k, v in frozen[0].items() if k != "recent_context"} == {
        k: v for k, v in before.items() if k != "recent_context"}


def test_single_oversized_message_is_intact_and_isolated(real_worker):
    c = real_worker
    scope = enqueue(c, "x" * 70000)
    enqueue(c, "later")
    first = queued(c, scope)[0]
    assert c.cache.pop_session_chat_log_4_consolidation(*scope) == [first]
    assert [i["content"] for i in queued(c, scope)] == ["later"]


def test_bad_row_is_isolated_from_valid_neighbors(real_worker):
    c = real_worker
    scope = enqueue(c, "before")
    c.cache.client.rpush(generate_consolidation_key(*scope), "{invalid json")
    enqueue(c, "after")
    first = c.cache.pop_session_chat_log_4_consolidation(*scope)
    assert [i["content"] for i in first] == ["before"]
    assert c.cache.clear_session_4_consolidation(*scope, first)
    c.step()  # Existing worker SKIP policy acknowledges only the corrupt row.
    assert not c.calls
    assert [i["content"] for i in queued(c, scope)] == ["after"]
    c.step()
    assert len(c.calls) == 1


def test_managed_worker_drains_full_batches_then_waits_for_idle(real_worker, monkeypatch):
    c = real_worker
    c.worker.idle_timeout_seconds = 300
    monkeypatch.setattr(settings, "consolidation_batch_bytes", 1500)
    monkeypatch.setattr(c.cache, "worker_candidates", lambda now, idle: candidates(c, now, idle))
    for index in range(5):
        scope = enqueue(c, str(index) + "x" * 800)
    original = queued(c, scope)
    for _ in range(5):
        c.step()
    assert len(c.calls) == 4
    assert queued(c, scope) == original[-1:]
    assert [json.loads(text)["CURRENT_BATCH"][0]["item_id"] for text in c.calls] == [i["item_id"] for i in original[:-1]]
    ref = generate_session_ref(scope[0], scope[2], scope[1])
    c.cache.client.zadd(get_section_key_active_sessions(), {ref: time.time() - 301})
    c.step()
    assert len(c.calls) == 5 and not queued(c, scope)
    assert c.cache.worker_state(*scope[:2]) is None


def test_size_trigger_does_not_bypass_retry_delay(real_worker, monkeypatch):
    import httpx
    c = real_worker
    monkeypatch.setattr(settings, "consolidation_batch_bytes", 1000)
    monkeypatch.setattr(c.cache, "worker_candidates", lambda now, idle: candidates(c, now, idle))
    enqueue(c, "x" * 1500)
    def fail():
        raise httpx.ReadTimeout("synthetic transient failure")
    c.action[0] = fail
    c.step()
    enqueue(c, "new pending" * 200)
    c.step()
    assert len(c.calls) == 1
    c.due()
    c.action[0] = None
    c.step()
    assert len(c.calls) == 2 and c.calls[0] == c.calls[1]


def test_existing_larger_snapshot_is_never_resized(real_worker, monkeypatch):
    c = real_worker
    scope = enqueue(c, "x" * 2000)
    enqueue(c, "y" * 2000)
    frozen = c.cache.pop_session_chat_log_4_consolidation(*scope)
    monkeypatch.setattr(settings, "consolidation_batch_bytes", 1000)
    enqueue(c, "later")
    assert c.cache.pop_session_chat_log_4_consolidation(*scope) == frozen
    assert len(frozen) == 2
    assert [i["content"] for i in queued(c, scope)] == ["later"]


def test_active_scan_rotates_through_bounded_pages(real_worker, monkeypatch):
    from tomte.engine.providers import redis_provider
    c = real_worker
    # Keep the pagination test independent of other existing active sessions.
    index = f"Tomte:test-active:{c.t}"
    monkeypatch.setattr(redis_provider, "get_section_key_active_sessions", lambda: index)
    monkeypatch.setattr(settings, "consolidation_batch_bytes", 1000)
    expected = {enqueue(c, "x" * 1200, session=f"s{i}") for i in range(5)}
    seen = set()
    for _ in range(3):
        page = RedisCacheProvider.worker_candidates(c.cache, time.time(), 300, limit=2)
        assert len(page) <= 2
        seen.update(page)
    assert seen == expected and c.cache._scan_offset == 0
