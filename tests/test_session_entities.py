"""Session continuity unit tests; real-service cases require explicit local flags."""
import pytest
from test_postgres_transactions import local_db
from tomte.engine.core.config import TomteSettings
from tomte.engine.utils.keys import generate_session_entity_key


def test_session_settings_and_keys(monkeypatch):
    s = TomteSettings(_env_file=None)
    assert (s.session_entity_cosine_floor, s.session_entity_activation_cap,
            s.session_entity_retention_cap, s.session_entity_return_cap,
            s.session_entity_ttl_seconds) == (0.25, 5, 10, 5, 86400)
    assert generate_session_entity_key("tenant", "user", "session") == "Tomte:session:entities:tenant:user:session"
    from tomte.engine.core.config import settings
    monkeypatch.setattr(settings, "isolate_chat_sessions", False)
    assert len({generate_session_entity_key(*scope) for scope in
                [("t", "u", "s"), ("other", "u", "s"), ("t", "other", "s"), ("t", "u", "other")]}) == 4
    with pytest.raises(ValueError):
        generate_session_entity_key("t:other", "u", "s")
    with pytest.raises(ValueError):
        TomteSettings(_env_file=None, session_entity_activation_cap=0)


@pytest.fixture
def local_session():
    import os
    from uuid import uuid4
    from urllib.parse import urlsplit
    from dotenv import dotenv_values
    from tomte.engine.providers.redis_provider import RedisCacheProvider
    if os.environ.get("TOMTE_RUN_REDIS_TESTS") != "1":
        pytest.skip("Enable TOMTE_RUN_REDIS_TESTS=1 for local Redis")
    config = dotenv_values(".env")
    url = os.environ.get("REDIS_URL") or config.get("REDIS_URL")
    if not url or urlsplit(url).hostname not in ("localhost", "127.0.0.1", "::1"):
        pytest.fail("Session tests require local Redis", pytrace=False)
    cache = RedisCacheProvider(url)
    suffix = uuid4().hex
    tenant = "session-test-" + suffix
    try:
        cache.client.ping()
        yield cache, tenant
    finally:
        # Only this fixture's unique namespace; no flush and no other values read.
        keys = list(cache.client.scan_iter(match="*" + suffix + "*"))
        if keys:
            cache.client.delete(*keys)
        from tomte.engine.utils.keys import get_section_key_active_sessions
        active = get_section_key_active_sessions()
        refs = [ref for ref, _ in cache.client.zscan_iter(active) if suffix in ref]
        if refs:
            cache.client.zrem(active, *refs)
        cache.client.close()


def test_redis_priority_dedup_retention_and_return(local_session):
    cache, tenant = local_session
    cache.promote_session_entities(tenant, "u", "s", ["A", "B", "C", "A"], 10, 86400)
    assert cache.get_session_entity_ids(tenant, "u", "s", 10) == ["A", "B", "C"]
    cache.promote_session_entities(tenant, "u", "s", [str(i) for i in range(9)], 10, 86400)
    assert cache.get_session_entity_ids(tenant, "u", "s", 5) == [str(i) for i in range(5)]
    assert cache.get_session_entity_ids(tenant, "u", "s", 10) == [str(i) for i in range(9)] + ["A"]
    cache.promote_session_entities(tenant, "u", "s", ["A"], 10, 86400)
    assert cache.get_session_entity_ids(tenant, "u", "s", 10)[0] == "A"


def test_redis_touch_ttl_and_namespace_isolation(local_session):
    cache, tenant = local_session
    key = generate_session_entity_key(tenant, "u", "s")
    cache.touch_session_entities(tenant, "u", "s", 86400)
    cache.promote_session_entities(tenant, "u", "s", [], 10, 86400)
    assert not cache.client.exists(key)
    cache.promote_session_entities(tenant, "u", "s", ["A", "B"], 10, 86400)
    cache.client.expire(key, 30)
    cache.touch_session_entities(tenant, "u", "s", 86400)
    assert 86390 <= cache.client.ttl(key) <= 86400
    cache.client.expire(key, 30)
    cache.promote_session_entities(tenant, "u", "s", [], 10, 86400)
    assert 86390 <= cache.client.ttl(key) <= 86400
    assert cache.get_session_entity_ids(tenant, "u", "s", 10) == ["A", "B"]
    for scope in [(tenant + "-other", "u", "s"), (tenant, "v", "s"), (tenant, "u", "other")]:
        assert cache.get_session_entity_ids(*scope, 10) == []


def test_redis_concurrent_promotions_are_atomic(local_session):
    from concurrent.futures import ThreadPoolExecutor
    cache, tenant = local_session
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: cache.promote_session_entities(tenant, "u", "s", [str(i), "shared"], 10, 86400), range(8)))
    ids = cache.get_session_entity_ids(tenant, "u", "s", 20)
    assert set(ids) == {str(i) for i in range(8)} | {"shared"}
    assert len(ids) == 9  # No lost unique promotions when under the retention cap.
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: cache.promote_session_entities(tenant, "u", "s", [str(i)], 10, 86400), range(30)))
    ids = cache.get_session_entity_ids(tenant, "u", "s", 20)
    assert len(ids) == len(set(ids)) == 10
    assert 86390 <= cache.client.ttl(generate_session_entity_key(tenant, "u", "s")) <= 86400


@pytest.fixture
def local_session_db(local_session):
    import os
    from urllib.parse import urlsplit
    from dotenv import dotenv_values
    from tomte.engine.providers.postgres_provider import PostgresProvider
    if os.environ.get("TOMTE_RUN_POSTGRES_TESTS") != "1":
        pytest.skip("Enable TOMTE_RUN_POSTGRES_TESTS=1 for local PostgreSQL")
    url = os.environ.get("POSTGRES_URL") or dotenv_values(".env").get("POSTGRES_URL")
    if not url or urlsplit(url).hostname not in ("localhost", "127.0.0.1", "::1"):
        pytest.fail("Session tests require local PostgreSQL", pytrace=False)
    db = PostgresProvider(url)
    cache, tenant = local_session
    class Rollback(Exception):
        pass
    try:
        try:
            with db.transaction():
                yield db, cache, tenant
                raise Rollback()
        except Rollback:
            pass
    finally:
        db.close()


def insert_entity(db, tenant, user, eid, name, aliases=(), vector=None, facts=()):
    vector = vector or [1.0] + [0.0] * 1535
    payload = dict(entity_id=eid, entity_name=name, entity_domain="assets_and_possessions",
                   entity_type="vehicle", aliases=list(aliases), facts=list(facts))
    assert db.commit_graph_updates([dict(mode="INSERT", tenant_id=tenant, user_id=user,
        entity_id=eid, entity_name=name, entity_domain=payload["entity_domain"],
        entity_type=payload["entity_type"], payload=payload, embedding=vector)])


def test_scoped_discovery_lexical_boundaries_and_current_ordered_fetch(local_session_db):
    import json
    db, cache, tenant = local_session_db
    insert_entity(db, tenant, "u", "a", "Tesla Model Y", ["Little Dog"])
    insert_entity(db, tenant, "u", "b", "Terry", vector=[0.5, 0.8660254] + [0.0] * 1534)
    insert_entity(db, tenant, "other", "a", "Tesla Model Y", ["Little Dog"])
    insert_entity(db, tenant + "-other", "u", "a", "Tesla Model Y", ["Little Dog"])
    assert db.find_session_entity_mentions(tenant, "u", "I named LITTLE DOG.", [1.0] + [0.0] * 1535, 5)[0]["entity_id"] == "a"
    assert db.find_session_entity_mentions(tenant, "u", "Ter", [1.0] + [0.0] * 1535, 5) == []
    assert db.find_session_entity_mentions(tenant, "u", "TERRY!", [1.0] + [0.0] * 1535, 5)[0]["entity_id"] == "b"
    assert db.find_session_entity_mentions(tenant, "absent", "Little Dog", [1.0] + [0.0] * 1535, 5) == []
    with db.conn.cursor() as cur:
        cur.execute("UPDATE tomte_memory_engine SET mention_count=10000 WHERE tenant_id=%s AND user_id=%s AND entity_id='b'", (tenant, "u"))
        cur.execute("UPDATE tomte_memory_engine SET last_mentioned_at=NOW()-INTERVAL '1 day' WHERE tenant_id=%s AND user_id=%s AND entity_id='a'", (tenant, "u"))
    vector = [1.0] + [0.0] * 1535
    assert db.semantic_retrieval(tenant, "u", vector)[0]["entity_id"] == "b"
    rows = db.discover_session_entities(tenant, "u", vector, 0.25, 5)
    assert [r["entity_id"] for r in rows] == ["a", "b"]
    assert rows[0]["raw_cosine"] == pytest.approx(1)
    assert [r["entity_id"] for r in db.discover_session_entities(tenant, "u", vector, 0.75, 5)] == ["a"]
    assert [r["entity_id"] for r in db.get_entities_by_ids(tenant, "u", ["b", "missing", "a", "a"])] == ["b", "a"]
    assert db.get_entities_by_ids(tenant, "absent", ["a"]) == []
    assert all(r["tenant_id"] == tenant and r["user_id"] == "u" for r in rows)
    with db.conn.cursor() as cur:
        cur.execute("UPDATE tomte_memory_engine SET payload=jsonb_set(payload, '{entity_name}', %s::jsonb) WHERE tenant_id=%s AND user_id=%s AND entity_id='a'", (json.dumps("Current Tesla"), tenant, "u"))
    assert db.get_entities_by_ids(tenant, "u", ["a"])[0]["entity_name"] == "Current Tesla"


def test_raw_cosine_ties_have_stable_entity_order(local_session_db):
    db, _, tenant = local_session_db
    for eid in ["c", "a", "b"]:
        insert_entity(db, tenant, "u", eid, eid)
    assert [r["entity_id"] for r in db.discover_session_entities(tenant, "u", [1.0] + [0.0] * 1535, 0.25, 2)] == ["a", "b"]


def test_persona_does_not_consume_session_caps(session_workflow):
    from tomte.engine.core.enums import ChatSenderRole, MemoryDomain
    workflow, db, cache, tenant, _ = session_workflow
    vector = [1.0] + [0.0] * 1535
    # Profiles outrank vehicles lexically and semantically; exclusion must happen
    # before each discovery LIMIT, rather than dropping bounded results afterward.
    for i in range(6):
        insert_entity(db, tenant, "u", "profile-" + str(i), "Tesla Profile " + str(i), ["Tesla"])
    with db.conn.cursor() as cur:
        cur.execute("UPDATE tomte_memory_engine SET entity_domain=%s, payload=jsonb_set(payload, '{entity_domain}', to_jsonb(%s::text)) WHERE tenant_id=%s AND user_id=%s",
                    (MemoryDomain.IDENTITY.value, MemoryDomain.IDENTITY.value, tenant, "u"))
    for i in range(5):
        insert_entity(db, tenant, "u", "vehicle-" + str(i), "Tesla Vehicle " + str(i), ["Tesla"],
                      vector=[0.5, 0.8660254] + [0.0] * 1534)
    expected = ["vehicle-" + str(i) for i in range(5)]
    assert [r["entity_id"] for r in db.find_session_entity_mentions(tenant, "u", "Tesla", vector, 5)] == expected
    assert [r["entity_id"] for r in db.discover_session_entities(tenant, "u", vector, 0.25, 5)] == expected
    result = workflow.on_chat(tenant, "s", "My Tesla needs an oil change.", ChatSenderRole.USER, "u")
    assert cache.get_session_entity_ids(tenant, "u", "s", 10) == expected
    assert "Tesla Profile" in result["context_text"]  # Persona retrieval remains active.
    assert all("Tesla Vehicle " + str(i) in result["context_text"] for i in range(5))


def test_fresh_session_payload_wins_over_earlier_live_retrieval(session_workflow, monkeypatch):
    import json
    from copy import deepcopy
    from tomte.engine.core.enums import ChatSenderRole
    workflow, db, _, tenant, _ = session_workflow
    insert_entity(db, tenant, "u", "tesla", "Old Tesla", ["Tesla"],
                  facts=[dict(attribute_name="paint", value="blue", importance=4)])
    semantic = db.get_raw_semantic_context
    def update_after_old_reads(*args, **kwargs):
        old = semantic(*args, **kwargs)
        fresh = deepcopy({k: v for k, v in old[0].items() if k not in ("id", "tenant_id", "user_id", "relevance_score")})
        fresh["entity_name"] = "Fresh Tesla"
        fresh["facts"][0]["value"] = "red"
        with db.conn.cursor() as cur:
            cur.execute("UPDATE tomte_memory_engine SET payload=%s::jsonb WHERE tenant_id=%s AND user_id=%s AND entity_id='tesla'",
                        (json.dumps(fresh), tenant, "u"))
        return old
    monkeypatch.setattr(db, "get_raw_semantic_context", update_after_old_reads)
    result = workflow.on_chat(tenant, "s", "My Tesla needs an oil change.", ChatSenderRole.USER, "u")["context_text"]
    assert "Fresh Tesla" in result and "paint: red" in result
    assert "Old Tesla" not in result and "paint: blue" not in result
    assert result.count("[[ENTITY: tesla |") == 1


@pytest.fixture
def session_workflow(local_session_db, monkeypatch):
    from types import SimpleNamespace
    from tomte.engine import Tomte
    from tomte.engine.workflows.chat_hook import ChatHookWorkflow
    db, cache, tenant = local_session_db
    vector = lambda i: [float(j == i) for j in range(1536)]
    vectors = {"My Tesla needs an oil change.": vector(0), "My dog has been scratching.": vector(1)}
    calls = []
    def embed(text):
        calls.append(text)
        return vectors.get(text, vector(2))  # Generic followups cannot rediscover either.
    monkeypatch.setattr(Tomte, "_db", db)
    monkeypatch.setattr(Tomte, "_cache", cache)
    monkeypatch.setattr(Tomte, "_llm", SimpleNamespace(embed_text=embed))
    monkeypatch.setattr("threading.Thread.start", lambda *args: pytest.fail("Worker started"))
    return ChatHookWorkflow(), db, cache, tenant, calls


def test_workflow_continuity_topic_switch_assistant_and_fresh_values(session_workflow, monkeypatch):
    import json
    from tomte.engine.core.enums import ChatSenderRole
    workflow, db, cache, tenant, calls = session_workflow
    insert_entity(db, tenant, "u", "tesla", "Tesla Model Y", ["Tesla"])
    insert_entity(db, tenant, "u", "dog", "Milo", ["my dog"], vector=[0.0, 1.0] + [0.0] * 1534)
    # Isolate continuity evidence from the still-preserved old retrieval streams.
    for name in ("get_recent_entities", "get_raw_semantic_context", "get_important_entities"):
        monkeypatch.setattr(db, name, lambda *args, **kwargs: [])
    def send(text, role=ChatSenderRole.USER, session="s", user="u", t=tenant):
        return workflow.on_chat(t, session, text, role, user)["context_text"]
    assert "Tesla Model Y" in send("My Tesla needs an oil change.")
    assert cache.get_session_entity_ids(tenant, "u", "s", 10) == ["tesla"]
    for text in ("What kind should I use?", "Is synthetic better?"):
        send("Tesla Tesla Tesla", ChatSenderRole.ASSISTANT)
        assert "Tesla Model Y" in send(text)
        assert cache.get_session_entity_ids(tenant, "u", "s", 10) == ["tesla"]
    assert "Milo" in send("My dog has been scratching.")
    assert cache.get_session_entity_ids(tenant, "u", "s", 10) == ["dog", "tesla"]
    key = generate_session_entity_key(tenant, "u", "s")
    cache.client.expire(key, 30)
    send("Tesla Tesla Tesla", ChatSenderRole.ASSISTANT)
    assert cache.get_session_entity_ids(tenant, "u", "s", 10) == ["dog", "tesla"]
    assert 86390 <= cache.client.ttl(key) <= 86400
    assert "Tesla Model Y" in send("What is the weather like?")
    assert cache.get_session_entity_ids(tenant, "u", "s", 10) == ["dog", "tesla"]
    with db.conn.cursor() as cur:
        cur.execute("UPDATE tomte_memory_engine SET payload=jsonb_set(payload, '{entity_name}', %s::jsonb) WHERE tenant_id=%s AND user_id=%s AND entity_id='tesla'", (json.dumps("Fresh Tesla"), tenant, "u"))
    assert "Fresh Tesla" in send("What kind should I use?")
    for t, u, s in [(tenant, "u", "other"), (tenant, "other", "s"), (tenant + "-other", "u", "s")]:
        assert send("What kind should I use?", t=t, user=u, session=s) == ""
    assert "Tesla Tesla Tesla" not in calls
    assert len(calls) == 9
    # Missing/deleted IDs are ignored, without returning a stale cached snapshot.
    with db.conn.cursor() as cur:
        cur.execute("DELETE FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s AND entity_id='tesla'", (tenant, "u"))
    assert "Tesla" not in send("What kind should I use?")


def test_workflow_activation_priority_cap_and_dedup(session_workflow):
    from tomte.engine.core.enums import ChatSenderRole
    workflow, db, cache, tenant, _ = session_workflow
    workflow.llm.embed_text = lambda text: [1.0] + [0.0] * 1535
    for i in range(8):
        insert_entity(db, tenant, "u", str(i), "Object " + str(i))
    workflow.on_chat(tenant, "s", "Object 7", ChatSenderRole.USER, "u")
    # Lexical 7 first, then cosine ties by canonical ID. Final cap is five.
    assert cache.get_session_entity_ids(tenant, "u", "s", 10) == ["7", "0", "1", "2", "3"]
    payload = workflow.on_chat(tenant, "s", "Object 6", ChatSenderRole.USER, "u")
    ids = cache.get_session_entity_ids(tenant, "u", "s", 10)
    assert ids == ["6", "0", "1", "2", "3", "7"]
    assert len(cache.get_session_entity_ids(tenant, "u", "s", 5)) == 5
    assert payload["context_text"].count("[[ENTITY: 0 ") == 1
    workflow.on_chat(tenant, "s", "Object 7", ChatSenderRole.USER, "u")
    assert cache.get_session_entity_ids(tenant, "u", "s", 10) == ["7", "0", "1", "2", "3", "6"]


def test_optional_failures_preserve_existing_context_and_scope_errors_propagate(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from tomte.engine import Tomte
    from tomte.engine.exceptions import TomteSessionStateUnavailable, TomteSecurityViolationError
    from tomte.engine.workflows.chat_hook import ChatHookWorkflow
    from tomte.engine.core.enums import ChatSenderRole
    entity = dict(entity_id="a", entity_name="Tesla", tenant_id="t", user_id="u", aliases=[], facts=[])
    db = SimpleNamespace(upsert_user_activity=Mock(), get_recent_entities=Mock(return_value=[entity]),
        get_important_entities=Mock(return_value=[entity]),
        get_raw_semantic_context=Mock(return_value=[]),
        find_session_entity_mentions=Mock(side_effect=TomteSessionStateUnavailable()),
        discover_session_entities=Mock(return_value=[entity]), get_entities_by_ids=Mock(return_value=[entity]))
    cache = SimpleNamespace(get_persona_entities=Mock(return_value=[entity]), get_chat_history=Mock(return_value=[]),
        append_chat_history=Mock(), enqueue_for_consolidation=Mock(),
        promote_session_entities=Mock(side_effect=TomteSessionStateUnavailable()),
        get_session_entity_ids=Mock(return_value=["a"]), touch_session_entities=Mock())
    for name, value in [("_db", db), ("_cache", cache), ("_llm", SimpleNamespace(embed_text=Mock(return_value=[1.0])))]:
        monkeypatch.setattr(Tomte, name, value)
    workflow = ChatHookWorkflow()
    assert "Tesla" in workflow.on_chat("t", "s", "hello", ChatSenderRole.USER, "u")["context_text"]
    cache.promote_session_entities.assert_called_once()
    assert cache.promote_session_entities.call_args.args[3] == ["a"]
    cache.get_session_entity_ids.side_effect = TomteSessionStateUnavailable()
    assert "Tesla" in workflow.on_chat("t", "s", "hello", ChatSenderRole.USER, "u")["context_text"]
    cache.get_session_entity_ids.side_effect = None
    db.get_entities_by_ids.side_effect = TomteSessionStateUnavailable()
    assert "Tesla" in workflow.on_chat("t", "s", "hello", ChatSenderRole.USER, "u")["context_text"]
    db.get_entities_by_ids.side_effect = None
    db.find_session_entity_mentions.side_effect = None
    db.find_session_entity_mentions.return_value = [entity]
    db.discover_session_entities.side_effect = TomteSessionStateUnavailable()
    assert "Tesla" in workflow.on_chat("t", "s", "hello", ChatSenderRole.USER, "u")["context_text"]
    assert cache.promote_session_entities.call_args.args[3] == ["a"]
    cache.touch_session_entities.side_effect = TomteSessionStateUnavailable()
    workflow.on_chat("t", "s", "Tesla Tesla", ChatSenderRole.ASSISTANT, "u")
    cache.touch_session_entities.assert_called_once()
    # Duck-typed legacy providers remain usable; absent optional methods are unsupported.
    del db.find_session_entity_mentions
    del db.discover_session_entities
    assert "Tesla" in workflow.on_chat("t", "s", "hello", ChatSenderRole.USER, "u")["context_text"]
    db.discover_session_entities = Mock(return_value=[{**entity, "tenant_id": "other"}])
    with pytest.raises(TomteSecurityViolationError):
        workflow.on_chat("t", "s", "hello", ChatSenderRole.USER, "u")


def test_workflow_same_names_and_ids_in_other_namespaces_do_not_leak(session_workflow):
    from tomte.engine.core.enums import ChatSenderRole
    workflow, db, cache, tenant, _ = session_workflow
    for t, u, value in [(tenant, "u", "one"), (tenant, "other", "two"), (tenant + "-other", "u", "three")]:
        insert_entity(db, t, u, "tesla", "Tesla Model Y", ["Tesla"],
                      facts=[dict(attribute_name="synthetic_marker", value=value, importance=4)])
    for t, u, value in [(tenant, "u", "one"), (tenant, "other", "two"), (tenant + "-other", "u", "three")]:
        result = workflow.on_chat(t, "s", "My Tesla needs an oil change.", ChatSenderRole.USER, u)
        assert "synthetic_marker: " + value in result["context_text"]
        for other in {"one", "two", "three"} - {value}:
            assert "synthetic_marker: " + other not in result["context_text"]
        assert cache.get_session_entity_ids(t, u, "s", 10) == ["tesla"]


def test_redis_session_expires_without_close(local_session):
    import time
    cache, tenant = local_session
    cache.promote_session_entities(tenant, "u", "s", ["A"], 10, 1)
    time.sleep(1.2)
    assert cache.get_session_entity_ids(tenant, "u", "s", 10) == []
    cache.touch_session_entities(tenant, "u", "s", 86400)
    assert not cache.client.exists(generate_session_entity_key(tenant, "u", "s"))


def test_exact_alias_can_activate_below_cosine_floor(session_workflow):
    from tomte.engine.core.enums import ChatSenderRole
    workflow, db, cache, tenant, _ = session_workflow
    insert_entity(db, tenant, "u", "tesla", "Tesla Model Y", ["Little Dog"],
                  vector=[0.0, 1.0] + [0.0] * 1534)
    workflow.llm.embed_text = lambda text: [1.0] + [0.0] * 1535
    assert db.discover_session_entities(tenant, "u", [1.0] + [0.0] * 1535, 0.25, 5) == []
    workflow.on_chat(tenant, "s", "LITTLE DOG needs service.", ChatSenderRole.USER, "u")
    assert cache.get_session_entity_ids(tenant, "u", "s", 10) == ["tesla"]
    assert db.find_session_entity_mentions(tenant, "u", "Little Doggie", [1.0] + [0.0] * 1535, 5) == []


def test_malformed_redis_type_fails_without_replacing_data(local_session):
    import redis
    cache, tenant = local_session
    key = generate_session_entity_key(tenant, "u", "s")
    cache.client.set(key, "synthetic wrong type", ex=60)
    with pytest.raises(redis.ResponseError):
        cache.promote_session_entities(tenant, "u", "s", ["a"], 10, 86400)
    with pytest.raises(redis.ResponseError):
        cache.get_session_entity_ids(tenant, "u", "s", 5)
    assert cache.client.get(key) == "synthetic wrong type"
    assert 0 < cache.client.ttl(key) <= 60


@pytest.mark.parametrize("retention,ttl", [(1.5, 86400), (10, 86400.5)])
def test_invalid_mru_bounds_cannot_partially_write_state(local_session, retention, ttl):
    cache, tenant = local_session
    key = generate_session_entity_key(tenant, "u", "s")
    with pytest.raises(ValueError, match="Invalid session entity update"):
        cache.promote_session_entities(tenant, "u", "s", ["a"], retention, ttl)
    assert not cache.client.exists(key)


def test_guests_and_assistants_do_not_discover(session_workflow, monkeypatch):
    from unittest.mock import Mock
    from tomte.engine.core.enums import ChatSenderRole
    workflow, db, cache, tenant, calls = session_workflow
    lexical, cosine, heartbeat = Mock(), Mock(), Mock()
    monkeypatch.setattr(db, "find_session_entity_mentions", lexical)
    monkeypatch.setattr(db, "discover_session_entities", cosine)
    monkeypatch.setattr(db, "upsert_user_activity", heartbeat)
    workflow.on_chat(tenant, "s", "Hello", ChatSenderRole.USER)
    workflow.on_chat(tenant, "s", "Tesla Tesla", ChatSenderRole.ASSISTANT)
    workflow.on_chat(tenant, "s", "Tesla Tesla", ChatSenderRole.ASSISTANT, "u")
    lexical.assert_not_called()
    cosine.assert_not_called()
    heartbeat.assert_not_called()
    assert calls == []
    assert not cache.client.exists(generate_session_entity_key(tenant, "u", "s"))


def test_provider_outages_are_explicit_but_malformed_rows_are_not_hidden(monkeypatch):
    from contextlib import nullcontext
    from unittest.mock import Mock
    import psycopg
    import redis
    from tomte.engine.providers import postgres_provider
    from tomte.engine.providers.postgres_provider import PostgresProvider
    from tomte.engine.providers.redis_provider import RedisCacheProvider
    from tomte.engine.exceptions import TomteSessionStateUnavailable
    cache = RedisCacheProvider.__new__(RedisCacheProvider)
    cache.client = Mock()
    for name, call in [("eval", lambda: cache.promote_session_entities("t", "u", "s", ["a"], 10, 86400)),
                       ("lrange", lambda: cache.get_session_entity_ids("t", "u", "s", 5)),
                       ("expire", lambda: cache.touch_session_entities("t", "u", "s", 86400))]:
        getattr(cache.client, name).side_effect = redis.ConnectionError("synthetic outage")
        with pytest.raises(TomteSessionStateUnavailable):
            call()
    db = PostgresProvider.__new__(PostgresProvider)
    db.conn = object()
    monkeypatch.setattr(db, "transaction", nullcontext)
    monkeypatch.setattr(postgres_provider.operations, "get_entities_by_ids", Mock(side_effect=psycopg.OperationalError("synthetic outage")))
    with pytest.raises(TomteSessionStateUnavailable):
        db.get_entities_by_ids("t", "u", ["a"])
    monkeypatch.setattr(postgres_provider.operations, "get_entities_by_ids", Mock(side_effect=psycopg.ProgrammingError("synthetic SQL defect")))
    with pytest.raises(psycopg.ProgrammingError):
        db.get_entities_by_ids("t", "u", ["a"])
    cache.client.eval.side_effect = redis.ResponseError("synthetic Redis defect")
    with pytest.raises(redis.ResponseError):
        cache.promote_session_entities("t", "u", "s", ["a"], 10, 86400)
    monkeypatch.setattr(postgres_provider.operations, "get_entities_by_ids", Mock(return_value=[{"payload": {}}]))
    with pytest.raises(ValueError, match="missing a nonempty entity_id"):
        db.get_entities_by_ids("t", "u", ["a"])


@pytest.mark.parametrize("method,args", [
    ("discover_session_entities", ([1.0] + [0.0] * 1535, 0.25, 5)),
    ("find_session_entity_mentions", ("Tesla", [1.0] + [0.0] * 1535, 5)),
    ("get_entities_by_ids", (["a"],)),
])
def test_new_retrieval_normalization_failure_poison_outer_transaction(local_db, monkeypatch, method, args):
    from tomte.engine.database.postgre.transactions import TransactionScopeError
    db, tenant, user = local_db.db, local_db.tenant, local_db.users[0]
    def fail(rows):
        raise ValueError("synthetic normalization failure")
    monkeypatch.setattr(db, "_normalize_memory_rows", fail)
    with pytest.raises(TransactionScopeError):
        with db.transaction():
            db.upsert_user_activity(tenant, user)
            with pytest.raises(ValueError, match="synthetic normalization failure"):
                getattr(db, method)(tenant, user, *args)
    assert local_db.visible(user) == 0
