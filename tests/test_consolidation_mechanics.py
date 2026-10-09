"""Deterministic mechanics against bootstrapped local PostgreSQL and Redis.

Enable TOMTE_RUN_POSTGRES_TESTS=1 and TOMTE_RUN_REDIS_TESTS=1. Requires local
POSTGRES_URL/REDIS_URL in the environment or .env and 1536-dimensional vectors.
Only unique synthetic fixtures are written. Teardown removes those fixtures;
the workflow itself never acknowledges the Redis queue. No model APIs run.
"""

import os
import threading
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import dotenv_values
import psycopg
from psycopg.pq import TransactionStatus
import pytest

from tomte.engine import Tomte
from tomte.engine.core.config import settings
from tomte.engine.core.enums import ChatSenderRole
from tomte.engine.core.schema import Entity, Fact, TomteMemorySchema
from tomte.engine.dictionary import service
from tomte.engine.llm.embedder import get_entity_summary
from tomte.engine.providers.postgres_provider import PostgresProvider
from tomte.engine.providers.redis_provider import RedisCacheProvider
from tomte.engine.utils.keys import (
    generate_common_key, generate_consolidation_key, generate_consolidation_batch_key, generate_persona_key,
    generate_session_ref, get_section_key_active_sessions,
)
from tomte.engine.workflows.chat_hook import ChatHookWorkflow
from tomte.engine.workflows.consolidation import ConsolidationWorkflow


REAL_THREAD_START = threading.Thread.start

MESSAGE = "My favorite programming language is Python."
QUESTION = "What programming language do I like?"


class DeterministicInsertLLM:
    def __init__(self, db, index, source):
        self.db, self.index, self.calls = db, index, []
        self.entity = Entity(
            entity_id="pre_01", entity_name="Programming language preference",
            entity_domain="preferences_and_traits", entity_type="programming_preference",
            aliases=["my favorite programming language"],
            facts=[Fact(
                attribute_name="favorite_programming_language", value="Python",
                importance=4, confidence=1.0, source_chat_id=source,
                last_mentioned_at="2026-10-08T00:00:00Z",
            )],
        )

    def vector(self):
        return [float(i == self.index) for i in range(1536)]

    def record(self, method):
        assert self.db.conn.info.transaction_status == TransactionStatus.IDLE
        expected = ["embed_text", "is_junk", "extract", "reconcile", "embed_batch", "embed_text"]
        assert len(self.calls) < len(expected) and method == expected[len(self.calls)]
        self.calls.append(method)

    def get_vector_dimension(self):
        return 1536

    def embed_text(self, text):
        assert text == (MESSAGE if not self.calls else QUESTION)
        self.record("embed_text")
        return self.vector()

    def embed_batch(self, texts):
        assert texts == [get_entity_summary(self.entity.model_dump(mode="json"))]
        self.record("embed_batch")
        return [self.vector()]

    def is_junk(self, chat_log):
        assert chat_log == "user: " + MESSAGE
        self.record("is_junk")
        return False

    def parse_structured(self, system_prompt, user_text, schema, taxonomy=None, lexicon=None):
        if schema is TomteMemorySchema:
            assert user_text == "user: " + MESSAGE
            assert isinstance(taxonomy, list) and isinstance(lexicon, list)
            self.record("extract")
            return TomteMemorySchema(preferences_and_traits=[self.entity])
        if schema.__name__ == "ReconciliationResponse":
            assert "EXISTING_GOLDEN_RECORDS (SKELETON):\n[]" in user_text
            self.record("reconcile")
            return schema.model_validate({"mappings": [{
                "extracted_entity_id": "pre_01", "mode": "INSERT",
                "matched_db_id": None, "reasoning": "Deterministic synthetic insertion",
            }]})
        raise AssertionError("Unexpected structured model call")


@pytest.fixture
def mechanics(monkeypatch):
    if os.environ.get("TOMTE_RUN_POSTGRES_TESTS") != "1" or os.environ.get("TOMTE_RUN_REDIS_TESTS") != "1":
        pytest.skip("Enable local PostgreSQL and Redis integration tests explicitly")
    config = dotenv_values(".env")
    pg_url = os.environ.get("POSTGRES_URL") or config.get("POSTGRES_URL")
    redis_url = os.environ.get("REDIS_URL") or config.get("REDIS_URL")
    for url in (pg_url, redis_url):
        if not url or urlsplit(url).hostname not in ("localhost", "127.0.0.1", "::1"):
            pytest.fail("Mechanics tests require local PostgreSQL and Redis URLs", pytrace=False)
    assert not settings.enable_pii_scrubbing, "This no-network check requires PII model loading disabled"
    db = PostgresProvider(pg_url)
    cache = RedisCacheProvider(redis_url)
    observer = psycopg.connect(pg_url, autocommit=True, options="-c default_transaction_read_only=on")
    tag = uuid4().hex
    tenant, user, session = ("mechanics-test-" + kind + "-" + tag for kind in ("tenant", "user", "session"))
    created_anchors = []
    keys = [generate_common_key(tenant, session, user), generate_persona_key(tenant, user), generate_consolidation_key(tenant, user, session), generate_consolidation_batch_key(tenant, user, session)]
    def forbidden_start(*args):
        raise AssertionError("Background workers are forbidden in this check")
    monkeypatch.setattr("threading.Thread.start", forbidden_start)
    for name in ("_db", "_cache", "_llm", "_worker"):
        monkeypatch.setattr(Tomte, name, None)
    monkeypatch.setattr(service, "_dictionary_service_instance", None)
    create = db.create_dictionary_anchor
    def track_anchor(vector):
        anchor = create(vector)
        created_anchors.append(anchor)
        return anchor
    monkeypatch.setattr(db, "create_dictionary_anchor", track_anchor)
    try:
        assert cache.client.ping()
        assert not list(cache.client.scan_iter(match="*" + tag + "*"))
        # Choose an unused deterministic basis vector; never nudge existing anchors.
        index = next(i for i in range(1536) if not db.find_dictionary_matches_above_threshold([float(j == i) for j in range(1536)], settings.llm_threshold_low))
        llm = DeterministicInsertLLM(db, index, "synthetic-chat-" + tag)
        Tomte.init(db, cache, llm, start_daemon_worker=False)
        yield SimpleNamespace(db=db, cache=cache, observer=observer, tenant=tenant, user=user, session=session, llm=llm, anchors=created_anchors)
    finally:
        try:
            db.delete_user_memory(tenant, user)
            # No provider API deletes a standalone dictionary anchor. Test-only
            # cleanup uses the shared scope and only UUIDs captured at creation.
            with db.transaction():
                with db.conn.cursor() as cursor:
                    cursor.execute("DELETE FROM tomte_processed_consolidations WHERE tenant_id=%s AND user_id=%s", (tenant, user))
                    for anchor in created_anchors:
                        cursor.execute("DELETE FROM tomte_semantic_anchors WHERE dictionary_id=%s", (anchor,))
            cache.client.delete(*keys)
            cache.client.zrem(get_section_key_active_sessions(), generate_session_ref(tenant, session, user))
        finally:
            observer.close()
            cache.client.close()
            db.close()


def queued_message(case):
    assert case.db.get_user_graph_skeleton(case.tenant, case.user) == []
    assert case.observer.execute("SELECT count(*) FROM tomte_user_registry WHERE tenant_id=%s AND user_id=%s", (case.tenant, case.user)).fetchone()[0] == 0
    ChatHookWorkflow().on_chat(case.tenant, case.session, MESSAGE, ChatSenderRole.USER, case.user)
    queue = case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session)
    assert len(queue) == 1 and queue[0]["content"] == MESSAGE
    assert queue[0]["vector"] == case.llm.vector()
    return queue


def test_deterministic_insert_is_committed_and_retrievable(mechanics):
    case = mechanics
    queue = queued_message(case)
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, queue) is True
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session) == queue
    row = case.observer.execute("SELECT payload, vector_dims(embedding) FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s", (case.tenant, case.user)).fetchall()
    assert len(row) == 1 and row[0] == (case.llm.entity.model_dump(mode="json"), 1536)
    assert len(case.anchors) == 1
    keywords = case.db.get_anchor_keywords(str(case.anchors[0]))
    assert {(k["keyword"], k["keyword_type"], k["hits"]) for k in keywords} == {
        ("programming_preference", "ENTITY", 1), ("favorite_programming_language", "ATTRIBUTE", 1),
    }
    context = ChatHookWorkflow().on_chat(case.tenant, case.session, QUESTION, ChatSenderRole.USER, case.user)
    assert "favorite_programming_language: Python" in context["context_text"]
    assert context["context_text"].count("[[ENTITY: pre_01") == 1
    assert case.llm.calls == ["embed_text", "is_junk", "extract", "reconcile", "embed_batch", "embed_text"]
    assert case.db.conn.info.transaction_status == TransactionStatus.IDLE


def test_keyword_write_failure_rolls_back_and_retains_retry_queue(mechanics, monkeypatch):
    case = mechanics
    queue = queued_message(case)
    original = case.db.upsert_dictionary_keyword
    def fail_attribute(anchor, keyword, kind):
        # First keyword writes normally; the second causes a real SQL constraint failure.
        return original(anchor, None if kind == "ATTRIBUTE" else keyword, kind)
    monkeypatch.setattr(case.db, "upsert_dictionary_keyword", fail_attribute)
    with pytest.raises(psycopg.errors.NotNullViolation):
        ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, queue)
    assert case.db.conn.info.transaction_status == TransactionStatus.IDLE
    assert case.db.get_user_graph_skeleton(case.tenant, case.user) == []
    assert len(case.anchors) == 1
    assert case.db.get_dictionary_anchor_details(case.anchors[0]) is None
    assert case.db.get_anchor_keywords(str(case.anchors[0])) == []
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session) == queue
    assert case.cache.client.zscore(get_section_key_active_sessions(), generate_session_ref(case.tenant, case.session, case.user)) is not None
    assert case.llm.calls == ["embed_text", "is_junk", "extract", "reconcile", "embed_batch"]
    assert case.db.conn.info.transaction_status == TransactionStatus.IDLE


class LifecycleLLM(DeterministicInsertLLM):
    """Prescribed outputs test persistence, not model extraction quality."""

    def prepare(self, mode, value, message, db_id=None, step=0):
        self.mode, self.message, self.db_id, self.step = mode, message, db_id, step
        self.calls = []
        fact = self.entity.facts[0].model_copy(update={
            "value": value, "importance": (4, 5, 3)[step],
            "confidence": (0.8, 0.9, 1.0)[step],
            "source_chat_id": "synthetic-step-" + str(step),
            "last_mentioned_at": f"2026-10-08T00:0{step}:00Z",
        })
        # An explicitly prescribed unrelated fixture fact checks merge retention.
        facts = [fact]
        if mode == "INSERT":
            facts.append(fact.model_copy(update={
                "attribute_name": "preferred_editor", "value": "VSCode", "importance": 3,
            }))
        self.entity = self.entity.model_copy(update={
            "facts": facts, "aliases": ["my favorite programming language" if step == 0 else "my preferred language"],
        })

    def vector(self):
        vector = super().vector()
        vector[(self.index + 1) % 1536] = self.step * 0.1
        return vector

    def record(self, method):
        assert self.db.conn.info.transaction_status == TransactionStatus.IDLE
        self.calls.append(method)

    def embed_text(self, text):
        assert text in (self.message, "What programming language do I prefer?")
        self.record("embed_text")
        return self.vector()

    def is_junk(self, chat_log):
        assert chat_log == "user: " + self.message
        self.record("is_junk")
        return False

    def parse_structured(self, system_prompt, user_text, schema, taxonomy=None, lexicon=None):
        if schema is TomteMemorySchema:
            assert user_text == "user: " + self.message
            self.record("extract")
            return TomteMemorySchema(preferences_and_traits=[self.entity])
        assert schema.__name__ == "ReconciliationResponse"
        if self.db_id:
            assert str(self.db_id) in user_text
        self.record("reconcile")
        return schema.model_validate({"mappings": [{
            "extracted_entity_id": self.entity.entity_id, "mode": self.mode,
            "matched_db_id": str(self.db_id) if self.db_id else None,
            "reasoning": "Prescribed local lifecycle decision",
        }]})

    def embed_batch(self, texts):
        assert len(texts) == 1
        assert self.entity.facts[0].value in texts[0] and "VSCode" in texts[0]
        self.record("embed_batch")
        return [self.vector()]


def lifecycle(case):
    case.llm = LifecycleLLM(case.db, case.llm.index, "synthetic-lifecycle")
    case.llm.prepare("INSERT", "Python", MESSAGE)
    Tomte.init(case.db, case.cache, case.llm, start_daemon_worker=False)
    return case


def memory_rows(case):
    # Separate read-only connection verifies committed visibility.
    with case.observer.cursor(row_factory=psycopg.rows.dict_row) as cursor:
        cursor.execute("SELECT id, entity_id, payload, importance, mention_count, created_at, last_mentioned_at, embedding::text AS vector FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s ORDER BY id", (case.tenant, case.user))
        return cursor.fetchall()


def dictionary_state(case):
    with case.observer.cursor() as cursor:
        cursor.execute("SELECT embedding::text FROM tomte_semantic_anchors WHERE dictionary_id=%s", (case.anchors[0],))
        vector = cursor.fetchone()
        cursor.execute("SELECT keyword, keyword_type, hits, created_at, last_modified_at FROM tomte_keyword_registry WHERE dictionary_id=%s ORDER BY keyword, keyword_type", (case.anchors[0],))
        return vector, cursor.fetchall()


def language(row):
    return next(f for f in row["payload"]["facts"] if f["attribute_name"] == "favorite_programming_language")


def run_message(case):
    ChatHookWorkflow().on_chat(case.tenant, case.session, case.llm.message, ChatSenderRole.USER, case.user)
    queue = case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session)
    assert len(queue) == 1 and queue[0]["content"] == case.llm.message
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, queue)
    assert case.llm.calls == ["embed_text", "is_junk", "extract", "reconcile", "embed_batch"]
    assert case.db.conn.info.transaction_status == TransactionStatus.IDLE
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session) == queue
    return queue


def acknowledge(case):
    # Simulate the existing worker's successful acknowledgment, without a worker.
    snapshot = case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session)
    assert case.cache.clear_session_4_consolidation(case.tenant, case.user, case.session, snapshot)


def assert_context(case, value):
    context = ChatHookWorkflow().on_chat(case.tenant, case.session, "What programming language do I prefer?", ChatSenderRole.USER, case.user)["context_text"]
    assert context.count("[[ENTITY: pre_01") == 1
    assert "favorite_programming_language: " + value in context
    if value == "Rust":
        assert "favorite_programming_language: Python" not in context
    acknowledge(case)  # Only this synthetic question's queue is acknowledged.


def test_insert_reinforce_mutate_preserves_one_entity_and_retrieves_current_fact(mechanics):
    case = lifecycle(mechanics)
    run_message(case)
    inserted, = memory_rows(case)
    assert inserted["entity_id"] == "pre_01" and inserted["mention_count"] == 1
    assert language(inserted)["value"] == "Python"
    unrelated = inserted["payload"]["facts"][1]
    acknowledge(case)

    case.llm.prepare("REINFORCE", "Python", "I still really like Python.", inserted["id"], 1)
    run_message(case)
    reinforced, = memory_rows(case)
    assert reinforced["id"] == inserted["id"] and reinforced["entity_id"] == "pre_01"
    assert reinforced["mention_count"] == 2
    assert reinforced["created_at"] == inserted["created_at"]
    assert reinforced["last_mentioned_at"] > inserted["last_mentioned_at"]
    assert reinforced["vector"] != inserted["vector"]
    assert reinforced["importance"] == 4 and language(reinforced)["importance"] == 5
    assert language(reinforced)["confidence"] == 0.9
    assert language(reinforced)["mention_count"] == 1
    assert language(reinforced)["last_mentioned_at"] == "2026-10-08T00:01:00Z"
    acknowledge(case)
    assert_context(case, "Python")

    case.llm.prepare("MUTATE", "Rust", "Actually, I prefer Rust now.", inserted["id"], 2)
    run_message(case)
    mutated, = memory_rows(case)
    assert mutated["id"] == inserted["id"] and mutated["entity_id"] == "pre_01"
    assert mutated["mention_count"] == 3 and mutated["importance"] == 3
    assert mutated["created_at"] == inserted["created_at"]
    assert mutated["last_mentioned_at"] > reinforced["last_mentioned_at"]
    assert mutated["vector"] != reinforced["vector"]
    assert language(mutated)["value"] == "Rust" and language(mutated)["confidence"] == 1.0
    assert language(mutated)["timeline"] == [] and language(mutated)["mention_count"] == 1
    assert unrelated in mutated["payload"]["facts"]
    assert set(mutated["payload"]["aliases"]) == {"my favorite programming language", "my preferred language"}
    hits = {k["keyword"]: k["hits"] for k in case.db.get_anchor_keywords(str(case.anchors[0]))}
    assert hits == {"programming_preference": 3, "favorite_programming_language": 3, "preferred_editor": 1}
    assert len(case.anchors) == 1
    acknowledge(case)
    assert_context(case, "Rust")
    print("Lifecycle: one pre_01, same UUID; Python -> Python -> Rust; row counts 1/2/3; keyword hits 3/3/1; empty fact history; Rust context verified")


@pytest.mark.parametrize("mode,value,step", [("REINFORCE", "Python", 1), ("MUTATE", "Rust", 2)])
def test_lifecycle_dictionary_failure_rolls_back_existing_memory(mechanics, monkeypatch, mode, value, step):
    case = lifecycle(mechanics)
    run_message(case)
    before = memory_rows(case)
    dictionary_before = dictionary_state(case)
    acknowledge(case)
    case.llm.prepare(mode, value, "I still really like Python." if mode == "REINFORCE" else "Actually, I prefer Rust now.", before[0]["id"], step)
    ChatHookWorkflow().on_chat(case.tenant, case.session, case.llm.message, ChatSenderRole.USER, case.user)
    queue = case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session)
    original = case.db.upsert_dictionary_keyword
    def fail_attribute(anchor, keyword, kind):
        return original(anchor, None if kind == "ATTRIBUTE" else keyword, kind)
    monkeypatch.setattr(case.db, "upsert_dictionary_keyword", fail_attribute)
    with pytest.raises(psycopg.errors.NotNullViolation):
        ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, queue)
    assert memory_rows(case) == before
    assert dictionary_state(case) == dictionary_before
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session) == queue
    assert case.db.conn.info.transaction_status == TransactionStatus.IDLE
    assert case.db.get_user_graph_skeleton(case.tenant, case.user)[0]["id"] == before[0]["id"]
    assert case.cache.client.zscore(get_section_key_active_sessions(), generate_session_ref(case.tenant, case.session, case.user)) is not None


@pytest.mark.parametrize("mode,value,step", [("INSERT", "Python", 0), ("REINFORCE", "Python", 1), ("MUTATE", "Rust", 2)])
def test_successful_queue_replay_is_idempotent(mechanics, mode, value, step):
    case = lifecycle(mechanics)
    queue = run_message(case)
    inserted, = memory_rows(case)
    if mode != "INSERT":
        acknowledge(case)
        case.llm.prepare(mode, value, "I still really like Python." if mode == "REINFORCE" else "Actually, I prefer Rust now.", inserted["id"], step)
        queue = run_message(case)
    before, dictionary_before = memory_rows(case), dictionary_state(case)
    markers_before = processed_count(case)
    case.llm.calls = []
    # Simulate a crash after PostgreSQL commit, before Redis acknowledgment.
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session) == queue
    for _ in range(3):
        assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, queue)
    assert case.llm.calls == []
    assert memory_rows(case) == before and dictionary_state(case) == dictionary_before
    assert processed_count(case) == markers_before
    assert len(before) == 1 and before[0]["id"] == inserted["id"]
    assert case.db.conn.info.transaction_status == TransactionStatus.IDLE


@pytest.mark.parametrize("mode", ["REINFORCE", "MUTATE"])
def test_matched_row_with_new_extraction_id_preserves_golden_identity(mechanics, monkeypatch, mode):
    """Temporary extraction IDs must not replace a matched Golden Record ID."""
    case = lifecycle(mechanics)
    run_message(case)
    inserted, = memory_rows(case)
    acknowledge(case)
    value, step = ("Python", 1) if mode == "REINFORCE" else ("Rust", 2)
    case.llm.prepare(mode, value, "I still really like Python." if mode == "REINFORCE" else "Actually, I prefer Rust now.", inserted["id"], step)
    case.llm.entity = case.llm.entity.model_copy(update={"entity_id": "pre_99"})
    commit = case.db.commit_graph_updates
    def verify_identity(operations):
        operation, = operations
        assert operation["mode"] == mode
        assert operation["db_id"] == str(inserted["id"])
        assert operation["entity_id"] == operation["payload"]["entity_id"] == "pre_01"
        return commit(operations)
    monkeypatch.setattr(case.db, "commit_graph_updates", verify_identity)
    run_message(case)
    updated, = memory_rows(case)
    assert updated["id"] == inserted["id"] and updated["entity_id"] == "pre_01"
    assert updated["payload"]["entity_id"] == "pre_01"
    assert language(updated)["value"] == value
    assert inserted["payload"]["facts"][1] in updated["payload"]["facts"]
    assert set(updated["payload"]["aliases"]) == {"my favorite programming language", "my preferred language"}
    for key in ("entity_name", "entity_type", "entity_domain"):
        assert updated["payload"][key] == case.llm.entity.model_dump(mode="json")[key]
    context = ChatHookWorkflow().on_chat(case.tenant, case.session, "What programming language do I prefer?", ChatSenderRole.USER, case.user)["context_text"]
    assert context.count("[[ENTITY: pre_01") == 1 and "[[ENTITY: pre_99" not in context
    assert "favorite_programming_language: " + value in context


def test_insert_uses_extracted_id_as_new_canonical_identity(mechanics, monkeypatch):
    case = lifecycle(mechanics)
    case.llm.entity = case.llm.entity.model_copy(update={"entity_id": "pre_99"})
    commit = case.db.commit_graph_updates
    def verify_identity(operations):
        operation, = operations
        assert operation["mode"] == "INSERT" and "db_id" not in operation
        assert operation["entity_id"] == operation["payload"]["entity_id"] == "pre_99"
        return commit(operations)
    monkeypatch.setattr(case.db, "commit_graph_updates", verify_identity)
    run_message(case)
    inserted, = memory_rows(case)
    assert inserted["entity_id"] == inserted["payload"]["entity_id"] == "pre_99"
    context = ChatHookWorkflow().on_chat(case.tenant, case.session, "What programming language do I prefer?", ChatSenderRole.USER, case.user)["context_text"]
    assert context.count("[[ENTITY: pre_99") == 1


@pytest.mark.parametrize("mode", ["MUTATE", "REINFORCE"])
def test_missing_matched_golden_record_fails_before_persistence(mechanics, monkeypatch, mode):
    case = lifecycle(mechanics)
    run_message(case)
    before, dictionary_before = memory_rows(case), dictionary_state(case)
    acknowledge(case)
    missing_id = uuid4()
    value, step = ("Python", 1) if mode == "REINFORCE" else ("Rust", 2)
    case.llm.prepare(mode, value, "I still really like Python." if mode == "REINFORCE" else "Actually, I prefer Rust now.", missing_id, step)
    parse = case.llm.parse_structured
    def missing_match(system_prompt, user_text, schema, taxonomy=None, lexicon=None):
        if schema is TomteMemorySchema:
            return parse(system_prompt, user_text, schema, taxonomy, lexicon)
        case.llm.record("reconcile")
        return schema.model_validate({"mappings": [{
            "extracted_entity_id": "pre_01", "mode": mode,
            "matched_db_id": str(missing_id), "reasoning": "Synthetic nonexistent match",
        }]})
    monkeypatch.setattr(case.llm, "parse_structured", missing_match)
    ChatHookWorkflow().on_chat(case.tenant, case.session, case.llm.message, ChatSenderRole.USER, case.user)
    queue = case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session)
    with pytest.raises(ValueError, match="Matched Golden Record not found"):
        ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, queue)
    assert case.llm.calls == ["embed_text", "is_junk", "extract", "reconcile"]
    assert memory_rows(case) == before and dictionary_state(case) == dictionary_before
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session) == queue
    assert case.db.conn.info.transaction_status == TransactionStatus.IDLE


def processed_count(case):
    return case.observer.execute("SELECT count(*) FROM tomte_processed_consolidations WHERE tenant_id=%s AND user_id=%s", (case.tenant, case.user)).fetchone()[0]


def test_failed_batch_has_no_marker_and_retry_can_commit(mechanics, monkeypatch):
    case = lifecycle(mechanics)
    run_message(case)
    before, = memory_rows(case)
    acknowledge(case)
    case.llm.prepare("MUTATE", "Rust", "Actually, I prefer Rust now.", before["id"], 2)
    ChatHookWorkflow().on_chat(case.tenant, case.session, case.llm.message, ChatSenderRole.USER, case.user)
    queue = case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session)
    batch_id = ConsolidationWorkflow._batch_id(case.tenant, case.user, queue)
    dictionary_before = dictionary_state(case)
    original = case.db.upsert_dictionary_keyword
    def fail_attribute(anchor, keyword, kind):
        return original(anchor, None if kind == "ATTRIBUTE" else keyword, kind)
    monkeypatch.setattr(case.db, "upsert_dictionary_keyword", fail_attribute)
    with pytest.raises(psycopg.errors.NotNullViolation):
        ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, queue)
    assert not case.db.is_consolidation_processed(case.tenant, case.user, batch_id)
    assert processed_count(case) == 1
    assert memory_rows(case) == [before] and dictionary_state(case) == dictionary_before
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session) == queue
    monkeypatch.setattr(case.db, "upsert_dictionary_keyword", original)
    case.llm.calls = []
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, queue)
    updated, = memory_rows(case)
    assert updated["id"] == before["id"] and language(updated)["value"] == "Rust"
    assert updated["mention_count"] == before["mention_count"] + 1
    assert processed_count(case) == 2
    assert case.llm.calls == ["is_junk", "extract", "reconcile", "embed_batch"]
    assert case.db.conn.info.transaction_status == TransactionStatus.IDLE


def test_later_identical_message_is_a_new_batch_and_survives_old_ack(mechanics):
    case = lifecycle(mechanics)
    first = run_message(case)
    inserted, = memory_rows(case)
    first_id = ConsolidationWorkflow._batch_id(case.tenant, case.user, first)
    # Arrives during the commit/ack crash window: it must not enlarge the retry.
    ChatHookWorkflow().on_chat(case.tenant, case.session, MESSAGE, ChatSenderRole.USER, case.user)
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session) == first
    before, dictionary_before = memory_rows(case), dictionary_state(case)
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, first)
    assert memory_rows(case) == before and dictionary_state(case) == dictionary_before
    assert case.cache.clear_session_4_consolidation(case.tenant, case.user, case.session, first)
    assert case.cache.client.zscore(get_section_key_active_sessions(), generate_session_ref(case.tenant, case.session, case.user)) is not None
    second = case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session)
    assert len(second) == 1 and second[0]["content"] == first[0]["content"]
    assert second[0]["item_id"] != first[0]["item_id"]
    assert ConsolidationWorkflow._batch_id(case.tenant, case.user, second) != first_id
    # A delayed old acknowledgment cannot delete the newly frozen batch.
    assert not case.cache.clear_session_4_consolidation(case.tenant, case.user, case.session, first)
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session) == second
    case.llm.prepare("REINFORCE", "Python", MESSAGE, inserted["id"], 1)
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, second)
    updated, = memory_rows(case)
    assert updated["id"] == inserted["id"] and updated["mention_count"] == 2
    assert processed_count(case) == 2
    acknowledge(case)
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session) == []


def test_legacy_queue_gets_persisted_retry_stable_item_ids(mechanics):
    import json
    case = lifecycle(mechanics)
    key = generate_consolidation_key(case.tenant, case.user, case.session)
    legacy = {"tenant_id": case.tenant, "user_id": case.user, "content": MESSAGE, "role": "user", "vector": case.llm.vector(), "timestamp": 123.0}
    case.cache.client.rpush(key, json.dumps(legacy))
    first = case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session)
    assert first[0]["item_id"] and first[0]["session_id"] == case.session
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session) == first
    for field, value in legacy.items():
        assert first[0][field] == value


@pytest.mark.parametrize("already_processed", [False, True])
def test_direct_worker_acknowledges_new_and_replayed_success(mechanics, monkeypatch, already_processed):
    from tomte.engine.workers.dictionary_maintenance_worker import run_consolidation_job
    case = lifecycle(mechanics)
    if already_processed:
        run_message(case)
        case.llm.calls = []
    else:
        ChatHookWorkflow().on_chat(case.tenant, case.session, MESSAGE, ChatSenderRole.USER, case.user)
    sessions = iter([(case.tenant, case.user, case.session), None])
    monkeypatch.setattr(case.cache, "get_next_idle_session_4_consolidation", lambda **kwargs: next(sessions))
    run_consolidation_job()
    assert len(memory_rows(case)) == 1 and processed_count(case) == 1
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session) == []
    assert case.cache.client.zscore(get_section_key_active_sessions(), generate_session_ref(case.tenant, case.session, case.user)) is None
    assert case.llm.calls == ([] if already_processed else ["embed_text", "is_junk", "extract", "reconcile", "embed_batch"])


@pytest.mark.parametrize("raises", [False, True])
def test_direct_worker_retains_failed_batch(mechanics, monkeypatch, raises):
    from tomte.engine.workers.dictionary_maintenance_worker import run_consolidation_job
    case = lifecycle(mechanics)
    ChatHookWorkflow().on_chat(case.tenant, case.session, MESSAGE, ChatSenderRole.USER, case.user)
    queue = case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session)
    sessions = iter([(case.tenant, case.user, case.session), None])
    monkeypatch.setattr(case.cache, "get_next_idle_session_4_consolidation", lambda **kwargs: next(sessions))
    def failure(*args):
        if raises:
            raise RuntimeError("Controlled consolidation failure")
        return False
    monkeypatch.setattr(ConsolidationWorkflow, "run_consolidation_cycle", failure)
    run_consolidation_job()
    assert memory_rows(case) == [] and processed_count(case) == 0
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session) == queue
    assert case.cache.client.get(f"Tomte:lock:consolidation:{case.tenant}:{case.user}") is None



def test_concurrent_batch_claim_applies_memory_and_feedback_only_once(mechanics, monkeypatch):
    """Race prepared persistence on two independent PostgreSQL connections."""
    from concurrent.futures import ThreadPoolExecutor
    from tomte.engine.dictionary.service import DictionaryService
    case = lifecycle(mechanics)
    queue = queued_message(case)
    batch_id = ConsolidationWorkflow._batch_id(case.tenant, case.user, queue)
    entity = case.llm.entity.model_dump(mode="json")
    operation = {
        "mode": "INSERT", "tenant_id": case.tenant, "user_id": case.user,
        **{key: entity[key] for key in ("entity_id", "entity_name", "entity_domain", "entity_type")},
        "payload": entity, "embedding": case.llm.vector(), "importance": 4,
        "extracted_facts": entity["facts"],
    }
    config = dotenv_values(".env")
    url = os.environ.get("POSTGRES_URL") or config.get("POSTGRES_URL")
    barrier = threading.Barrier(2, timeout=10)
    def attempt():
        provider = PostgresProvider(url)
        workflow = ConsolidationWorkflow.__new__(ConsolidationWorkflow)
        dictionary = DictionaryService.__new__(DictionaryService)
        dictionary.db = provider
        workflow.db, workflow.dict_service = provider, dictionary
        claim = provider.claim_consolidation_batch
        def racing_claim(*args):
            # Both transaction owners enter before either attempts the unique insert.
            barrier.wait()
            return claim(*args)
        provider.claim_consolidation_batch = racing_claim
        try:
            workflow._commit_refined_operations([operation], case.tenant, case.user, batch_id)
            assert provider.conn.info.transaction_status == TransactionStatus.IDLE
        finally:
            provider.close()
    try:
        # Only these controlled test threads run; the daemon remains disabled.
        with monkeypatch.context() as scoped:
            scoped.setattr(threading.Thread, "start", REAL_THREAD_START)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(attempt) for _ in range(2)]
                for future in futures:
                    future.result(timeout=20)
        inserted, = memory_rows(case)
        assert inserted["entity_id"] == "pre_01" and inserted["mention_count"] == 1
        assert processed_count(case) == 1
        matches = case.db.find_dictionary_matches_above_threshold(case.llm.vector(), 0.999)
        assert len(matches) == 1
        hits = {k["keyword"]: k["hits"] for k in case.db.get_anchor_keywords(str(matches[0]["dictionary_id"]))}
        assert hits == {"programming_preference": 1, "favorite_programming_language": 1, "preferred_editor": 1}
        assert case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session) == queue
    finally:
        # The race uses separate providers; capture only its initially unused vector's anchors.
        for match in case.db.find_dictionary_matches_above_threshold(case.llm.vector(), 0.999):
            if match["dictionary_id"] not in case.anchors:
                case.anchors.append(match["dictionary_id"])


@pytest.mark.parametrize("field,value", [("item_id", None), ("tenant_id", "other-tenant"), ("user_id", "other-user"), ("session_id", None)])
def test_batch_identity_rejects_missing_or_wrong_scope_before_model_calls(mechanics, field, value):
    case = lifecycle(mechanics)
    queue = queued_message(case)
    bad = [{**queue[0], field: value}]
    case.llm.calls = []
    with pytest.raises(ValueError, match="consistently scoped"):
        ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, bad)
    assert case.llm.calls == [] and memory_rows(case) == [] and processed_count(case) == 0



def test_processed_marker_survives_user_memory_deletion(mechanics):
    case = lifecycle(mechanics)
    queue = run_message(case)
    dictionary_before = dictionary_state(case)
    case.db.delete_user_memory(case.tenant, case.user)
    assert processed_count(case) == 1 and memory_rows(case) == []
    case.llm.calls = []
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, queue)
    assert case.llm.calls == [] and memory_rows(case) == []
    assert dictionary_state(case) == dictionary_before
