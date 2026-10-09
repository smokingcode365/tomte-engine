"""Real local PostgreSQL checks.

Run with TOMTE_RUN_POSTGRES_TESTS=1 and POSTGRES_URL configured in .env or the
process environment. Requires the existing bootstrapped local Tomte database.
No workers or model APIs are used. Each test removes only its unique user rows.
"""

import os
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import dotenv_values
import psycopg
from psycopg.pq import TransactionStatus
import pytest

from tomte.engine.database.postgre import schema, transactions
from tomte.engine.dictionary.service import DictionaryService
from tomte.engine.providers.postgres_provider import PostgresProvider
from tomte.engine.workflows.consolidation import ConsolidationWorkflow


@pytest.fixture
def local_db():
    if os.environ.get("TOMTE_RUN_POSTGRES_TESTS") != "1":
        pytest.skip("Set TOMTE_RUN_POSTGRES_TESTS=1 to use bootstrapped local PostgreSQL")
    url = os.environ.get("POSTGRES_URL") or dotenv_values(".env").get("POSTGRES_URL")
    if not url or urlsplit(url).hostname not in ("localhost", "127.0.0.1", "::1"):
        pytest.fail("Transaction tests require a local POSTGRES_URL", pytrace=False)
    db = None
    try:
        db = PostgresProvider(url)
        observer = psycopg.connect(url, autocommit=True, options="-c default_transaction_read_only=on")
    except Exception:
        if db is not None:
            db.close()
        pytest.fail("Unable to connect to local PostgreSQL; check bootstrap and configuration", pytrace=False)
    tenant = "transaction-test-" + uuid4().hex
    users = ["user-one", "user-two"]
    def visible(user):
        return observer.execute(
            "SELECT count(*) FROM tomte_user_registry WHERE tenant_id=%s AND user_id=%s",
            (tenant, user),
        ).fetchone()[0]
    try:
        assert db.conn.info.transaction_status == TransactionStatus.IDLE
        yield SimpleNamespace(db=db, observer=observer, tenant=tenant, users=users, visible=visible)
    finally:
        # Business transaction contexts must have completed before fixture cleanup.
        for user in users:
            db.delete_user_memory(tenant, user)
        observer.close()
        db.close()


def idle(db):
    assert db.conn.info.transaction_status == TransactionStatus.IDLE


def test_standalone_reads_and_all_retrieval_methods_leave_idle(local_db):
    db, tenant, user = local_db.db, local_db.tenant, local_db.users[0]
    idle(db)
    for read in (
        lambda: db.get_recent_entities(tenant, user),
        lambda: db.get_frequent_entities(tenant, user),
        lambda: db.get_important_entities(tenant, user),
        lambda: db.semantic_retrieval(tenant, user, [1.0] + [0.0] * 1535),
        lambda: db.get_user_graph_skeleton(tenant, user),
        lambda: db.find_exact_fact(tenant, user, "pref_01", "favorite_language"),
        lambda: db.get_ontology_dictionaries(tenant, [1.0] + [0.0] * 1535),
        lambda: db.get_semantic_keywords([1.0] + [0.0] * 1535, 0.8, 3),
        lambda: db.find_dictionary_matches_above_threshold([1.0] + [0.0] * 1535, 0.8),
        lambda: db.get_dictionary_anchor_details(uuid4()),
        lambda: db.get_anchor_keywords(str(uuid4())),
        db.get_cross_anchor_merge_candidates,
        db.get_anchors_for_internal_merge,
    ):
        read()
        idle(db)


def test_standalone_write_commits_and_failed_select_recovers(local_db):
    db, tenant, user = local_db.db, local_db.tenant, local_db.users[0]
    db.upsert_user_activity(tenant, user)
    idle(db)
    assert local_db.visible(user) == 1
    with pytest.raises(psycopg.Error):
        db.get_recent_entities(tenant, user, limit=-1)
    idle(db)
    assert db.get_recent_entities(tenant, user) == []
    idle(db)


def test_failed_standalone_write_recovers(local_db):
    db = local_db.db
    with pytest.raises(psycopg.Error):
        db.upsert_user_activity(local_db.tenant, None)
    idle(db)
    db.upsert_user_activity(local_db.tenant, local_db.users[0])
    idle(db)
    assert local_db.visible(local_db.users[0]) == 1


def test_explicit_reads_and_writes_commit_together(local_db):
    db, tenant = local_db.db, local_db.tenant
    with db.transaction():
        for user in local_db.users:
            db.upsert_user_activity(tenant, user)
            assert local_db.visible(user) == 0  # Another connection sees no early commit.
        assert db.get_recent_entities(tenant, local_db.users[0]) == []
        assert db.conn.info.transaction_status == TransactionStatus.INTRANS
        assert local_db.visible(local_db.users[0]) == 0
    idle(db)
    assert [local_db.visible(u) for u in local_db.users] == [1, 1]


def test_exception_rolls_back_all_writes_and_connection_is_reusable(local_db):
    db, tenant = local_db.db, local_db.tenant
    with pytest.raises(RuntimeError, match="forced failure"):
        with db.transaction():
            for user in local_db.users:
                db.upsert_user_activity(tenant, user)
            raise RuntimeError("forced failure")
    idle(db)
    assert [local_db.visible(u) for u in local_db.users] == [0, 0]
    db.upsert_user_activity(tenant, local_db.users[0])
    idle(db)
    assert local_db.visible(local_db.users[0]) == 1


def test_failed_sql_in_outer_scope_rolls_back_and_recovers(local_db):
    db, tenant, user = local_db.db, local_db.tenant, local_db.users[0]
    with pytest.raises(psycopg.Error):
        with db.transaction():
            db.upsert_user_activity(tenant, user)
            db.get_recent_entities(tenant, user, limit=-1)
    idle(db)
    assert local_db.visible(user) == 0
    db.upsert_user_activity(tenant, user)
    assert local_db.visible(user) == 1
    idle(db)


def test_nested_scope_joins_without_commit(local_db):
    db, tenant = local_db.db, local_db.tenant
    with db.transaction():
        db.upsert_user_activity(tenant, local_db.users[0])
        with db.transaction():
            db.upsert_user_activity(tenant, local_db.users[1])
        assert [local_db.visible(u) for u in local_db.users] == [0, 0]
        assert db.conn.info.transaction_status == TransactionStatus.INTRANS
    assert [local_db.visible(u) for u in local_db.users] == [1, 1]
    idle(db)


def test_caught_nested_failure_aborts_whole_scope(local_db):
    db, tenant, user = local_db.db, local_db.tenant, local_db.users[0]
    with pytest.raises(transactions.TransactionScopeError, match="entire transaction"):
        with db.transaction():
            db.upsert_user_activity(tenant, user)
            try:
                with db.transaction():
                    raise ValueError("synthetic child failure")
            except ValueError:
                pass
    assert local_db.visible(user) == 0
    idle(db)


def test_caught_failed_sql_does_not_silently_commit(local_db):
    db, tenant, user = local_db.db, local_db.tenant, local_db.users[0]
    with pytest.raises(transactions.TransactionScopeError, match="entire transaction"):
        with db.transaction():
            db.upsert_user_activity(tenant, user)
            try:
                db.get_recent_entities(tenant, user, limit=-1)
            except psycopg.Error:
                assert db.conn.info.transaction_status == TransactionStatus.INERROR
    assert local_db.visible(user) == 0
    idle(db)


def memory_update(tenant, user):
    entity = {"entity_id": "pref_01", "entity_name": "Synthetic preference", "entity_domain": "preferences_and_traits", "entity_type": "preference", "aliases": [], "facts": []}
    return {"mode": "INSERT", "tenant_id": tenant, "user_id": user, **entity, "payload": entity, "embedding": [1.0] + [0.0] * 1535, "extracted_facts": [], "importance": 3}


@pytest.mark.parametrize("method", ["get_important_entities", "get_recent_entities", "get_frequent_entities", "semantic_retrieval"])
def test_caught_normalization_failure_rolls_back_earlier_writes(local_db, method):
    db, tenant, user = local_db.db, local_db.tenant, local_db.users[0]
    malformed = {**memory_update(tenant, user), "payload": {}}
    with pytest.raises(transactions.TransactionScopeError, match="entire transaction"):
        with db.transaction():
            db.upsert_user_activity(tenant, user)
            assert db.commit_graph_updates([malformed])
            assert local_db.visible(user) == 0
            arguments = (tenant, user, [1.0] + [0.0] * 1535) if method == "semantic_retrieval" else (tenant, user)
            # The SQL read succeeds; provider normalization fails and is caught.
            with pytest.raises(ValueError, match="missing a nonempty entity_id"):
                getattr(db, method)(*arguments)
    assert local_db.visible(user) == 0
    assert local_db.observer.execute(
        "SELECT count(*) FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s", (tenant, user)
    ).fetchone()[0] == 0
    assert db.get_recent_entities(tenant, user) == []
    idle(db)


def test_bulk_memory_operations_join_outer_transaction(local_db):
    db, tenant, user = local_db.db, local_db.tenant, local_db.users[0]
    op = memory_update(tenant, user)
    with pytest.raises(RuntimeError, match="forced failure"):
        with db.transaction():
            assert db.commit_graph_updates([op])
            assert db.get_recent_entities(tenant, user)[0]["entity_id"] == "pref_01"
            assert local_db.observer.execute("SELECT count(*) FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s", (tenant, user)).fetchone()[0] == 0
            raise RuntimeError("forced failure")
    assert db.get_recent_entities(tenant, user) == []
    idle(db)


def test_standalone_bulk_write_commits_and_failed_batch_rolls_back(local_db):
    db, tenant, user = local_db.db, local_db.tenant, local_db.users[0]
    op = memory_update(tenant, user)
    assert db.commit_graph_updates([op])
    idle(db)
    assert local_db.observer.execute(
        "SELECT count(*) FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s", (tenant, user)
    ).fetchone()[0] == 1
    good = memory_update(tenant, local_db.users[1])
    bad = {**good, "entity_id": "pref_02", "embedding": [1.0]}
    assert db.commit_graph_updates([good, bad]) is False
    idle(db)
    assert db.get_recent_entities(tenant, local_db.users[1]) == []
    idle(db)


def test_failed_bulk_write_in_outer_scope_propagates_and_rolls_back(local_db):
    db, tenant, user = local_db.db, local_db.tenant, local_db.users[0]
    good = memory_update(tenant, user)
    bad = {**good, "entity_id": "pref_02", "embedding": [1.0]}
    with pytest.raises(psycopg.Error):
        with db.transaction():
            db.upsert_user_activity(tenant, user)
            db.commit_graph_updates([good, bad])
    idle(db)
    assert local_db.visible(user) == 0
    assert db.get_recent_entities(tenant, user) == []
    idle(db)


def test_dictionary_feedback_failure_rolls_back_consolidation_memory(local_db, monkeypatch):
    db, tenant, user = local_db.db, local_db.tenant, local_db.users[0]
    service = DictionaryService.__new__(DictionaryService)
    service.db = db
    anchor_ids = []
    create = db.create_dictionary_anchor
    def recording_create(vector):
        anchor = create(vector)
        anchor_ids.append(anchor)
        return anchor
    monkeypatch.setattr(db, "create_dictionary_anchor", recording_create)
    # Guarantee fixture feedback uses a new anchor, without altering existing ones.
    monkeypatch.setattr(db, "find_dictionary_matches_above_threshold", lambda *args: [])
    def failed_keyword(*args):
        raise RuntimeError("forced feedback failure")
    monkeypatch.setattr(db, "upsert_dictionary_keyword", failed_keyword)
    workflow = ConsolidationWorkflow.__new__(ConsolidationWorkflow)
    workflow.db, workflow.dict_service = db, service
    with pytest.raises(RuntimeError, match="forced feedback failure"):
        workflow._commit_refined_operations([memory_update(tenant, user)])
    assert anchor_ids
    assert db.get_recent_entities(tenant, user) == []
    assert all(db.get_dictionary_anchor_details(a) is None for a in anchor_ids)
    idle(db)


def test_internal_merges_and_cooldown_are_atomic(local_db, monkeypatch):
    from tomte.engine.database.postgre import dictionary_ops
    db = local_db.db
    anchor_ids = []
    with pytest.raises(RuntimeError, match="forced maintenance failure"):
        with db.transaction():
            anchor = db.create_dictionary_anchor([1.0] + [0.0] * 1535)
            anchor_ids.append(anchor)
            db.upsert_dictionary_keyword(anchor, "Python", "ENTITY")
            def fail_marker(*args):
                raise RuntimeError("forced maintenance failure")
            monkeypatch.setattr(dictionary_ops, "mark_anchor_internal_merge_complete", fail_marker)
            db.apply_internal_merges(str(anchor), [{"kw_type": "ENTITY", "survivor": "Python", "victims": [], "total_hits": 8}])
    assert db.get_dictionary_anchor_details(anchor_ids[0]) is None
    idle(db)


def test_schema_children_join_one_bootstrap_transaction(local_db, monkeypatch):
    def core(conn, dimension):
        local_db.db.upsert_user_activity(local_db.tenant, local_db.users[0])
    def dictionary(conn, dimension):
        assert local_db.visible(local_db.users[0]) == 0
        raise RuntimeError("forced schema failure")
    monkeypatch.setattr(schema, "ensure_core_infrastructure", core)
    monkeypatch.setattr(schema, "ensure_dictionary_infrastructure", dictionary)
    with pytest.raises(RuntimeError, match="forced schema failure"):
        local_db.db.ensure_infrastructure(1536)
    assert local_db.visible(local_db.users[0]) == 0
    idle(local_db.db)


def test_unmanaged_transaction_is_rejected_without_committing(local_db):
    db = local_db.db
    with db.conn.transaction():
        with pytest.raises(transactions.TransactionScopeError, match="Unmanaged transaction"):
            db.get_recent_entities(local_db.tenant, local_db.users[0])
        assert db.conn.info.transaction_status == TransactionStatus.INTRANS
    idle(db)
