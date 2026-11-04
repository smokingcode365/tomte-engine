"""Real local SQL invariants with unique namespaces; no external model calls."""
from copy import deepcopy
from uuid import uuid4
import pytest

from tomte.engine.core.assertions import apply_mutations
from tomte.engine.database.postgre.operations import bulk_commit_updates
from tomte.engine.llm.embedder import changed_search_entries
from tomte.engine.workers.control import StaleSnapshot
from test_postgres_transactions import local_db
from test_memory_retrieval import entity
from test_memory_values import CHAT, fields, NOW


def vector(index):
    return [float(i == index) for i in range(1536)]


def operation(case, payload=None, old=None, user=None):
    payload = deepcopy(payload or entity())
    if old is None:
        payload["entity_id"] = str(uuid4())
    op = {"tenant_id": case.tenant, "user_id": user or case.users[0], "payload": payload,
          **{k: payload[k] for k in ("entity_id", "entity_name", "entity_domain", "entity_type")},
          "mode": "INSERT" if old is None else "MUTATE", "embedding": vector(0)}
    op["reserved_id" if old is None else "db_id"] = payload["entity_id"]
    if old is not None:
        op["_expected_payload"] = deepcopy(old)
    op["search_entries"] = [{**{k: v for k, v in entry.items() if k != "text"},
                             "embedding": vector(1 if entry["history_index"] >= 0 else 0)}
                            for entry in changed_search_entries(payload, old)]
    return op


@pytest.fixture
def ready(local_db):
    # Additive schema upgrade; no old memory is cleared by fixture bootstrap.
    local_db.db.ensure_infrastructure(1536)
    return local_db


def test_canonical_payload_and_history_search_commit_atomically_and_are_scoped(ready):
    case = ready
    op = operation(case)
    with case.db.transaction():
        assert case.db.claim_consolidation_batch(case.tenant, case.users[0], "new-contract")
        assert case.db.commit_graph_updates([op])
        assert case.observer.execute("SELECT count(*) FROM tomte_memory_engine WHERE tenant_id=%s", (case.tenant,)).fetchone()[0] == 0
    result = case.db.get_raw_semantic_context(case.tenant, case.users[0], vector(1), 5)
    assert len(result) == 1 and result[0]["entity_id"] == op["entity_id"]
    assert result[0]["raw_cosine"] == pytest.approx(1)
    assert result[0]["facts"] == op["payload"]["facts"]
    history = case.db.get_history_candidates(case.tenant, case.users[0], [op["entity_id"]], vector(1), 5)
    assert len(history) == 1 and history[0]["history_index"] == 0
    assert case.db.get_raw_semantic_context(case.tenant, case.users[1], vector(1), 5) == []
    assert case.db.get_history_candidates(case.tenant, case.users[1], [op["entity_id"]], vector(1), 5) == []
    assert case.db.get_raw_semantic_context(case.tenant + "-other", case.users[0], vector(1), 5) == []
    assert case.db.is_consolidation_processed(case.tenant, case.users[0], "new-contract")


def test_mention_only_write_keeps_search_vectors_and_original_message_time(ready):
    op = operation(ready)
    assert ready.db.commit_graph_updates([op])
    old = deepcopy(op["payload"])
    updated = deepcopy(old)
    updated["facts"] = apply_mutations(old["facts"], [{"operation": "MENTION", "target_fact_id": old["facts"][0]["fact_id"],
        "source_chat_ids": ["m1"]}], CHAT, old["entity_id"], "mention", recorded_at=NOW)
    write = operation(ready, updated, old)
    write["mode"] = "REINFORCE"
    del write["embedding"]
    del write["search_entries"]
    assert ready.db.commit_graph_updates([write])
    row = ready.db.get_recent_entities(ready.tenant, ready.users[0])[0]
    assert row["facts"][0]["_mention_count"] == old["facts"][0]["_mention_count"] + 1
    assert row["facts"][0]["value"] == old["facts"][0]["value"]
    assert ready.db.get_raw_semantic_context(ready.tenant, ready.users[0], vector(1), 1)[0]["raw_cosine"] == pytest.approx(1)


def test_stale_payload_rolls_back_batch_marker_without_overwriting(ready):
    op = operation(ready)
    assert ready.db.commit_graph_updates([op])
    stale = operation(ready, op["payload"], op["payload"])
    newer = deepcopy(op["payload"])
    newer["facts"][0]["_mention_count"] += 1
    assert ready.db.commit_graph_updates([operation(ready, newer, op["payload"])])
    with pytest.raises(StaleSnapshot):
        with ready.db.transaction():
            ready.db.claim_consolidation_batch(ready.tenant, ready.users[0], "stale")
            ready.db.commit_graph_updates([stale])
    assert not ready.db.is_consolidation_processed(ready.tenant, ready.users[0], "stale")
    assert ready.db.get_recent_entities(ready.tenant, ready.users[0])[0]["facts"] == newer["facts"]


@pytest.mark.parametrize("defect", ["uuid", "history", "missing_index", "wrong_owner"])
def test_storage_rejects_invalid_writes_and_rolls_back_prior_work(ready, defect):
    op = operation(ready)
    assert ready.db.commit_graph_updates([op])
    changed = deepcopy(op["payload"])
    changed["facts"][0]["value"]["employer_name"]["text"] = "Birch Labs"
    write = operation(ready, changed, op["payload"])
    if defect == "uuid":
        write["payload"]["entity_id"] = str(uuid4())
    elif defect == "history":
        write["payload"]["facts"][0]["history_values"][0]["value"]["employer_name"]["text"] = "invented"
    elif defect == "missing_index":
        write["search_entries"] = []
    else:
        write["user_id"] = ready.users[1]
    with pytest.raises((ValueError, StaleSnapshot)):
        with ready.db.transaction():
            ready.db.upsert_user_activity(ready.tenant, ready.users[1])
            ready.db.commit_graph_updates([write])
    assert ready.visible(ready.users[1]) == 0
    assert ready.db.get_recent_entities(ready.tenant, ready.users[0])[0]["facts"] == op["payload"]["facts"]


def test_delta_search_updates_preserve_history_and_remove_obsolete_current_fields(ready):
    op = operation(ready)
    assert ready.db.commit_graph_updates([op])
    changed = deepcopy(op["payload"])
    fact = changed["facts"][0]
    fact["value"].pop("job_title")
    fact["value"]["employer_name"]["text"] = "Birch Labs"
    write = operation(ready, changed, op["payload"])
    assert len(write["search_entries"]) == 1
    assert ready.db.commit_graph_updates([write])
    keys = ready.observer.execute("SELECT history_index,field_name FROM tomte_memory_search WHERE entity_uuid=%s", (op["entity_id"],)).fetchall()
    assert (-1, "job_title") not in keys and (0, "job_title") in keys
    assert ready.db.get_raw_semantic_context(ready.tenant, ready.users[0], vector(1), 1)[0]["raw_cosine"] == pytest.approx(1)
