from resolution_helpers import resolution_response, mutation_aware
"""Deterministic contract regressions; opt-in real local services, no model API calls."""
from copy import deepcopy
import json
from types import SimpleNamespace
from uuid import uuid4

import psycopg
import pytest

from tomte.engine import Tomte
from tomte.engine.core.extraction import ExtractionMemorySchema, source_timestamps
from tomte.engine.dictionary.service import DictionaryService
from tomte.engine.dictionary.terminology import normalize_facts, merge_facts
from tomte.engine.utils.cognitive import CognitiveProcessor
from tomte.engine.workflows.consolidation import ConsolidationWorkflow
from test_postgres_transactions import local_db, memory_update
from test_consolidation_mechanics import mechanics, memory_rows


def fact(key, value, source="old-source", stamp="2026-10-09T12:00:00Z"):
    return {"attribute_name": key, "value": value, "status": "active", "importance": 4,
            "source_chat_id": source, "last_mentioned_at": stamp, "timeline": []}


@pytest.mark.parametrize("alias", ["ownership", "ownership_status", "vehicle_ownership"])
def test_state_synonyms_replace_current_state_and_preserve_real_provenance(alias):
    old = fact(alias, "Owned")
    new = fact("ownership_status", "Sold", "sale-source", "2026-10-09T12:01:00Z")
    result, = merge_facts([old], [new])
    assert result["attribute_name"] == "ownership_status" and result["value"] == "Sold"
    assert result["source_chat_id"] == "sale-source"
    assert result["timeline"] == [{"event_description": "Previous ownership_status: Owned",
                                 "source_chat_id": "old-source", "timestamp": old["last_mentioned_at"]}]
    assert old["attribute_name"] == alias and old["timeline"] == []  # No input mutation.


def test_same_state_synonyms_do_not_duplicate_or_fabricate_history():
    old = fact("ownership_status", "Sold")
    new = fact("ownership", "Sold", "repeat-source")
    result, = merge_facts([old], [new])
    assert result["value"] == "Sold" and result["timeline"] == []
    assert result["source_chat_id"] == "repeat-source"


def test_unrelated_attributes_and_future_state_are_not_collapsed():
    future = fact("ownership", "Will buy")
    future["status"] = "future"
    result = merge_facts([fact("paint_color", "red")], [future])
    assert {f["attribute_name"] for f in result} == {"paint_color", "ownership_status"}
    assert result[1]["status"] == "future"


def test_sale_event_status_does_not_make_current_ownership_past():
    incoming = fact("ownership", "Sold")
    incoming["status"] = "past"
    assert normalize_facts([incoming])[0]["status"] == "active"
    incoming["value"] = "Owned"
    assert normalize_facts([incoming])[0]["status"] == "past"  # Former ownership is not current.


def test_conflicting_synonyms_fail_instead_of_choosing_arbitrary_current_state():
    with pytest.raises(ValueError, match="Conflicting synonymous"):
        normalize_facts([fact("ownership", "Owned"), fact("ownership_status", "Sold")])


def test_context_preserves_distinct_db_rows_with_corrupt_logical_id():
    first = {"id": uuid4(), "entity_id": "veh_01", "entity_name": "Tesla", "facts": []}
    second = {"id": uuid4(), "entity_id": "veh_01", "entity_name": "Bicycle", "facts": []}
    processor = CognitiveProcessor()
    assert processor.merge_entities([first, second], [deepcopy(first)]) == [first, second]
    assert processor.merge_entities([{ "entity_id": "flat"}], [{"entity_id": "flat", "facts": [1]}]) == [{"entity_id": "flat", "facts": [1]}]


def test_insert_collision_gets_unique_canonical_ids_and_both_retrieve(local_db):
    db, tenant, user = local_db.db, local_db.tenant, local_db.users[0]
    first, second = memory_update(tenant, user), memory_update(tenant, user)
    for op in (first, second):
        op["entity_id"] = op["payload"]["entity_id"] = "veh_01"
    second["entity_name"] = second["payload"]["entity_name"] = "Unrelated bicycle"
    assert db.commit_graph_updates([first, second])
    assert first["entity_id"] == "veh_01" and second["entity_id"] != "veh_01"
    rows = db.get_recent_entities(tenant, user)
    assert len(rows) == len({r["id"] for r in rows}) == len({r["entity_id"] for r in rows}) == 2
    assert {r["entity_id"] for r in db.semantic_retrieval(tenant, user, first["embedding"])} == {first["entity_id"], second["entity_id"]}
    assert len(CognitiveProcessor().merge_entities(rows, db.get_frequent_entities(tenant, user))) == 2
    assert all(r["entity_id"] == r["payload"]["entity_id"] for r in db.get_user_graph_skeleton(tenant, user))
    # Identical labels in another user namespace are allowed.
    other = memory_update(tenant, local_db.users[1])
    other["entity_id"] = other["payload"]["entity_id"] = "veh_01"
    assert db.commit_graph_updates([other]) and other["entity_id"] == "veh_01"


def test_unique_db_index_rejects_bypass_and_rolls_back_earlier_write(local_db):
    db, tenant, user = local_db.db, local_db.tenant, local_db.users[0]
    db.ensure_infrastructure(1536)
    assert db.commit_graph_updates([memory_update(tenant, user)])
    from tomte.engine.database.postgre.transactions import TransactionScopeError
    with pytest.raises(TransactionScopeError):
        with db.transaction():
            db.upsert_user_activity(tenant, user)
            try:
                with db.transaction():
                    db.conn.execute("""INSERT INTO tomte_memory_engine
                        (tenant_id,user_id,entity_id,entity_name,entity_domain,entity_type,payload)
                        SELECT tenant_id,user_id,entity_id,entity_name,entity_domain,entity_type,payload
                        FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s""", (tenant,user))
            except psycopg.errors.UniqueViolation:
                pass
    assert local_db.visible(user) == 0
    assert len(db.get_recent_entities(tenant, user)) == 1


@pytest.mark.parametrize("value,expected", [("Sold", "REINFORCE"), ("Owned", "MUTATE")])
def test_reconciliation_sees_canonical_state_keys(monkeypatch, value, expected):
    db_id = uuid4()
    old = {"id": db_id, "entity_id": "veh_01", "entity_name": "Tesla", "entity_type": "vehicle",
           "entity_domain": "assets_and_possessions", "payload": {"aliases": ["Tesla"], "facts": [fact("ownership_status", "Sold")]}}
    incoming = {"entity_id": "veh_99", "entity_name": "Tesla", "entity_type": "vehicle",
                "entity_domain": "assets_and_possessions", "facts": [fact("ownership", value)]}
    def model(system_prompt, user_text, schema):
        data = json.loads(user_text)
        extracted = data["EXTRACTED_SEMANTIC_INFORMATION"][0]
        candidate = data["CANDIDATES_BY_OBJECT"][extracted["id"]][0]
        assert set(extracted["facts"]) == set(candidate["facts"]) == {"ownership_status"}
        mode = "REINFORCE" if extracted["facts"] == candidate["facts"] else "MUTATE"
        return resolution_response(schema, {"mappings": [{"extracted_entity_id": "veh_99", "mode": mode,
                     "matched_db_id": str(db_id), "reasoning": "Deterministic supplied values"}]}, user_text)
    monkeypatch.setattr(Tomte, "_db", SimpleNamespace())
    monkeypatch.setattr(Tomte, "_llm", SimpleNamespace(parse_structured=model))
    monkeypatch.setattr("tomte.engine.workflows.consolidation.get_dictionary_service", lambda: None)
    op, = ConsolidationWorkflow()._reconcile_with_golden_records("tenant","user",[incoming],[old])
    assert op["mode"] == expected and op["entity_id"] == op["payload"]["entity_id"] == "veh_01"
    assert len(op["payload"]["facts"]) == 1


def test_weak_nickname_reconciliation_gets_bounded_antecedents_and_persists_one_record(mechanics, monkeypatch):
    case = mechanics
    seed = memory_update(case.tenant,case.user)
    seed["entity_id"] = seed["payload"]["entity_id"] = "veh_01"
    seed["entity_name"] = seed["payload"]["entity_name"] = "Tesla Model Y"
    seed["entity_type"] = seed["payload"]["entity_type"] = "vehicle"
    assert case.db.commit_graph_updates([seed])
    skeleton = case.db.get_user_graph_skeleton(case.tenant,case.user)
    original_id = skeleton[0]["id"]
    reference = [{"role":"user","content":"old ignored"}]*4 + [
        {"role":"user","content":"I bought a Tesla Model Y."},
        {"role":"user","content":"I painted it red."}]
    def model(system_prompt,user_text,schema,taxonomy=None,lexicon=None):
        if schema is ExtractionMemorySchema:
            data = json.loads(user_text)
            assert [m["content"] for m in data["CURRENT_BATCH"]] == ["I named it Little Dog."]
            assert data["RECENT_CONTEXT_REFERENCE_ONLY"] == reference[-4:]
            return schema.model_validate({"assets_and_possessions":[{
                "entity_id":"veh_99","entity_name":"Little Dog","entity_type":"vehicle",
                "entity_domain":"assets_and_possessions","aliases":["Little Dog"],
                "facts":[{"attribute_name":"nickname","value":"Little Dog","confidence":1}]}]})
        data = json.loads(user_text)
        context = data["REFERENCE_ONLY_CONTEXT"]
        assert len(context) == 4 and context[-2:] == reference[-2:]
        assert "identity only" in system_prompt and "extract historical facts as new updates" in system_prompt
        new = data["EXTRACTED_SEMANTIC_INFORMATION"][0]
        candidate = data["CANDIDATES_BY_OBJECT"][new["id"]][0]
        assert new["name"] == "Little Dog" and new["facts"] == {"nickname":"Little Dog"}
        assert candidate["name"] == "Tesla Model Y"
        # Local policy consumes supplied antecedents; this is not proof of model quality.
        assert context[-1]["content"] == "I painted it red."
        return resolution_response(schema, {"mappings":[{"extracted_entity_id":new["id"],"mode":"MUTATE",
            "matched_db_id":candidate["id"],"reasoning":"Reference resolves it to the Tesla"}]}, user_text)
    monkeypatch.setattr(case.llm,"parse_structured",mutation_aware(model))
    monkeypatch.setattr(case.llm,"embed_text",lambda text:case.llm.vector())
    monkeypatch.setattr(case.llm,"embed_batch",lambda texts:[case.llm.vector() for _ in texts])
    monkeypatch.setattr(case.llm,"is_junk",lambda text:False)
    for message in reference:
        case.cache.append_chat_history(case.tenant,case.session,message["content"],"user",case.user)
    from tomte.engine.workflows.chat_hook import ChatHookWorkflow
    from tomte.engine.core.enums import ChatSenderRole
    ChatHookWorkflow().on_chat(case.tenant,case.session,"I named it Little Dog.",ChatSenderRole.USER,case.user)
    queue=case.cache.pop_session_chat_log_4_consolidation(case.tenant,case.user,case.session)
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant,case.user,queue)
    row,=memory_rows(case)
    assert row["id"] == original_id and row["entity_id"] == "veh_01"
    nickname = next(f for f in row["payload"]["facts"] if f["attribute_name"] == "nickname")
    assert nickname["value"] == "Little Dog" and nickname["source_chat_id"] == queue[0]["item_id"]
    assert all(f["attribute_name"] != "paint_color" for f in row["payload"]["facts"])


def test_same_batch_late_alias_contract_preserves_sources_and_one_vehicle(mechanics, monkeypatch):
    case=mechanics
    messages=["I bought a Tesla.","I painted Little Dog red.","Oh, I named the Tesla Little Dog."]
    for text in messages:
        case.cache.enqueue_for_consolidation(case.tenant,case.user,case.session,text,"user",case.llm.vector())
    queue=case.cache.pop_session_chat_log_4_consolidation(case.tenant,case.user,case.session)
    def model(system_prompt,user_text,schema,taxonomy=None,lexicon=None):
        if schema is ExtractionMemorySchema:
            batch=json.loads(user_text)["CURRENT_BATCH"]
            assert [x["content"] for x in batch] == messages  # Later declaration actually reaches extraction.
            return schema.model_validate({"assets_and_possessions":[{"entity_id":"veh_01",
                "entity_name":"Tesla","entity_type":"vehicle","entity_domain":"assets_and_possessions",
                "aliases":["Little Dog"],"facts":[
                {"attribute_name":"vehicle_make","value":"Tesla","confidence":1,"source_chat_id":batch[0]["item_id"]},
                {"attribute_name":"paint_color","value":"red","confidence":1,"source_chat_id":batch[1]["item_id"]},
                {"attribute_name":"nickname","value":"Little Dog","confidence":1,"source_chat_id":batch[2]["item_id"]}]}]})
        return resolution_response(schema, {"mappings":[{"extracted_entity_id":"veh_01","mode":"INSERT",
                                    "reasoning":"Prescribed same-batch resolved entity"}]}, user_text)
    monkeypatch.setattr(case.llm,"is_junk",lambda text:False)
    monkeypatch.setattr(case.llm,"parse_structured",mutation_aware(model))
    monkeypatch.setattr(case.llm,"embed_batch",lambda texts:[case.llm.vector() for text in texts])
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant,case.user,queue)
    row,=memory_rows(case)
    assert row["payload"]["aliases"] == ["Little Dog"]
    timestamps=source_timestamps(queue)
    for f in row["payload"]["facts"]:
        assert f["last_mentioned_at"] == timestamps[f["source_chat_id"]]
    assert [f["value"] for f in row["payload"]["facts"]] == ["Tesla","red","Little Dog"]
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant,case.user,queue)
    assert len(memory_rows(case)) == 1


def test_dictionary_feedback_terms_return_when_vectors_match(mechanics):
    case=mechanics
    service=DictionaryService()
    op=memory_update(case.tenant,case.user)
    op["entity_type"]="vehicle"
    op["embedding"]=case.llm.vector()
    op["extracted_facts"]=[fact("ownership_status","Owned")]
    op["_dictionary_evidence"] = [
        {"keyword": word, "keyword_type": kind, "vector": case.llm.vector(), "dictionary_id": None}
        for word, kind in [("vehicle", "ENTITY"), ("ownership_status", "ATTRIBUTE")]]
    service.apply_feedback([op])
    # Only explicit accepted candidate identities may update these anchors.
    for evidence, anchor in zip(op["_dictionary_evidence"], case.anchors):
        evidence["dictionary_id"] = str(anchor)
    from tomte.engine.core.enums import DictionaryKeywordType
    first=service.bootstrap([case.llm.vector()])
    assert first[DictionaryKeywordType.ENTITY] == ["vehicle"]
    assert first[DictionaryKeywordType.ATTRIBUTE] == ["ownership_status"]
    service.apply_feedback([op])
    assert {k["hits"] for k in case.db.get_anchor_keywords(str(case.anchors[0]))} == {2}
    orthogonal=[float(i==(case.llm.index+1)%1536) for i in range(1536)]
    assert all(not terms for terms in service.bootstrap([orthogonal]).values())


def test_concurrent_insert_allocation_serializes_without_duplicate_ids(local_db):
    import os
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from dotenv import dotenv_values
    from tomte.engine.providers.postgres_provider import PostgresProvider
    url = os.environ.get("POSTGRES_URL") or dotenv_values(".env").get("POSTGRES_URL")
    barrier = Barrier(2)
    def insert():
        db = PostgresProvider(url)
        try:
            op = memory_update(local_db.tenant,local_db.users[0])
            op["entity_id"] = op["payload"]["entity_id"] = "veh_01"
            barrier.wait(timeout=10)
            assert db.commit_graph_updates([op])
            return op["entity_id"]
        finally:
            db.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(lambda _:insert(),range(2)))
    assert len(set(ids)) == 2 and "veh_01" in ids
    assert len(local_db.db.get_recent_entities(local_db.tenant,local_db.users[0])) == 2
