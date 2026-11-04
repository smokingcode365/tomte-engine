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
from memory_fixtures import simple_entity, simple_fact, prepare_write, field_entries, fact_text, value as item_value
from tomte.engine.core.assertions import validate_facts
from tomte.engine.utils.cognitive import CognitiveProcessor
from tomte.engine.workflows.consolidation import ConsolidationWorkflow
from test_postgres_transactions import local_db, memory_update
from test_consolidation_mechanics import mechanics, memory_rows


def fact(key,value,source="old-source"):
    return {"attribute_name":key,"value":item_value(value),"source_chat_id":source,"importance":4}


@pytest.mark.parametrize("attribute",["ownership","ownership_status","vehicle_ownership","所有权"])
def test_python_validation_preserves_attribute_language_without_synonym_rules(attribute):
    source=simple_fact(attribute,"No longer owned; sold last year")
    assert validate_facts([source])==[source]


def test_separate_facts_with_same_attribute_are_not_collapsed():
    first=simple_fact("employment","Doctor at Cedar Hospital")
    second=simple_fact("employment","Lecturer at Pine University")
    assert validate_facts([first,second])==[first,second]


def test_context_preserves_distinct_db_rows_with_corrupt_logical_id():
    first = {"id": uuid4(), "entity_id": "veh_01", "entity_name": "Tesla", "facts": []}
    second = {"id": uuid4(), "entity_id": "veh_01", "entity_name": "Bicycle", "facts": []}
    processor = CognitiveProcessor()
    assert processor.merge_entities([first, second], [deepcopy(first)]) == [first, second]
    assert processor.merge_entities([{ "entity_id": "flat"}], [{"entity_id": "flat", "facts": [1]}]) == [{"entity_id": "flat", "facts": [1]}]


def test_insert_uuid_collision_rejects_instead_of_reassigning_identity(local_db):
    db,tenant,user=local_db.db,local_db.tenant,local_db.users[0]
    first=memory_update(tenant,user);second=deepcopy(first)
    second["payload"]["entity_name"]="Unrelated bicycle";second["entity_name"]="Unrelated bicycle"
    second=prepare_write(tenant,user,second["payload"])
    with pytest.raises(ValueError,match="Duplicate entity"):
        with db.transaction():db.commit_graph_updates([first,second])
    assert db.get_recent_entities(tenant,user)==[]
    assert db.commit_graph_updates([first])
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db.transaction():db.commit_graph_updates([second])
    rows=db.get_recent_entities(tenant,user)
    assert len(rows)==1 and rows[0]["entity_id"]==first["entity_id"]
    assert rows[0]["entity_name"]==first["entity_name"]
    other=memory_update(tenant,local_db.users[1])
    assert db.commit_graph_updates([other]) and other["entity_id"]!=first["entity_id"]


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
                        (id,tenant_id,user_id,entity_id,entity_name,entity_domain,entity_type,payload)
                        SELECT id,tenant_id,user_id,entity_id,entity_name,entity_domain,entity_type,payload
                        FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s""", (tenant,user))
            except psycopg.errors.UniqueViolation:
                pass
    assert local_db.visible(user) == 0
    assert len(db.get_recent_entities(tenant, user)) == 1


@pytest.mark.parametrize("attribute",["ownership", "ownership_status"])
def test_identity_reconciliation_receives_model_selected_keys_without_python_canonicalization(monkeypatch,attribute):
    payload=simple_entity("Tesla",kind="vehicle",domain="assets_and_possessions",aliases=["Tesla"],
                          facts=[simple_fact("ownership_status","Sold")])
    old={"id":payload["entity_id"],**{k:payload[k] for k in ("entity_id","entity_name","entity_type","entity_domain")},"payload":payload}
    incoming={"entity_id":"tmp_veh_99","entity_name":"Tesla","entity_type":"vehicle",
              "entity_domain":"assets_and_possessions","facts":[fact(attribute,"Sold")]}
    def model(system_prompt,user_text,schema):
        data=json.loads(user_text);extracted=data["EXTRACTED_SEMANTIC_INFORMATION"][0]
        candidate=data["CANDIDATES_BY_OBJECT"][extracted["id"]][0]
        assert extracted["facts"][0]["attribute_name"]==attribute
        assert candidate["facts"][0]["attribute_name"]=="ownership_status"
        return resolution_response(schema,{"mappings":[{"extracted_entity_id":"tmp_veh_99","mode":"MUTATE",
            "matched_db_id":old["id"],"reasoning":"Prescribed identity"}]},user_text)
    monkeypatch.setattr(Tomte,"_db",SimpleNamespace());monkeypatch.setattr(Tomte,"_llm",SimpleNamespace(parse_structured=model))
    monkeypatch.setattr("tomte.engine.workflows.consolidation.get_dictionary_service",lambda:None)
    op,=ConsolidationWorkflow()._reconcile_with_golden_records("tenant","user",[incoming],[old])
    assert op["entity_id"]==payload["entity_id"] and op["payload"]["facts"]==payload["facts"]


def test_weak_nickname_reconciliation_gets_bounded_antecedents_and_persists_one_record(mechanics, monkeypatch):
    case = mechanics
    seed = memory_update(case.tenant,case.user)
    seed["entity_name"] = seed["payload"]["entity_name"] = "Tesla Model Y"
    seed["entity_type"] = seed["payload"]["entity_type"] = "vehicle"
    seed=prepare_write(case.tenant,case.user,seed["payload"],vector=case.llm.vector())
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
                "entity_id":"tmp_veh_99","entity_name":"Little Dog","entity_type":"vehicle",
                "entity_domain":"assets_and_possessions","aliases":["Little Dog"],
                "facts":[{"attribute_name":"nickname","value":field_entries("Little Dog")}]}]})
        data = json.loads(user_text)
        context = data["REFERENCE_ONLY_CONTEXT"]
        assert len(context) == 4 and context[-2:] == reference[-2:]
        assert "identity only" in system_prompt and "extract historical facts as new updates" in system_prompt
        new = data["EXTRACTED_SEMANTIC_INFORMATION"][0]
        candidate = data["CANDIDATES_BY_OBJECT"][new["id"]][0]
        assert new["name"] == "Little Dog" and new["facts"] == [{"attribute_name":"nickname","value":item_value("Little Dog")}]
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
    assert row["id"] == original_id and row["entity_id"] == str(original_id)
    nickname = next(f for f in row["payload"]["facts"] if f["attribute_name"] == "nickname")
    assert fact_text(nickname) == "Little Dog" and nickname["_last_mentioned_at"] == source_timestamps(queue)[queue[0]["item_id"]]
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
            return schema.model_validate({"assets_and_possessions":[{"entity_id":"tmp_veh_01",
                "entity_name":"Tesla","entity_type":"vehicle","entity_domain":"assets_and_possessions",
                "aliases":["Little Dog"],"facts":[
                {"attribute_name":"vehicle_make","value":field_entries("Tesla"),"source_chat_id":batch[0]["item_id"]},
                {"attribute_name":"paint_color","value":field_entries("red"),"source_chat_id":batch[1]["item_id"]},
                {"attribute_name":"nickname","value":field_entries("Little Dog"),"source_chat_id":batch[2]["item_id"]}]}]})
        current=json.loads(user_text)['CURRENT_BATCH']
        return schema.model_validate({'resolutions':[{
            'extracted_entity_id':'tmp_veh_01','kind':'NEW','explicitly_new':True,
            'new_entity_evidence':current[0]['content'],'targets':[],
            'reasoning':'Prescribed same-batch resolved entity',
            'accepted_aliases':['Little Dog'],
            'alias_evidence':[{'alias':'Little Dog','source_chat_id':current[2]['item_id'],
                               'evidence_quote':current[2]['content']}]}]})
    monkeypatch.setattr(case.llm,"is_junk",lambda text:False)
    monkeypatch.setattr(case.llm,"parse_structured",mutation_aware(model))
    monkeypatch.setattr(case.llm,"embed_batch",lambda texts:[case.llm.vector() for text in texts])
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant,case.user,queue)
    row,=memory_rows(case)
    assert row["payload"]["aliases"] == ["Little Dog"]
    timestamps=source_timestamps(queue)
    for f,source in zip(row["payload"]["facts"],queue):
        assert f["_last_mentioned_at"] == timestamps[source["item_id"]] and "source_chat_id" not in f
    assert [fact_text(f) for f in row["payload"]["facts"]] == ["Tesla","red","Little Dog"]
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
            barrier.wait(timeout=10)
            assert db.commit_graph_updates([op])
            return op["entity_id"]
        finally:
            db.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(lambda _:insert(),range(2)))
    assert len(set(ids)) == 2
    assert len(local_db.db.get_recent_entities(local_db.tenant,local_db.users[0])) == 2
