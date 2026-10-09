"""Deterministic lifecycle through real local PostgreSQL/Redis and production workflow."""
from copy import deepcopy
import json
from types import SimpleNamespace
import pytest
from tomte.engine.core.assertions import project_facts
from tomte.engine.core.extraction import ExtractionMemorySchema
from tomte.engine.workflows.chat_hook import ChatHookWorkflow
from tomte.engine.workflows.consolidation import ConsolidationWorkflow
from tomte.engine.core.enums import ChatSenderRole
from test_consolidation_mechanics import mechanics, memory_rows, processed_count
from test_postgres_transactions import local_db, memory_update
from resolution_helpers import resolution_response


@pytest.fixture
def assertion_case(mechanics,monkeypatch):
    case=mechanics
    case.plan_kind="ADD";case.plan_value="owned";case.fail_plan=False
    def model(system_prompt,user_text,schema,**kwargs):
        data=json.loads(user_text)
        if schema is ExtractionMemorySchema:
            value=case.plan_value
            return schema.model_validate({"assets_and_possessions":[{
                "entity_id":"veh_01","entity_name":"Tesla Model Y","entity_type":"vehicle",
                "entity_domain":"assets_and_possessions","aliases":["Tesla"],
                "facts":[] if value is None else [{"attribute_name":"ownership_status","value":value,"confidence":1}]}]})
        if schema.__name__=="ResolutionResponse":
            candidates=data["CANDIDATES_BY_OBJECT"]["veh_01"]
            return resolution_response(schema,{"mappings":[{"extracted_entity_id":"veh_01",
                "mode":"MUTATE" if candidates else "INSERT", "matched_db_id":candidates[0]["id"] if candidates else None,
                "reasoning":"Deterministic local identity fixture"}]},user_text)
        assert schema.__name__=="MutationPlanResponse"
        target,=data["TARGETS"]
        active=next((f for f in target["assertions"] if f["assertion_state"]=="ACTIVE"),None)
        current=data["CURRENT_BATCH"][0]["item_id"]
        operation={"operation":case.plan_kind,"target_fact_id":None if case.plan_kind=="ADD" else active["fact_id"],
            "attribute_name":"ownership_status","new_value":case.plan_value,
            "source_chat_ids":[current],"interpretation_confidence":0.96,"evidence_type":"USER_EXPLICIT"}
        operations=[operation]
        if case.fail_plan:
            operations=[{"operation":"ADD","attribute_name":"color","new_value":"red",
                         "source_chat_ids":[current],"interpretation_confidence":1,"evidence_type":"USER_EXPLICIT"},
                         {**operation,"target_fact_id":"foreign-or-invented-fact"}]
        return schema.model_validate({"plans":[{"target_key":target["target_key"],"operations":operations}]})
    monkeypatch.setattr(case.llm,"parse_structured",model)
    monkeypatch.setattr(case.llm,"is_junk",lambda text:False)
    monkeypatch.setattr(case.llm,"embed_text",lambda text:case.llm.vector())
    monkeypatch.setattr(case.llm,"embed_batch",lambda texts:[case.llm.vector() for text in texts])
    return case


def cycle(case,message):
    ChatHookWorkflow().on_chat(case.tenant,case.session,message,ChatSenderRole.USER,case.user)
    queue=case.cache.pop_session_chat_log_4_consolidation(case.tenant,case.user,case.session)
    workflow=ConsolidationWorkflow()
    assert workflow.run_consolidation_cycle(case.tenant,case.user,queue)
    before=deepcopy(memory_rows(case));markers=processed_count(case)
    assert workflow.run_consolidation_cycle(case.tenant,case.user,queue)
    assert memory_rows(case)==before and processed_count(case)==markers
    assert case.cache.clear_session_4_consolidation(case.tenant,case.user,case.session,queue)
    return before[0]


def test_real_all_five_operations_replay_projection_and_original_truth(assertion_case):
    case=assertion_case
    created=cycle(case,"I bought a Tesla.")
    original=created["payload"]["facts"][0]
    case.plan_kind="REINFORCE";case.plan_value=None
    reinforced=cycle(case,"I still own my Tesla.")
    reinforced_fact,=reinforced["payload"]["facts"]
    assert reinforced_fact["fact_id"]==original["fact_id"] and reinforced_fact["mention_count"]==2
    assert len(reinforced_fact["evidence"])==2
    case.plan_kind="SUPERSEDE";case.plan_value="sold"
    sold=cycle(case,"I sold my Tesla.")
    old,current=sold["payload"]["facts"]
    assert old["value"]=="owned" and old["assertion_state"]=="HISTORICAL"
    assert current["value"]=="sold" and current["supersedes_fact_id"]==old["fact_id"]
    assert current["fact_id"]!=old["fact_id"]
    case.plan_kind="CORRECT";case.plan_value="owned"
    corrected=cycle(case,"No, I never sold it. I still own the Tesla.")
    assertions=corrected["payload"]["facts"]
    invalid=next(f for f in assertions if f["fact_id"]==current["fact_id"])
    replacement,=project_facts(assertions)
    assert invalid["assertion_state"]=="CORRECTED_INVALID"
    assert replacement["value"]=="owned" and replacement["corrects_fact_id"]==invalid["fact_id"]
    assert all(f["value"]!="sold" for f in project_facts(assertions,"history"))
    case.plan_kind="RETRACT";case.plan_value=None
    final=cycle(case,"I no longer want you to treat that ownership assertion as current.")
    assert project_facts(final["payload"]["facts"])==[]
    assert len(final["payload"]["facts"])==3 and processed_count(case)==5
    assert {row["id"] for row in [created,reinforced,sold,corrected,final]}=={created["id"]}
    assert all(row["payload"]["entity_name"]=="Tesla Model Y" for row in [created,reinforced,sold,corrected,final])


def test_invalid_later_mutation_rolls_back_whole_batch_and_retains_snapshot(assertion_case):
    case=assertion_case
    cycle(case,"I bought a Tesla.")
    before=memory_rows(case);markers=processed_count(case)
    case.plan_kind="RETRACT";case.plan_value=None;case.fail_plan=True
    ChatHookWorkflow().on_chat(case.tenant,case.session,"Ignore current ownership.",ChatSenderRole.USER,case.user)
    queue=case.cache.pop_session_chat_log_4_consolidation(case.tenant,case.user,case.session)
    with pytest.raises(ValueError,match="offered entity"):
        ConsolidationWorkflow().run_consolidation_cycle(case.tenant,case.user,queue)
    assert memory_rows(case)==before and processed_count(case)==markers
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant,case.user,case.session)==queue


def test_stale_preparation_rejects_batch_and_cannot_overwrite_concurrent_assertions(local_db):
    case=local_db;op=memory_update(case.tenant,case.users[0])
    assert case.db.commit_graph_updates([op])
    record,=case.db.get_user_graph_skeleton(case.tenant,case.users[0])
    stale={**op,"mode":"MUTATE","db_id":str(record["id"]),"_expected_payload":deepcopy(record["payload"])}
    concurrent={**op,"mode":"MUTATE","db_id":str(record["id"]),
                "payload":{**op["payload"],"aliases":["concurrent alias"]}}
    assert case.db.commit_graph_updates([concurrent])
    with pytest.raises(ValueError,match="Stale assertion"):
        with case.db.transaction():
            case.db.claim_consolidation_batch(case.tenant,case.users[0],"stale-batch")
            case.db.commit_graph_updates([stale])
    current,=case.db.get_user_graph_skeleton(case.tenant,case.users[0])
    assert current["payload"]["aliases"]==["concurrent alias"]
    assert not case.db.is_consolidation_processed(case.tenant,case.users[0],"stale-batch")
