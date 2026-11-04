"""Deterministic lifecycle through real local PostgreSQL/Redis and production workflow."""
from copy import deepcopy
import json
from types import SimpleNamespace
import pytest
from memory_fixtures import field_entries, fact_text, prepare_write
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
    case.plan_kind="ADD";case.plan_value="owned";case.fail_plan=False;case.change_type="enrichment"
    def model(system_prompt,user_text,schema,**kwargs):
        data=json.loads(user_text)
        if schema.__name__ in {"TerminologyAlignmentResponse", "FieldAlignmentResponse", "ContextualActionResponse"}:
            return schema()
        if schema is ExtractionMemorySchema:
            value=case.plan_value
            return schema.model_validate({"assets_and_possessions":[{
                "entity_id":"tmp_veh_01","entity_name":"Tesla Model Y","entity_type":"vehicle",
                "entity_domain":"assets_and_possessions","aliases":["Tesla"],
                "facts":[{"attribute_name":"ownership_status","value":field_entries(value or "owned")}]}]})
        if schema.__name__=="ResolutionResponse":
            candidates=data["CANDIDATES_BY_OBJECT"]["tmp_veh_01"]
            return resolution_response(schema,{"mappings":[{"extracted_entity_id":"tmp_veh_01",
                "mode":"MUTATE" if candidates else "INSERT", "matched_db_id":candidates[0]["id"] if candidates else None,
                "reasoning":"Deterministic local identity fixture"}]},user_text)
        assert schema.__name__=="MutationPlanResponse"
        target,=data["TARGETS"]
        active=next(iter(target["facts"]),None)
        current=data["CURRENT_BATCH"][0]["item_id"]
        operation={"operation":case.plan_kind,"target_fact_id":None if case.plan_kind=="ADD" else active["fact_id"],
            "attribute_name":"ownership_status","fields":field_entries(case.plan_value) if case.plan_value else [],
            "source_chat_ids":[current]}
        if case.plan_kind == "UPDATE":
            operation.update(change_type=case.change_type, update_mode="replace_value")
        operations=[operation]
        if case.fail_plan:
            operations=[{"operation":"ADD","attribute_name":"color","fields":field_entries("red"),
                         "source_chat_ids":[current]},
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


def test_real_value_lifecycle_keeps_one_fact_immutable_history_and_replay(assertion_case):
    case=assertion_case
    created=cycle(case,"I bought a Tesla.")
    original=created["payload"]["facts"][0]
    case.plan_kind="MENTION";case.plan_value=None
    reinforced=cycle(case,"I still own my Tesla.")
    reinforced_fact,=reinforced["payload"]["facts"]
    assert reinforced_fact["fact_id"]==original["fact_id"] and reinforced_fact["_mention_count"]==2
    assert reinforced_fact["value"] == original["value"]
    case.plan_kind="UPDATE";case.plan_value="sold";case.change_type="real_change"
    sold=cycle(case,"I sold my Tesla.")
    current,=sold["payload"]["facts"]
    assert fact_text(current)=="sold" and current["fact_id"]==original["fact_id"]
    assert current["history_values"][0]["value"]==original["value"]
    case.plan_value="owned";case.change_type="correction"
    corrected=cycle(case,"No, I never sold it. I still own the Tesla.")
    replacement,=corrected["payload"]["facts"]
    assert fact_text(replacement)=="owned" and replacement["fact_id"]==original["fact_id"]
    assert replacement["history_values"]==current["history_values"]
    case.plan_value="No longer owned";case.change_type="real_change"
    final=cycle(case,"I no longer own that Tesla.")
    final_fact,=final["payload"]["facts"]
    assert fact_text(final_fact)=="No longer owned" and len(final_fact["history_values"])==2
    assert final_fact["_mention_count"]==5 and processed_count(case)==5
    assert {row["id"] for row in [created,reinforced,sold,corrected,final]}=={created["id"]}
    assert all(row["payload"]["entity_name"]=="Tesla Model Y" for row in [created,reinforced,sold,corrected,final])


def test_invalid_later_mutation_isolated_while_valid_sibling_commits_once(assertion_case):
    case=assertion_case
    cycle(case,"I bought a Tesla.")
    before=memory_rows(case);markers=processed_count(case)
    case.plan_kind="MENTION";case.plan_value=None;case.fail_plan=True
    ChatHookWorkflow().on_chat(case.tenant,case.session,"I painted my Tesla red.",ChatSenderRole.USER,case.user)
    queue=case.cache.pop_session_chat_log_4_consolidation(case.tenant,case.user,case.session)
    workflow=ConsolidationWorkflow()
    assert workflow.run_consolidation_cycle(case.tenant,case.user,queue)
    after=memory_rows(case)
    assert len(after)==len(before)==1 and after[0]["id"]==before[0]["id"]
    old,=before[0]["payload"]["facts"]
    kept,color=after[0]["payload"]["facts"]
    assert kept==old
    assert color["attribute_name"]=="color" and fact_text(color)=="red"
    assert color["fact_id"]!=old["fact_id"] and color["_mention_count"]==1
    assert workflow.last_mutation_recovery["skipped_units"]==1
    assert processed_count(case)==markers+1
    assert workflow.run_consolidation_cycle(case.tenant,case.user,queue)
    assert memory_rows(case)==after and processed_count(case)==markers+1
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant,case.user,case.session)==queue


def test_stale_preparation_rejects_batch_and_cannot_overwrite_concurrent_assertions(local_db):
    case=local_db;op=memory_update(case.tenant,case.users[0])
    assert case.db.commit_graph_updates([op])
    record,=case.db.get_user_graph_skeleton(case.tenant,case.users[0])
    stale=prepare_write(case.tenant,case.users[0],record["payload"],old=record["payload"])
    concurrent=prepare_write(case.tenant,case.users[0],{**record["payload"],"aliases":["concurrent alias"]},old=record["payload"])
    assert case.db.commit_graph_updates([concurrent])
    with pytest.raises(ValueError,match="Stale"):
        with case.db.transaction():
            case.db.claim_consolidation_batch(case.tenant,case.users[0],"stale-batch")
            case.db.commit_graph_updates([stale])
    current,=case.db.get_user_graph_skeleton(case.tenant,case.users[0])
    assert current["payload"]["aliases"]==["concurrent alias"]
    assert not case.db.is_consolidation_processed(case.tenant,case.users[0],"stale-batch")
