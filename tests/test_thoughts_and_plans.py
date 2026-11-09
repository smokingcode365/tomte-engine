"""Thought-domain mechanics; prescribed model outputs do not prove semantic accuracy."""
from copy import deepcopy
from datetime import datetime
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from memory_fixtures import field_entries, simple_entity, simple_fact
from tomte.engine.core.assertions import MutationPlanResponse
from tomte.engine.core.extraction import ExtractionMemorySchema, resolve_extraction
from tomte.engine.core.memory import Entity, public_entity
from tomte.engine.core.resolution import ResolutionResponse
from tomte.engine.core.relationships import prepare_proposals
from tomte.engine.core.schema import TomteMemorySchema
from tomte.engine.llm.embedder import get_search_entries
from tomte.engine.providers.postgres_provider import PostgresProvider
from tomte.engine.workflows.consolidation import ConsolidationWorkflow

DOMAIN = "thoughts_and_plans"
FIRST = "2026-01-05T10:00:00Z"
LATER = "2026-02-06T11:00:00Z"


def chat(source="first", stamp=FIRST, role="user"):
    return dict(item_id=source, role=role, timestamp=datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp(),
                content="I am considering buying a Tesla.")


def thought():
    # Build a legacy-compatible fixture before adding the new domain fields.
    data = simple_entity("Car purchase intention", kind="purchase_intention",
                         facts=[simple_fact("intention", "Considering a Tesla.")])
    return dict(data, entity_domain=DOMAIN, first_mentioned_at=FIRST)


def wire():
    return dict(entity_id="tmp_idea_1", entity_name="Car purchase intention", entity_domain=DOMAIN,
                entity_type="purchase_intention", facts=[dict(attribute_name="intention",
                value=field_entries("Considering a Tesla."), source_chat_id="first")])


def resolved(raw=None):
    raw = raw or wire()
    schema = resolve_extraction(ExtractionMemorySchema.model_validate({raw["entity_domain"]: [raw]}), [chat()])
    return getattr(schema, raw["entity_domain"])[0].model_dump(mode="json")


def plan(incoming=None, *, old=None, operations=None, messages=None):
    incoming = incoming or resolved()
    record = None if old is None else dict(id=old["entity_id"], payload=old)
    prepared = ConsolidationWorkflow._prepare_operation("tenant", "user", incoming, record)
    prepared = prepare_proposals([prepared])[0][0]
    response = dict(plans=[dict(target_key="target_0", operations=operations or [dict(operation="ADD",
        attribute_name="intention", fields=field_entries("Considering a Tesla."), source_chat_ids=["first"])])])
    seen = []
    def parse(**kwargs):
        seen.append(json.loads(kwargs["user_text"]))
        return MutationPlanResponse.model_validate(response)
    workflow = ConsolidationWorkflow(db=SimpleNamespace(), llm=SimpleNamespace(parse_structured=parse))
    result = workflow._plan_fact_mutations([prepared], dict(CURRENT_BATCH=messages or [chat()],
                    RECENT_CONTEXT_REFERENCE_ONLY=[]), "test-batch")
    return result, seen


def test_thought_domain_extracts_and_flattens_without_requesting_system_time():
    extracted = ExtractionMemorySchema.model_validate({DOMAIN: [wire()]})
    resolved = resolve_extraction(extracted, [chat()])
    workflow = object.__new__(ConsolidationWorkflow)
    entities = workflow._flatten_schema(resolved)
    assert len(entities) == 1 and entities[0]["entity_domain"] == DOMAIN
    assert "first_mentioned_at" not in ExtractionMemorySchema.model_json_schema()["$defs"]["ExtractedEntity"]["properties"]
    assert DOMAIN in TomteMemorySchema.model_fields


def test_first_mention_uses_accepted_sources_not_processing_or_rejected_items():
    messages = [chat("unrelated", "2025-01-01T00:00:00Z"), chat(), chat("later", LATER), chat("assistant", "2024-01-01T00:00:00Z", "assistant")]
    operations = [dict(operation="ADD", attribute_name="intention", fields=field_entries("Considering a Tesla."), source_chat_ids=["later", "first"]),
                  dict(operation="ADD", attribute_name="invalid", fields=field_entries("Unusable."), source_chat_ids=["assistant"])]
    result, _ = plan(operations=operations, messages=messages)
    assert len(result) == 1
    saved = Entity.model_validate(result[0]["payload"]).model_dump(mode="json")
    assert saved["first_mentioned_at"] == FIRST
    assert len(saved["facts"]) == 1
    assert saved["facts"][0]["value"]["detail"]["_recorded_at"] != FIRST
    assert public_entity(saved)["first_mentioned_at"] == FIRST


@pytest.mark.parametrize("operation", ["MENTION", "UPDATE"])
def test_existing_thought_preserves_first_mention_and_shows_it_to_planner(operation):
    old = thought()
    mutation = dict(operation=operation, target_fact_id=old["facts"][0]["fact_id"], source_chat_ids=["later"])
    if operation == "UPDATE":
        mutation.update(fields=field_entries("Decided against the purchase."), change_type="real_change", update_mode="replace_value")
    result, seen = plan(old=old, operations=[mutation], messages=[chat("later", LATER)])
    saved = result[0]["payload"]
    assert saved["first_mentioned_at"] == FIRST
    assert seen[0]["TARGETS"][0]["identity"]["first_mentioned_at"] == FIRST
    if operation == "MENTION":
        assert saved["facts"][0]["value"] == old["facts"][0]["value"]
        assert result[0]["mode"] == "REINFORCE"
    else:
        assert saved["facts"][0]["history_values"][0]["value"] == old["facts"][0]["value"]


def test_older_original_message_can_move_first_mention_earlier():
    old = thought()
    old["first_mentioned_at"] = LATER
    mutation = dict(operation="MENTION", target_fact_id=old["facts"][0]["fact_id"], source_chat_ids=["first"])
    result, _ = plan(old=old, operations=[mutation])
    assert result[0]["payload"]["first_mentioned_at"] == FIRST


def test_rejected_group_cannot_move_first_mention_earlier():
    old = thought()
    old["first_mentioned_at"] = LATER
    incoming = resolved()
    shared = incoming["facts"][0]
    shared["_group_targets"] = [old["entity_id"], str(uuid4())]
    independent = deepcopy(shared)
    independent.pop("_group_targets")
    independent.update(attribute_name="budget", source_chat_id="later")
    incoming["facts"].append(independent)
    operations = [dict(operation="ADD", attribute_name="intention", incoming_fact_key="fact_0",
                       fields=field_entries("Shared idea."), source_chat_ids=["first"]),
                  dict(operation="ADD", attribute_name="budget", incoming_fact_key="fact_1",
                       fields=field_entries("Still considering the budget."), source_chat_ids=["later"])]
    result, _ = plan(incoming, old=old, operations=operations, messages=[chat(), chat("later", LATER)])
    assert len(result) == 1
    saved = result[0]["payload"]
    assert saved["first_mentioned_at"] == LATER
    assert [f["attribute_name"] for f in saved["facts"]] == ["intention", "budget"]


@pytest.mark.parametrize("stamp", [None, "not-a-date", "2026-02-30T10:00:00Z"])
def test_thought_storage_requires_valid_first_mention(stamp):
    data = thought()
    if stamp is None:
        data.pop("first_mentioned_at")
    else:
        data["first_mentioned_at"] = stamp
    with pytest.raises(ValidationError):
        Entity.model_validate(data)


def test_old_entities_keep_their_shape_and_flat_provider_reads():
    old = simple_entity(facts=[simple_fact("name", "Bob")])
    assert "first_mentioned_at" not in old
    assert "first_mentioned_at" not in public_entity(old)
    assert PostgresProvider._normalize_memory_rows([old]) == [old]
    saved = thought()
    assert PostgresProvider._normalize_memory_rows([saved])[0]["first_mentioned_at"] == FIRST


@pytest.mark.parametrize("incoming_domain,stored_domain", [(DOMAIN,"assets_and_possessions"),("assets_and_possessions",DOMAIN)])
def test_incompatible_resolution_is_not_offered_or_accepted(incoming_domain, stored_domain):
    stored = thought()
    stored["entity_domain"] = stored_domain
    raw = wire()
    raw["entity_domain"] = incoming_domain
    raw = resolved(raw)
    calls = []
    def parse(**kwargs):
        calls.append(json.loads(kwargs["user_text"]))
        return ResolutionResponse.model_validate(dict(resolutions=[dict(extracted_entity_id=raw["entity_id"],kind="MATCH_ONE",
            reasoning="Prescribed invalid match",targets=[dict(db_id=stored["entity_id"],resolution_confidence=1,evidence="EXPLICIT")])]))
    workflow = ConsolidationWorkflow(db=SimpleNamespace(), llm=SimpleNamespace(parse_structured=parse))
    row = dict(id=stored["entity_id"],payload=stored,**{k:stored[k] for k in ("entity_id","entity_name","entity_type","entity_domain")})
    result = workflow._reconcile_with_golden_records("tenant", "user", [raw], [row], current_batch=[chat()])
    assert result == []
    assert calls[0]["CANDIDATES_BY_OBJECT"][raw["entity_id"]] == []


def test_long_thought_name_cannot_remove_domain_from_search_chunks():
    data = thought()
    data["entity_name"] = "A long intention " * 100
    entries = get_search_entries(data)
    assert entries and all(DOMAIN in e["text"] for e in entries)
    assert all(FIRST in e["text"] for e in entries)


def test_contextual_lookup_keeps_thoughts_and_passes_first_mention():
    saved = thought()
    row = dict(id=saved["entity_id"],payload=saved,**{k:saved[k] for k in ("entity_id","entity_name","entity_type","entity_domain")})
    calls=[]
    def candidates(tenant,user,names,domain,*args):
        assert domain == ""
        return [row]
    def parse(**kwargs):
        calls.append(json.loads(kwargs["user_text"]))
        return kwargs["schema"].model_validate(dict(actions=[]))
    workflow=ConsolidationWorkflow(db=SimpleNamespace(get_resolution_candidates=candidates),llm=SimpleNamespace(parse_structured=parse))
    workflow._resolve_contextual_actions("tenant","user",dict(CURRENT_BATCH=[chat()],RECENT_CONTEXT_REFERENCE_ONLY=[]),[1.0])
    assert calls[0]["CANDIDATES"][0]["first_mentioned_at"] == FIRST


def test_thought_flows_through_consolidation_and_replay_without_extra_model_calls():
    from test_memory_pipeline import LLM, MemoryDB

    class ThoughtLLM(LLM):
        def parse_structured(self, system_prompt, user_text, schema):
            if schema is ExtractionMemorySchema:
                self.calls.append(schema)
                raw = wire()
                raw["facts"][0]["source_chat_id"] = "m1"
                return schema.model_validate({DOMAIN: [raw]})
            if schema is ResolutionResponse:
                self.calls.append(schema)
                return schema.model_validate(dict(resolutions=[dict(extracted_entity_id="tmp_idea_1", kind="NEW",
                    reasoning="Prescribed introduction of a tentative intention.", explicitly_new=True,
                    new_entity_evidence="I am considering buying a Tesla.")]))
            return super().parse_structured(system_prompt, user_text, schema)

    db, llm = MemoryDB(), ThoughtLLM()
    workflow = ConsolidationWorkflow(db, llm)
    batch = [dict(chat("m1"), tenant_id="tenant", user_id="user", session_id="session", vector=[1.0, 0.0])]
    assert workflow.run_consolidation_cycle("tenant", "user", batch)
    assert len(db.rows) == 1
    saved = Entity.model_validate(db.rows[0]["payload"]).model_dump(mode="json")
    assert saved["entity_domain"] == DOMAIN and saved["first_mentioned_at"] == FIRST
    assert saved["facts"][0]["value"]["detail"]["text"] == "Considering a Tesla."
    assert db.rows[0]["search_entries"]
    assert "source_chat" not in json.dumps(saved) and "tmp_" not in json.dumps(saved)
    assert llm.calls.count(ExtractionMemorySchema) == llm.calls.count(ResolutionResponse) == llm.calls.count(MutationPlanResponse) == 1
    before = len(llm.calls)
    assert workflow.run_consolidation_cycle("tenant", "user", batch)
    assert len(db.rows) == 1 and len(llm.calls) == before
