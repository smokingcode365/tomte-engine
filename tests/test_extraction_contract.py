from resolution_helpers import resolution_response, mutation_aware
"""Extraction/reconciliation contract checks; models are deterministic local stubs."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from tomte.engine import Tomte
from tomte.engine.core.extraction import (
    ExtractionMemorySchema, RECENT_CONTEXT_TURNS, build_extraction_input,
    resolve_extraction, source_timestamps,
)
from tomte.engine.llm.prompts.extractor import EXTRACTOR_SYSTEM_PROMPT
from tomte.engine.utils.cognitive import CognitiveProcessor
from tomte.engine.utils.keys import generate_common_key
from tomte.engine.workflows.consolidation import ConsolidationWorkflow
from test_consolidation_mechanics import mechanics, queued_message, memory_rows, processed_count, extraction_response


def item(source="current-1", timestamp=1791547200, content="I named it Little Dog."):
    return {"item_id": source, "timestamp": timestamp, "role": "user", "content": content}


def extracted(source=None, timeline=False):
    return ExtractionMemorySchema.model_validate({"assets_and_possessions": [{
        "entity_id": "veh_99", "entity_name": "Tesla Model Y", "entity_type": "vehicle",
        "entity_domain": "assets_and_possessions", "aliases": ["Little Dog"],
        "facts": [{"attribute_name": "nickname", "value": "Little Dog", "confidence": 1.0,
                   "source_chat_id": source,
                   "timeline": [{"event_description": "Named vehicle Little Dog", "source_chat_id": source}] if timeline else []}],
    }]})


def test_wire_schema_does_not_request_system_timestamps():
    schema = ExtractionMemorySchema.model_json_schema()
    assert "last_mentioned_at" not in schema["$defs"]["ExtractedFact"]["properties"]
    assert "timestamp" not in schema["$defs"]["ExtractedTimelineEvent"]["properties"]


def test_input_separates_bounded_antecedents_from_current_sources():
    first, second = item(), item("current-2", 1791547210, "I sold Little Dog.")
    history = [{"role": "user", "content": f"prior-{i}"} for i in range(7)]
    history[-1]["content"] = "I bought a Tesla Model Y."
    first["recent_context"] = history
    second["recent_context"] = [{"role": "user", "content": "must not replace first context"}]
    prompt = json.loads(build_extraction_input([first, second]))
    assert [x["item_id"] for x in prompt["CURRENT_BATCH"]] == ["current-1", "current-2"]
    assert prompt["RECENT_CONTEXT_REFERENCE_ONLY"] == history[-RECENT_CONTEXT_TURNS:]
    assert all(x["content"] not in [y["content"] for y in prompt["CURRENT_BATCH"]]
               for x in prompt["RECENT_CONTEXT_REFERENCE_ONLY"])
    assert "do not extract its facts as new updates" in EXTRACTOR_SYSTEM_PROMPT
    assert "temporary extraction-local labels" in EXTRACTOR_SYSTEM_PROMPT


def test_single_source_and_event_provenance_are_supplied_by_code():
    source = item()
    result = resolve_extraction(extracted(timeline=True), [source])
    fact = result.assets_and_possessions[0].facts[0]
    stamp = source_timestamps([source])[source["item_id"]]
    assert fact.source_chat_id == source["item_id"]
    assert fact.last_mentioned_at == stamp
    assert fact.timeline[0].source_chat_id == source["item_id"]
    assert fact.timeline[0].timestamp == stamp


@pytest.mark.parametrize("source", [None, "unknown", "invented-id", "historical-id"])
def test_multi_source_requires_a_real_current_message_selection(source):
    with pytest.raises(ValueError, match="CURRENT_BATCH"):
        resolve_extraction(extracted(source), [item(), item("current-2", 1791547210)])


def test_multi_source_resolves_fact_and_event_timestamps_independently():
    batch = [item(), item("current-2", 1791547210)]
    response = extracted("current-2", timeline=True)
    response.assets_and_possessions[0].facts[0].timeline[0].source_chat_id = "current-1"
    fact = resolve_extraction(response, batch).assets_and_possessions[0].facts[0]
    assert fact.last_mentioned_at == source_timestamps(batch)["current-2"]
    assert fact.timeline[0].timestamp == source_timestamps(batch)["current-1"]


@pytest.mark.parametrize("stamp", [None, 0, float("nan"), True])
def test_invalid_queue_timestamp_fails_before_extraction(stamp):
    with pytest.raises(ValueError, match="timestamps"):
        build_extraction_input([item(timestamp=stamp)])


def test_model_timestamp_even_if_epoch_is_rejected():
    data = extracted().model_dump(mode="json")
    data["assets_and_possessions"][0]["facts"][0]["last_mentioned_at"] = "1970-01-01T00:00:00Z"
    with pytest.raises(ValueError):
        ExtractionMemorySchema.model_validate(data)


@pytest.mark.parametrize("color, mode", [("red", "REINFORCE"), ("blue", "MUTATE")])
def test_reconciler_can_use_aliases_and_actual_old_new_values(monkeypatch, color, mode):
    record_id = uuid4()
    skeleton = [{"id": record_id, "entity_id": "veh_01", "entity_name": "Tesla Model Y",
                 "entity_type": "vehicle", "entity_domain": "assets_and_possessions",
                 "payload": {"aliases": ["my car", "Little Dog"],
                 "facts": [{"attribute_name": "paint_color", "value": "red"}]}}]
    new = {"entity_id": "veh_99", "entity_name": "Little Dog", "entity_type": "vehicle",
           "entity_domain": "assets_and_possessions", "aliases": ["Little Dog"],
           "facts": [{"attribute_name": "paint_color", "value": color}]}
    calls = []
    def reconcile(system_prompt, user_text, schema):
        data = json.loads(user_text)
        update = data["EXTRACTED_SEMANTIC_INFORMATION"][0]
        old = data["CANDIDATES_BY_OBJECT"][update["id"]][0]
        calls.append((old, update))
        # This local rule only reads the supplied input; it is not an LLM-quality claim.
        assert update["name"] in old["aliases"]
        decision = "REINFORCE" if update["facts"] == old["facts"] else "MUTATE"
        return resolution_response(schema, {"mappings": [{"extracted_entity_id": update["id"],
                    "mode": decision, "matched_db_id": old["id"], "reasoning": "Alias and values from supplied context"}]}, user_text)
    monkeypatch.setattr(Tomte, "_db", Mock())
    monkeypatch.setattr(Tomte, "_llm", SimpleNamespace(parse_structured=reconcile))
    monkeypatch.setattr("tomte.engine.workflows.consolidation.get_dictionary_service", lambda: Mock())
    op, = ConsolidationWorkflow()._reconcile_with_golden_records("tenant", "user", [new], skeleton)
    assert op["mode"] == mode and op["db_id"] == str(record_id)
    assert op["entity_id"] == op["payload"]["entity_id"] == "veh_01"
    assert calls[0][0]["facts"] == {"paint_color": "red"}
    assert calls[0][1]["facts"] == {"paint_color": color}
    assert calls[0][0]["aliases"] == ["my car", "Little Dog"]


def test_nickname_context_contains_identity_and_fact_status_without_timelines():
    text = CognitiveProcessor().format_entities([{
        "entity_id": "veh_01", "entity_name": "Tesla Model Y", "entity_type": "vehicle",
        "aliases": ["my car", "Little Dog"], "facts": [{"attribute_name": "ownership_status",
        "value": "sold", "importance": 4, "status": "past", "assertion_state":"ACTIVE", "timeline": [{"event_description": "hidden history"}]}],
    }])
    assert "Tesla Model Y" in text and "Little Dog" in text and "TYPE: vehicle" in text
    assert "ownership_status: sold" in text and "status: past" in text
    assert "hidden history" not in text


def test_real_queue_context_is_bounded_scoped_and_retry_stable(mechanics):
    case = mechanics
    history = [f"synthetic prior {i}" for i in range(6)]
    for message in history:
        case.cache.append_chat_history(case.tenant, case.session, message, "user", case.user)
    other_key = generate_common_key(case.tenant + "-other", case.session, case.user)
    try:
        case.cache.append_chat_history(case.tenant + "-other", case.session, "unrelated tenant", "user", case.user)
        queue = queued_message(case)
        assert [x["content"] for x in queue[0]["recent_context"]] == history[-4:]
        case.cache.append_chat_history(case.tenant, case.session, "future message", "user", case.user)
        frozen = case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session)
        assert frozen == queue
        context = json.loads(build_extraction_input(frozen))["RECENT_CONTEXT_REFERENCE_ONLY"]
        assert [x["content"] for x in context] == history[-4:]
        assert case.cache.client.ttl(generate_common_key(case.tenant, case.session, case.user)) > 86000
    finally:
        case.cache.client.delete(other_key)


def test_actual_single_message_provenance_is_persisted(mechanics):
    case = mechanics
    queue = queued_message(case)
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, queue)
    row, = memory_rows(case)
    fact = row["payload"]["facts"][0]
    assert fact["source_chat_id"] == queue[0]["item_id"]
    assert fact["last_mentioned_at"] == source_timestamps(queue)[queue[0]["item_id"]]
    assert processed_count(case) == 1


@pytest.mark.parametrize("defect", ["unknown", "invented-id", "epoch"])
def test_invalid_model_provenance_cannot_persist_or_claim_batch(mechanics, monkeypatch, defect):
    case = mechanics
    original = case.llm.parse_structured
    def invalid(system_prompt, user_text, schema, taxonomy=None, lexicon=None):
        result = original(system_prompt, user_text, schema, taxonomy, lexicon)
        if schema is ExtractionMemorySchema:
            data = result.model_dump(mode="json")
            fact = data["preferences_and_traits"][0]["facts"][0]
            if defect == "epoch":
                fact["last_mentioned_at"] = "1970-01-01T00:00:00Z"
            else:
                fact["source_chat_id"] = defect
            return schema.model_validate(data)
        return result
    monkeypatch.setattr(case.llm, "parse_structured", invalid)
    queue = queued_message(case)
    with pytest.raises(ValueError):
        ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, queue)
    assert memory_rows(case) == [] and processed_count(case) == 0
    assert case.anchors == []
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session) == queue


@pytest.mark.parametrize("field, value", [("source_chat_id", "unknown"),
                                         ("last_mentioned_at", "1970-01-01T00:00:00Z")])
def test_inherited_placeholders_fail_before_transaction_or_marker(monkeypatch, field, value):
    db = Mock()
    monkeypatch.setattr(Tomte, "_db", db)
    monkeypatch.setattr(Tomte, "_llm", Mock())
    monkeypatch.setattr("tomte.engine.workflows.consolidation.get_dictionary_service", lambda: Mock())
    fact = {"source_chat_id": "valid-historical-source", "last_mentioned_at": "2026-10-09T12:00:00Z"}
    fact[field] = value
    with pytest.raises(ValueError, match="placeholder provenance"):
        ConsolidationWorkflow()._commit_refined_operations([{"payload": {"facts": [fact]}}], "tenant", "user", "batch")
    db.transaction.assert_not_called()
    db.claim_consolidation_batch.assert_not_called()
