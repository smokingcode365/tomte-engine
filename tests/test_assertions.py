"""Fact operation invariants; lifecycle rows were replaced by embedded history."""
from copy import deepcopy
from uuid import UUID
import pytest

from tomte.engine.core.assertions import FactMutation, apply_mutations, validate_facts
from tomte.engine.core.extraction import ExtractionMemorySchema
from test_memory_values import NOW, LATER, CHAT, fields, job, update, apply


def test_add_uuid_is_application_owned_deterministic_for_the_frozen_plan_and_scoped():
    op = {"operation": "ADD", "attribute_name": "employment", "fields": fields(employer_name="A"), "source_chat_ids": ["m1"]}
    a = apply_mutations([], [op], CHAT, "entity-one", "batch", recorded_at=NOW)[0]
    b = apply_mutations([], [op], CHAT, "entity-one", "batch", recorded_at=NOW)[0]
    c = apply_mutations([], [op], CHAT, "entity-two", "batch", recorded_at=NOW)[0]
    assert a == b and UUID(a["fact_id"]) != UUID(c["fact_id"])
    assert "source_chat_id" not in a and "evidence" not in a


@pytest.mark.parametrize("source_ids", [["assistant"], ["m1", "assistant"]])
def test_assistant_evidence_rejected_without_changing_input(source_ids):
    old = job()
    before = deepcopy(old)
    messages = CHAT + [{**CHAT[0], "item_id": "assistant", "role": "assistant"}]
    with pytest.raises(ValueError, match="CURRENT USER"):
        apply_mutations([old], [{"operation": "MENTION", "target_fact_id": old["fact_id"], "source_chat_ids": source_ids}],
                        messages, "entity", "batch")
    assert old == before


def test_multiple_current_messages_count_once_each_and_use_original_message_times():
    old = job()
    messages = CHAT + [{**CHAT[0], "item_id": "later", "timestamp": CHAT[0]["timestamp"] + 120}]
    ops = [{"operation": "MENTION", "target_fact_id": old["fact_id"], "source_chat_ids": ["m1", "later"]},
           {"operation": "MENTION", "target_fact_id": old["fact_id"], "source_chat_ids": ["later"]}]
    result = apply_mutations([old], ops, messages, "entity", "batch", recorded_at=LATER)[0]
    from tomte.engine.core.extraction import source_timestamps
    assert result["_mention_count"] == old["_mention_count"] + 2
    assert result["_last_mentioned_at"] == source_timestamps(messages)["later"]
    assert result["value"] == old["value"]


def test_old_message_does_not_move_last_mention_backwards():
    old = job()
    old["_last_mentioned_at"] = LATER
    result = apply([old], {"operation": "MENTION", "target_fact_id": old["fact_id"]})[0]
    assert result["_last_mentioned_at"] == LATER


def test_end_is_descriptive_current_value_and_preserves_history_in_same_fact():
    old = job()
    result = apply([old], update(old, fields=fields(end="No longer employed at Cedar Labs"), change_type="real_change"))[0]
    assert result["fact_id"] == old["fact_id"]
    assert result["history_values"][0]["value"] == old["value"]
    assert result["value"]["end"]["text"] == "No longer employed at Cedar Labs"
    assert "status" not in result


def test_correction_never_restores_a_predecessor_or_corrects_history():
    old = job()
    changed = apply([old], update(old, fields=fields(employer_name="Pine Labs"), change_type="real_change"))[0]
    corrected = apply([changed], update(changed, fields=fields(employer_name="Birch Labs"), change_type="correction"))[0]
    assert corrected["history_values"] == changed["history_values"]
    assert corrected["value"]["employer_name"]["text"] == "Birch Labs"
    assert corrected["fact_id"] == old["fact_id"]


def test_whole_plan_failure_leaves_original_unchanged():
    old = job()
    before = deepcopy(old)
    with pytest.raises(ValueError, match="offered entity"):
        apply([old], update(old, fields=fields(employer_name="A")),
              {"operation": "MENTION", "target_fact_id": "foreign"})
    assert old == before


@pytest.mark.parametrize("source", ["fabricated-id", "unknown", "previous-session-message"])
def test_source_validation_rejects_fabrication_and_reference_only_messages(source):
    with pytest.raises(ValueError, match="CURRENT USER"):
        apply([], {"operation": "ADD", "attribute_name": "employment", "fields": fields(employer_name="A"),
                   "source_chat_ids": [source]})


@pytest.mark.parametrize("extra", [{"fact_id": "invented"}, {"_recorded_at": NOW}, {"history_id": "invented"},
                                   {"evidence_type": "USER_EXPLICIT"}, {"interpretation_confidence": 1}])
def test_planner_rejects_authoritative_or_retired_metadata(extra):
    with pytest.raises(ValueError):
        FactMutation(operation="ADD", attribute_name="employment", fields=fields(employer_name="A"),
                     source_chat_ids=["m1"], **extra)


@pytest.mark.parametrize("data", [
    {"operation": "ADD", "target_fact_id": "foreign", "attribute_name": "employment", "fields": fields(employer_name="A")},
    {"operation": "ADD", "attribute_name": "employment"},
    {"operation": "MENTION"},
    {"operation": "MENTION", "target_fact_id": "id", "fields": fields(employer_name="A")},
    {"operation": "UPDATE", "target_fact_id": "id"},
    {"operation": "UPDATE", "target_fact_id": "id", "fields": fields(employer_name="A"), "history_index": 0},
])
def test_invalid_operation_shapes_fail(data):
    with pytest.raises(ValueError):
        FactMutation.model_validate({**data, "source_chat_ids": ["m1"]})


@pytest.mark.parametrize("change_type", ["correction", "real_change", "enrichment"])
def test_append_history_ignores_update_only_controls_without_editing_current_value(change_type):
    old = job()
    operation = FactMutation(operation="APPEND_HISTORY", target_fact_id=old["fact_id"],
        fields=fields(employer_name="Earlier employer"), source_chat_ids=["m1"],
        update_mode="replace_value", change_type=change_type)
    assert operation.update_mode == "update_fields" and operation.change_type == "enrichment"
    result = apply([old], operation.model_dump(mode="json"), now=LATER)[0]
    assert result["value"] == old["value"] and result["fact_id"] == old["fact_id"]
    assert len(result["history_values"]) == len(old["history_values"]) + 1
    assert result["history_values"][-1]["value"]["employer_name"]["text"] == "Earlier employer"


def test_same_value_real_change_is_mention_without_extra_history():
    old = job()
    new = apply([old], update(old, fields=fields(employer_name="Cedar Labs"), change_type="real_change"), now=LATER)[0]
    assert new["value"] == old["value"] and new["history_values"] == []
    assert new["_mention_count"] == old["_mention_count"] + 1


def test_other_entity_fact_and_unoffered_fact_cannot_be_targeted():
    old = job()
    with pytest.raises(ValueError, match="offered entity"):
        apply([], {"operation": "MENTION", "target_fact_id": old["fact_id"]})
    with pytest.raises(ValueError, match="offered entity"):
        apply([old], {"operation": "MENTION", "target_fact_id": old["fact_id"]}, offered_fact_ids=[])


def test_old_scalar_memory_is_rejected_instead_of_synthesizing_ids_or_provenance():
    with pytest.raises(ValueError):
        validate_facts([{"attribute_name": "employment", "value": "old job", "status": "past"}])


def test_extraction_does_not_ask_model_for_storage_metadata_or_edges():
    schema = ExtractionMemorySchema.model_json_schema()
    properties = schema["$defs"]["ExtractedFact"]["properties"]
    assert not set(properties) & {"fact_id", "created_at", "assertion_state", "evidence", "history_values",
                                 "related_entity_ids", "related_entity_uuids", "mention_count", "_importance"}
    assert properties["value"]["type"] == "array"
