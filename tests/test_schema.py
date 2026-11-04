"""Stored and wire boundaries have intentionally different authorities."""
from copy import deepcopy
import pytest

from tomte.engine.core.memory import Entity, Fact
from tomte.engine.core.extraction import ExtractionMemorySchema, resolve_extraction
from test_memory_values import CHAT
from test_memory_retrieval import entity
from test_memory_pipeline import LLM


def test_stored_entity_and_fact_ids_are_canonical_uuids():
    data = entity()
    assert Entity.model_validate(data).model_dump(mode="json") == data
    for key in ("entity_id",):
        with pytest.raises(ValueError):
            Entity.model_validate({**data, key: "tmp_person"})
    with pytest.raises(ValueError):
        Fact.model_validate({**data["facts"][0], "fact_id": "fact_relationship"})


def test_duplicate_fact_uuid_is_rejected_but_same_attribute_is_allowed():
    data = entity()
    data["facts"] *= 2
    with pytest.raises(ValueError, match="Duplicate fact UUID"):
        Entity.model_validate(data)


@pytest.mark.parametrize("name", ["source_chat_id", "source_chat_sum", "related_entity_ids", "status", "confidence", "note", "history_id", "schema_version"])
def test_retired_fact_fields_are_not_silently_ignored(name):
    with pytest.raises(ValueError):
        Fact.model_validate({**entity()["facts"][0], name: "unexpected"})


@pytest.mark.parametrize("stamp", ["yesterday", "2026-02-30T10:00:00Z", "2026-11-02T10:00:00+00:00"])
def test_metadata_dates_require_real_utc_timestamps(stamp):
    data = entity()["facts"][0]
    data["value"]["job_title"]["_recorded_at"] = stamp
    with pytest.raises(ValueError):
        Fact.model_validate(data)


def test_extraction_rejects_extra_root_keys_instead_of_ignoring_model_noise():
    with pytest.raises(ValueError):
        ExtractionMemorySchema.model_validate({"}": " "})


def test_extraction_group_handles_must_be_present_without_inventing_entities():
    data = LLM().parse_structured(None, "", ExtractionMemorySchema).model_dump(mode="json")
    original = deepcopy(data)
    data["relationships"][0]["facts"][0]["group_participant_handles"] = ["tmp_missing"]
    wire = ExtractionMemorySchema.model_validate(data)
    resolved = resolve_extraction(wire, CHAT)
    person, = resolved.relationships
    assert person.entity_id == original["relationships"][0]["entity_id"]
    assert [fact.attribute_name for fact in person.facts] == ["employment"]
    assert sum(len(items) for items in resolved.model_dump().values()) == 1
    assert wire.model_dump(mode="json") == data  # Boundary does not mutate proposals.
    assert original["relationships"][0]["facts"][0]["group_participant_handles"] == []
