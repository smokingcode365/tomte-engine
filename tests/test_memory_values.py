"""Fact update invariants independent of model language/decisions."""
from copy import deepcopy
from uuid import UUID

import pytest
from pydantic import ValidationError
from tomte.engine.core.assertions import FactMutation, apply_mutations
from tomte.engine.core.memory import Fact, public_fact, wire_value

NOW = "2026-11-02T10:00:00Z"
LATER = "2026-11-03T10:00:00Z"
CHAT = [{"item_id": "m1", "role": "user", "timestamp": 1793613600, "content": "test"}]


def fields(**values):
    return [{"field_name": key, "text": value, "fact_confidence": "high"} for key, value in values.items()]


def apply(facts, *ops, now=NOW, **kwargs):
    return apply_mutations(facts, [{"source_chat_ids": ["m1"], **op} for op in ops], CHAT,
                           "d5bcc612-2b8a-4bca-bc1f-9362d58f1456", "batch", recorded_at=now, **kwargs)


def job():
    return apply([], {"operation": "ADD", "attribute_name": "employment",
                      "fields": fields(job_title="researcher", employer_name="Cedar Labs")})[0]


def update(fact, **kwargs):
    return {"operation": "UPDATE", "target_fact_id": fact["fact_id"], **kwargs}


def test_real_change_keeps_uuid_and_full_snapshot_patch_preserves_other_fields():
    old = job()
    saved = deepcopy(old)
    new = apply([old], update(old, fields=fields(job_title="senior researcher"), change_type="real_change"), now=LATER)[0]
    assert UUID(new["fact_id"]) == UUID(old["fact_id"])
    assert new["history_values"] == [{"_archived_at": LATER, "value": old["value"]}]
    assert new["value"]["employer_name"] == old["value"]["employer_name"]
    assert new["value"]["job_title"]["_recorded_at"] == NOW
    assert new["value"]["job_title"]["_updated_at"] == LATER
    assert old == saved


@pytest.mark.parametrize("change_type", ["correction", "enrichment"])
def test_correction_and_enrichment_do_not_invent_history(change_type):
    old = job()
    new = apply([old], update(old, fields=fields(job_title="engineer"), change_type=change_type))[0]
    assert new["history_values"] == []
    assert new["fact_id"] == old["fact_id"]


def test_replacement_and_explicit_removal_have_distinct_omission_rules():
    old = job()
    replaced = apply([old], update(old, fields=fields(end="No longer employed at Cedar Labs"), update_mode="replace_value", change_type="real_change"))[0]
    assert set(replaced["value"]) == {"end"}
    removed = apply([old], update(old, remove_fields=["job_title"]))[0]
    assert set(removed["value"]) == {"employer_name"}
    with pytest.raises(ValueError, match="cannot be empty"):
        apply([old], update(old, remove_fields=list(old["value"])))


def test_mentions_deduplicate_per_message_without_changing_items_or_history():
    old = job()
    mention = {"operation": "MENTION", "target_fact_id": old["fact_id"]}
    new = apply([old], mention, mention, now=LATER)[0]
    assert new["_mention_count"] == old["_mention_count"] + 1
    assert new["value"] == old["value"]
    assert new["history_values"] == old["history_values"]
    repeated = apply([old], update(old, fields=fields(job_title="researcher"), change_type="real_change"), now=LATER)[0]
    assert repeated["value"] == old["value"]
    assert repeated["history_values"] == []


def test_concurrent_jobs_are_separate_facts_even_with_same_attribute():
    op = {"operation": "ADD", "attribute_name": "employment", "fields": fields(employer_name="Cedar Labs")}
    facts = apply([], op, {**op, "fields": fields(employer_name="Pine Labs")})
    assert len({f["fact_id"] for f in facts}) == 2


@pytest.mark.parametrize("reverse", [False, True])
def test_same_attribute_update_and_mentions_select_only_the_offered_uuid(reverse):
    facts = apply([], *[{"operation": "ADD", "attribute_name": "employment",
                       "fields": fields(employer_name=name)} for name in ("Cedar Labs", "Pine Labs")])
    first, second = deepcopy(facts)
    if reverse:
        facts.reverse()
    changed = apply(facts, update(second, attribute_name="employment",
                    fields=fields(end="No longer employed at Pine Labs"), change_type="real_change"), now=LATER)
    by_id = {fact["fact_id"]: fact for fact in changed}
    assert len(by_id) == 2
    assert by_id[first["fact_id"]] == first
    target = by_id[second["fact_id"]]
    assert target["value"]["employer_name"] == second["value"]["employer_name"]
    assert target["history_values"] == [{"_archived_at": LATER, "value": second["value"]}]
    assert target["_mention_count"] == second["_mention_count"] + 1
    mentioned = apply(changed, {"operation": "MENTION", "target_fact_id": first["fact_id"]})
    mentions = {fact["fact_id"]: fact for fact in mentioned}
    assert mentions[second["fact_id"]] == target
    assert mentions[first["fact_id"]]["_mention_count"] == first["_mention_count"] + 1


def test_history_enrichment_is_append_only_and_cannot_correct_history():
    old = job()
    changed = apply([old], update(old, fields=fields(employer_name="Pine Labs"), change_type="real_change"))[0]
    snapshot = deepcopy(changed)
    op = {"operation": "APPEND_HISTORY", "target_fact_id": old["fact_id"], "history_index": 0,
          "fields": fields(workplace_address="25 Pine Avenue")}
    new = apply([changed], op, now=LATER)[0]
    assert new["value"] == changed["value"]
    assert new["history_values"][0] == changed["history_values"][0]
    assert new["history_values"][1]["value"]["employer_name"] == old["value"]["employer_name"]
    assert changed == snapshot
    with pytest.raises(ValueError, match="cannot correct"):
        apply([changed], {**op, "fields": fields(employer_name="Different Labs")})
    with pytest.raises(ValueError, match="offered history"):
        apply([changed], op, offered_history={old["fact_id"]: []})


def test_history_distinct_episode_can_have_same_values():
    old = job()
    op = {"operation": "APPEND_HISTORY", "target_fact_id": old["fact_id"], "fields": fields(employer_name="Cedar Labs")}
    new = apply([old], op, op)[0]
    assert len(new["history_values"]) == 2


@pytest.mark.parametrize("bad", [True, 42, None, {"nested": "data"}])
def test_business_values_are_text_only(bad):
    with pytest.raises(ValidationError):
        wire_value([{"field_name": "works_remotely", "text": bad, "fact_confidence": "high"}])


def test_duplicate_reserved_and_conflicting_fields_rejected():
    assert wire_value(fields(employer_name="A") * 2) == wire_value(fields(employer_name="A"))
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        wire_value(fields(employer_name="A") + fields(employer_name="B"))
    with pytest.raises(ValueError, match="metadata"):
        wire_value(fields(_recorded_at="fake"))
    with pytest.raises(ValueError, match="set and remove"):
        FactMutation(operation="UPDATE", target_fact_id=job()["fact_id"], source_chat_ids=["m1"],
                     fields=fields(employer_name="A"), remove_fields=["employer_name"])


def test_assistant_unknown_sources_and_unoffered_target_rejected_atomically():
    old = job()
    op = update(old, fields=fields(employer_name="A"))
    with pytest.raises(ValueError, match="CURRENT USER"):
        apply([old], {**op, "source_chat_ids": ["unknown"]})
    with pytest.raises(ValueError, match="offered entity"):
        apply([old], op, offered_fact_ids=[])
    with pytest.raises(ValueError, match="CURRENT USER"):
        apply_mutations([old], [{**op, "source_chat_ids": ["m1"]}], [{**CHAT[0], "role": "assistant"}], "entity", "batch")


def test_strict_storage_roundtrip_and_public_allowlist():
    old = job()
    assert Fact.model_validate(old).model_dump(mode="json") == old
    with pytest.raises(ValidationError):
        Fact.model_validate({**old, "source_chat_id": "m1"})
    new = apply([old], update(old, fields=fields(job_title="senior"), change_type="real_change"))[0]
    result = public_fact(new, [])
    assert result["history_is_partial"] is True
    assert result["history_values"] == []
    assert all(set(item) == {"text", "fact_confidence"} for item in result["value"].values())
    assert not any(key.startswith("_") for key in result)
