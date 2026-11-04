from copy import deepcopy
import json
from types import SimpleNamespace
import pytest

from tomte.engine.core.config import settings
from tomte.engine.core.memory import public_entity
from tomte.engine.core.retrieval import HistorySelectionResponse, select_history
from tomte.engine.llm.embedder import get_search_entries, search_entry_key, changed_search_entries
from test_memory_values import job, apply, update, fields


def entity():
    fact = job()
    fact = apply([fact], update(fact, fields=fields(employer_name="Pine Labs"), change_type="real_change"))[0]
    fact["_importance"] = 1
    return {"entity_id": "da88b979-2011-4d97-b1cc-a682362507ec", "entity_name": "Bob",
            "entity_type": "person", "entity_domain": "relationships", "aliases": [], "facts": [fact]}


def test_search_includes_low_importance_current_and_old_values_without_metadata():
    value = entity()
    entries = get_search_entries(value)
    assert any("Cedar Labs" in e["text"] and e["history_index"] == 0 for e in entries)
    assert any("Pine Labs" in e["text"] and e["history_index"] == -1 for e in entries)
    assert all("_recorded_at" not in e["text"] for e in entries)


def test_same_attribute_facts_keep_distinct_search_history_and_public_values():
    original = entity()
    original["facts"] = apply([], *[{"operation": "ADD", "attribute_name": "employment",
        "fields": fields(employer_name=name)} for name in ("Cedar Labs", "Pine Labs")])
    value = deepcopy(original)
    target = value["facts"][1]
    value["facts"] = apply(value["facts"], update(target,
        fields=fields(end="No longer employed at Pine Labs"), change_type="real_change"))
    entries = get_search_entries(value)
    assert len({search_entry_key(entry) for entry in entries}) == len(entries) == 4
    assert {entry["fact_id"] for entry in entries} == {fact["fact_id"] for fact in value["facts"]}
    assert all(entry["fact_id"] == target["fact_id"] for entry in changed_search_entries(value, original))
    selection = select_history(value, "previous employments", [])
    assert selection == {value["facts"][0]["fact_id"]: [], target["fact_id"]: [0]}
    public = public_entity(value, selection)
    assert len(public["facts"]) == 2
    assert [fact["attribute_name"] for fact in public["facts"]] == ["employment", "employment"]
    assert public["facts"][0]["history_values"] == []
    assert public["facts"][1]["history_values"][0]["value"]["employer_name"]["text"] == "Pine Labs"


def test_search_chunks_preserve_every_character_of_long_text():
    value = entity()
    text = "不同的信息🙂" * 1500
    value["facts"][0]["value"] = {"detail": {**next(iter(value["facts"][0]["value"].values())), "text": text}}
    entries = [e for e in get_search_entries(value) if e["history_index"] == -1]
    assert len(entries) > 1 and all(len(e["text"]) <= 2000 for e in entries)
    # Prefix is identical, then chunks preserve the complete item representation.
    prefix = entries[0]["text"].split("detail:")[0]
    assert "".join(e["text"][len(prefix):] for e in entries) == "detail: " + text + " (fact_confidence: high)"


def test_small_history_returns_all_without_model_call_or_internal_metadata():
    value = entity()
    llm = SimpleNamespace(parse_structured=lambda **kw: (_ for _ in ()).throw(AssertionError("unneeded call")))
    selection = select_history(value, "previous work", [], llm)
    result = public_entity(value, selection)
    assert result["facts"][0]["history_is_partial"] is False
    assert len(result["facts"][0]["history_values"]) == 1
    serialized = json.dumps(result)
    for internal in ("_recorded_at", "_updated_at", "_archived_at", "_importance", "_mention_count"):
        assert internal not in serialized


def test_large_history_uses_offered_relevant_old_index_and_flags_omission(monkeypatch):
    value = entity()
    fact = value["facts"][0]
    fact["history_values"] *= 30
    original = deepcopy(value)
    monkeypatch.setattr(settings, "memory_history_bytes", 300)
    captured = []
    def choose(**kwargs):
        data = json.loads(kwargs["user_text"])
        captured.extend(data["history_candidates"])
        return HistorySelectionResponse.model_validate({"choices": [{"fact_id": fact["fact_id"], "history_indices": [0]}]})
    selection = select_history(value, "first workplace", [{"fact_id": fact["fact_id"], "history_index": 0}], SimpleNamespace(parse_structured=choose))
    assert captured[0]["history_index"] == 0
    assert selection[fact["fact_id"]] == [0]
    assert public_entity(value, selection)["facts"][0]["history_is_partial"] is True
    assert value == original


def test_invalid_history_selection_cannot_invent_memory(monkeypatch):
    value = entity()
    value["facts"][0]["history_values"] *= 5
    monkeypatch.setattr(settings, "memory_history_bytes", 300)
    llm = SimpleNamespace(parse_structured=lambda **kw: HistorySelectionResponse.model_validate({
        "choices": [{"fact_id": "foreign-fact", "history_indices": [999]}]}))
    result = public_entity(value, select_history(value, "history", [], llm))
    assert "foreign-fact" not in json.dumps(result)
    assert result["facts"][0]["history_is_partial"] is True


@pytest.mark.parametrize("same_episode,alteration,expected", [
    (True, None, [1]), (False, None, [0, 1]),
    (True, "text", [0, 1]), (True, "confidence", [0, 1]),
])
def test_history_coverage_needs_same_episode_and_exact_preservation(monkeypatch, same_episode, alteration, expected):
    from tomte.engine.core.memory import public_value
    value = entity()
    fact = value["facts"][0]
    original = deepcopy(fact["history_values"][0])
    original["value"]["employer_name"]["text"] = "Probably Cedar Labs; I am unsure"
    original["value"]["employer_name"]["fact_confidence"] = "low"
    fuller = deepcopy(original)
    fuller["value"]["workplace_address"] = {**fuller["value"]["employer_name"], "text": "25 Pine Avenue"}
    if alteration == "text":
        fuller["value"]["employer_name"]["text"] = "Cedar Labs"
    if alteration == "confidence":
        fuller["value"]["employer_name"]["fact_confidence"] = "high"
    fact["history_values"] = [original, fuller] + [deepcopy(original) for _ in range(8)]
    budget = sum(len(json.dumps({"value": public_value(h["value"])}, ensure_ascii=False).encode("utf-8")) for h in (original, fuller))
    monkeypatch.setattr(settings, "memory_history_bytes", budget)
    before = deepcopy(value)
    llm = SimpleNamespace(parse_structured=lambda **kw: HistorySelectionResponse.model_validate({
        "choices": [{"fact_id": fact["fact_id"], "history_indices": [0, 1]}],
        "coverage": [{"fact_id": fact["fact_id"], "covered_index": 0, "fuller_index": 1, "same_episode": same_episode}]}))
    result = select_history(value, "old workplace", [], llm)
    assert result[fact["fact_id"]] == expected
    assert public_entity(value, result)["facts"][0]["history_is_partial"] is True
    assert value == before


@pytest.mark.parametrize("index", [True, "1", -1, 1.5])
def test_history_indices_cannot_be_coerced(index):
    with pytest.raises(ValueError):
        HistorySelectionResponse.model_validate({"choices": [{"fact_id": "offered", "history_indices": [index]}]})
