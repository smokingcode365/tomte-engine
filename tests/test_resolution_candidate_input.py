"""Identity input size and per-object scope, without model or database calls."""
from copy import deepcopy
import json
from unittest.mock import Mock

from memory_fixtures import simple_entity, simple_fact, value
from tomte.engine.core.assertions import fact_values
from tomte.engine.core.capacity import check_provider_capacity
from tomte.engine.core.config import settings
from tomte.engine.core.resolution import ResolutionResponse
from tomte.engine.workflows.consolidation import ConsolidationWorkflow


def candidate(name):
    payload = simple_entity(name, facts=[simple_fact("detail", name + " detail " * 350)])
    payload["facts"][0]["history_values"] = [{
        "_archived_at": "2026-10-01T12:00:00Z",
        "value": value("Historical text " * 3000),
    }]
    return {"id": payload["entity_id"], "payload": payload,
            **{k: payload[k] for k in ("entity_id", "entity_name", "entity_type", "entity_domain")}}


def extracted(index):
    return {"entity_id": f"tmp_person_{index}", "entity_name": f"Person {index}",
            "entity_type": "person", "entity_domain": "relationships", "aliases": [],
            "facts": [{"attribute_name": "detail", "value": value("Mentioned"),
                       "importance": 3, "source_chat_id": "source"}]}


def test_repeated_identity_candidates_fit_without_dropping_evidence(monkeypatch):
    # Preserve the original failure's smaller budget even if the default grows.
    monkeypatch.setattr(settings, 'max_model_input_bytes', 65536)
    rows = [candidate(f"Person {i}") for i in range(5)]
    incoming = [extracted(i) for i in range(7)]
    sets = {e["entity_id"]: [{**row, "similarity": (i + j) / 20}
                              for j, row in enumerate(rows[i % 5:] + rows[:i % 5])]
            for i, e in enumerate(incoming)}
    sets[incoming[-1]["entity_id"]] = []
    original = deepcopy((incoming, rows, sets))
    seen = []

    def model(**kwargs):
        seen.append(json.loads(kwargs["user_text"]))
        check_provider_capacity("parse_structured", kwargs)
        return ResolutionResponse(resolutions=[])

    db, llm = Mock(), Mock()
    llm.parse_structured.side_effect = model
    result = ConsolidationWorkflow(db=db, llm=llm)._reconcile_with_golden_records(
        "tenant", "user", incoming, rows, candidate_sets=sets,
        current_batch=[{"item_id": "source", "role": "user", "content": "Mentioned people."}])
    assert result == []
    llm.parse_structured.assert_called_once()
    assert (incoming, rows, sets) == original
    data, = seen
    assert len(data["CANDIDATES_BY_ID"]) == 5
    assert data["CANDIDATES_BY_OBJECT"][incoming[-1]["entity_id"]] == []
    for handle, candidates in sets.items():
        assert data["CANDIDATES_BY_OBJECT"][handle] == [
            {"id": row["id"], "candidate_only_similarity": row["similarity"]} for row in candidates]
        for row in candidates:
            assert data["CANDIDATES_BY_ID"][row["id"]] == {
                "id": row["id"], "entity_id": row["entity_id"],
                "name": row["entity_name"], "type": row["entity_type"],
                "domain": row["entity_domain"], "aliases": row["payload"]["aliases"],
                "facts": fact_values(row["payload"]["facts"]),
            }
    assert "history_values" not in json.dumps(data)
    assert "_recorded_at" not in json.dumps(data)
    assert "Historical text" not in json.dumps(data)
    assert data["CURRENT_BATCH"][0]["content"] == "Mentioned people."
    assert data["CANDIDATE_SET_TRUNCATED"] == {e["entity_id"]: False for e in incoming}
    assert db.mock_calls == []


def test_shared_candidate_table_does_not_expand_an_objects_allowed_targets():
    first, second = candidate("First person"), candidate("Second person")
    incoming = [extracted(0), extracted(1)]
    sets = {incoming[0]["entity_id"]: [first], incoming[1]["entity_id"]: [second]}
    llm = Mock()
    # Second is present in the shared table, but was not offered for object zero.
    llm.parse_structured.return_value = ResolutionResponse.model_validate({"resolutions": [{
        "extracted_entity_id": incoming[0]["entity_id"], "kind": "MATCH_ONE",
        "reasoning": "Prescribed wrong target", "targets": [{
            "db_id": second["id"], "resolution_confidence": 1, "evidence": "EXPLICIT"}],
    }]})
    result = ConsolidationWorkflow(db=Mock(), llm=llm)._reconcile_with_golden_records(
        "tenant", "user", incoming, [first, second], candidate_sets=sets)
    data = json.loads(llm.parse_structured.call_args.kwargs["user_text"])
    assert set(data["CANDIDATES_BY_ID"]) == {first["id"], second["id"]}
    assert result == []
