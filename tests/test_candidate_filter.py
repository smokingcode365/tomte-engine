"""Short labels remain request-local; all identity writes retain UUID scope."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from memory_fixtures import simple_entity, simple_fact
from tomte.engine.core.candidate_filter import CandidateFilterResponse, CandidateFilterUnavailable
from tomte.engine.core.capacity import CapacityExceeded
from tomte.engine.core.resolution import ResolutionResponse
from tomte.engine.workflows.consolidation import ConsolidationWorkflow


def row(name, domain="relationships"):
    p = simple_entity(name, domain="relationships" if domain == "thoughts_and_plans" else domain,
                      facts=[simple_fact("detail", name + " detail")])
    if domain == "thoughts_and_plans":
        p.update(entity_domain=domain, first_mentioned_at="2026-10-09T12:00:00Z")
    return {"id": p["entity_id"], "payload": p,
            **{k: p[k] for k in ("entity_id", "entity_name", "entity_type", "entity_domain")}}


def incoming(name="Person", handle="tmp_person"):
    return {"entity_id": handle, "entity_name": name, "entity_type": "person",
            "entity_domain": "relationships", "aliases": [], "facts": []}


def run_filter(rows, response, *, target=None, entity=None):
    seen = []
    def parse(**kwargs):
        seen.append((kwargs["schema"], json.loads(kwargs["user_text"])))
        if kwargs["schema"] is CandidateFilterResponse:
            if isinstance(response, BaseException):
                raise response
            return response
        return ResolutionResponse.model_validate({"resolutions": [{
            "extracted_entity_id": (entity or incoming())["entity_id"],
            "kind": "MATCH_ONE", "targets": [{"db_id": target}], "reasoning": "Prescribed test identity",
        }] if target else []})
    workflow = ConsolidationWorkflow(Mock(), SimpleNamespace(supports_candidate_filter=True, parse_structured=parse))
    ops = workflow._reconcile_with_golden_records("tenant", "user", [entity or incoming()], rows,
        current_batch=[{"role": "user", "item_id": "m1", "content": "An ambiguous earlier memory."}])
    return workflow, ops, seen


def test_short_labels_map_to_uuid_and_twentieth_candidate_can_be_selected():
    rows = [row(f"Person {i}") for i in range(21)]
    original = deepcopy(rows)
    response = CandidateFilterResponse(keep=["c20"], drop=[f"c{i}" for i in range(1, 20)])
    workflow, ops, seen = run_filter(rows, response, target=rows[19]["id"])
    filtered = seen[0][1]
    assert len(filtered["CANDIDATES"]) == 20
    assert [c["candidate_label"] for c in filtered["CANDIDATES"]] == [f"c{i}" for i in range(1, 21)]
    assert not any(r["id"] in json.dumps(filtered) for r in rows)
    assert set(seen[1][1]["CANDIDATES_BY_ID"]) == {rows[19]["id"]}
    assert seen[1][1]["CANDIDATES_FILTERED"] == {"tmp_person": True}
    assert [o["entity_id"] for o in ops] == [rows[19]["id"]]
    assert ops[0]["tenant_id"] == "tenant" and ops[0]["user_id"] == "user"
    assert "candidate_label" not in json.dumps(ops)
    assert rows == original


@pytest.mark.parametrize("response", [
    CandidateFilterResponse(keep=["c1"], drop=[]),
    CandidateFilterResponse(keep=["c1"], drop=["c1"]),
    CandidateFilterResponse(keep=["c1"], drop=["c3"]),
    {"keep": ["c1"], "drop": ["c2"]},
    CandidateFilterUnavailable("unavailable"),
    TimeoutError("unavailable"),
])
def test_malformed_or_unavailable_filter_preserves_original_candidates(response):
    rows = [row("First"), row("Second")]
    workflow, ops, seen = run_filter(rows, response, target=rows[1]["id"])
    assert set(seen[-1][1]["CANDIDATES_BY_ID"]) == {r["id"] for r in rows}
    assert ops[0]["entity_id"] == rows[1]["id"]
    assert workflow.last_candidate_filter["tmp_person"]["retained"] == 2


@pytest.mark.parametrize("error", [CapacityExceeded("capacity"), RuntimeError("host error"), KeyboardInterrupt()])
def test_filter_does_not_swallow_capacity_host_or_cancellation_errors(error):
    with pytest.raises(type(error)):
        run_filter([row("Person")], error)


def test_empty_filter_still_uses_strong_model_and_preserves_truncation_metadata():
    rows = [dict(row("Person"), identity_matches_total=30)]
    workflow, ops, seen = run_filter(rows, CandidateFilterResponse(keep=[], drop=["c1"]), target=rows[0]["id"])
    assert seen[-1][0] is ResolutionResponse
    assert seen[-1][1]["CANDIDATES_BY_ID"] == {}
    assert seen[-1][1]["CANDIDATE_SET_TRUNCATED"] == {"tmp_person": True}
    assert ops == []  # A removed target cannot be smuggled back by the resolver.


def test_no_candidates_needs_no_filter_request():
    workflow, ops, seen = run_filter([], CandidateFilterResponse(keep=[], drop=[]))
    assert [schema for schema, _ in seen] == [ResolutionResponse]
    assert workflow.last_candidate_filter["tmp_person"]["outcome"] == "empty"


def test_labels_are_scoped_per_object_and_shared_table_cannot_expand_targets():
    first, second = row("First"), row("Second")
    entities = [incoming("First", "tmp_a"), incoming("Second", "tmp_b")]
    seen = []
    def parse(**kwargs):
        body = json.loads(kwargs["user_text"]); seen.append((kwargs["schema"], body))
        if kwargs["schema"] is CandidateFilterResponse:
            assert [c["candidate_label"] for c in body["CANDIDATES"]] == ["c1", "c2"]
            return CandidateFilterResponse(keep=["c1"], drop=["c2"])
        return ResolutionResponse.model_validate({"resolutions": [{
            "extracted_entity_id": "tmp_a", "kind": "MATCH_ONE", "targets": [{"db_id": second["id"]}],
            "reasoning": "Deliberate out-of-scope target",
        }]})
    workflow = ConsolidationWorkflow(Mock(), SimpleNamespace(supports_candidate_filter=True, parse_structured=parse))
    ops = workflow._reconcile_with_golden_records("tenant", "user", entities, [first, second],
        candidate_sets={"tmp_a": [first, second], "tmp_b": [second, first]})
    prompt = seen[-1][1]
    assert set(prompt["CANDIDATES_BY_ID"]) == {first["id"], second["id"]}
    assert [c["id"] for c in prompt["CANDIDATES_BY_OBJECT"]["tmp_a"]] == [first["id"]]
    assert ops == []


def test_keep_all_preserves_more_than_ten_candidates_and_excludes_history():
    rows = [row(f"Dog {i}") for i in range(12)]
    rows[0]["payload"]["facts"][0]["history_values"] = [{"value": {"detail": {"text": "Historical secret"}}}]
    _, _, seen = run_filter(rows, CandidateFilterResponse(keep=[f"c{i}" for i in range(1, 13)], drop=[]))
    assert len(seen[-1][1]["CANDIDATES_BY_ID"]) == 12
    bodies = json.dumps([body for _, body in seen])
    assert "Historical secret" not in bodies
    assert "_recorded_at" not in bodies


def test_thought_actual_boundary_precedes_filter():
    rows = [row("Actual car", "assets_and_possessions"), row("Car wish", "thoughts_and_plans")]
    entity = dict(incoming("Car wish"), entity_domain="thoughts_and_plans")
    _, _, seen = run_filter(rows, CandidateFilterResponse(keep=["c1"], drop=[]), entity=entity)
    assert [c["name"] for c in seen[0][1]["CANDIDATES"]] == ["Car wish"]
