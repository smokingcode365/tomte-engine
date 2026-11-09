"""Prescribed model choices verify routing, not semantic correctness."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from memory_fixtures import simple_entity, simple_fact
from test_unresolved_memories import pending_payload
from tomte.engine.core.resolution import ResolvedTarget
from tomte.engine.workflows.consolidation import ConsolidationWorkflow
from tomte.engine.workflows.unresolved import prepare_unresolved_bindings


def test_identity_target_needs_only_offered_uuid():
    identity = uuid4()
    target = ResolvedTarget.model_validate({"db_id": str(identity)})
    assert target.model_dump(mode="json") == {"db_id": str(identity)}
    assert set(ResolvedTarget.model_json_schema()["properties"]) == {"db_id"}


def record(payload):
    return dict(id=payload["entity_id"], payload=deepcopy(payload), tenant_id="tenant", user_id="user",
                **{k: payload[k] for k in ("entity_id", "entity_name", "entity_type", "entity_domain")})


def workflow(output, db=None):
    def model(**arguments):
        return arguments["schema"].model_validate(deepcopy(output))
    return ConsolidationWorkflow(db=db or Mock(), llm=SimpleNamespace(parse_structured=model))


@pytest.mark.parametrize("target", [{"db_id": "tmp_car"}, {"db_id": str(uuid4()), "other": "ignored?"}])
def test_target_still_rejects_invalid_uuid_and_unrecognized_fields(target):
    with pytest.raises(ValueError):
        ResolvedTarget.model_validate(target)


@pytest.mark.parametrize("metadata", [{}, {"resolution_confidence": .01, "evidence": "SEMANTIC_CANDIDATE"},
                                    {"resolution_confidence": "retired", "evidence": "arbitrary"}])
@pytest.mark.parametrize("route", ["identity", "contextual", "pending_binding"])
def test_model_selected_target_routes_without_semantic_metadata(route, metadata):
    target = record(simple_entity("Tesla", kind="car", domain="assets_and_possessions",
                                 facts=[simple_fact("paint", "Blue")]))
    pending = record(pending_payload())
    before = deepcopy((target, pending))
    quote = "The car I painted is my Tesla."
    current = [dict(item_id="u2", role="user", content=quote, timestamp=1793836800)]
    data = {"CURRENT_BATCH": current, "RECENT_CONTEXT_REFERENCE_ONLY": []}
    db = SimpleNamespace(get_resolution_candidates=lambda *a, **k: [deepcopy(target)],
                         get_unresolved_memories=lambda *a, **k: [deepcopy(pending)])
    choice = dict(db_id=target["id"], **metadata)
    if route == "identity":
        incoming = dict(entity_id="tmp_car", entity_name="Painted car", entity_type="car",
            entity_domain="assets_and_possessions", aliases=[], facts=[dict(attribute_name="paint",
            value={"color": {"text": "red", "fact_confidence": "high"}}, source_chat_id="u2")])
        resolution = dict(extracted_entity_id="tmp_car", kind="MATCH_ONE", targets=[choice], reasoning="Same car.")
        w = workflow({"resolutions": [resolution]}, db)
        ops = w._reconcile_with_golden_records("tenant", "user", [incoming], [target], current_batch=current)
    elif route == "contextual":
        action = dict(action_type="CLARIFICATION", reference_hint="Identifies the painted car.",
            evidence_quote=quote, source_chat_ids=["u2"], target=choice,
            target_fact_id=target["payload"]["facts"][0]["fact_id"])
        w = workflow({"actions": [action]}, db)
        ops = w._resolve_contextual_actions("tenant", "user", data, [1., 0.])
    else:
        binding = dict(unresolved_id=pending["id"], target=choice, source_chat_id="u2", evidence_quote=quote)
        w = workflow({"bindings": [binding]}, db)
        ops, retained = prepare_unresolved_bindings(w, "tenant", "user", data, [1., 0.], [])
        assert retained[pending["id"]]["payload"]["_unresolved"]["status"] == "pending"
        assert ops[0]["extracted_facts"][0]["stored_unresolved_id"] == pending["id"]
    assert len(ops) == 1 and ops[0]["entity_id"] == target["id"]
    assert (target, pending) == before  # Routing alone neither commits nor completes pending memory.


@pytest.mark.parametrize("route", ["contextual", "pending_binding"])
@pytest.mark.parametrize("defect", ["unoffered_target", "assistant_source", "foreign_source", "unsupported_quote"])
def test_source_and_target_checks_survive_without_semantic_veto(route, defect):
    target = record(simple_entity("Tesla", kind="car", domain="assets_and_possessions",
                                 facts=[simple_fact("paint", "Blue")]))
    pending = record(pending_payload())
    quote = "The car I painted is my Tesla."
    data = {"CURRENT_BATCH": [dict(item_id="u2", role="assistant" if defect == "assistant_source" else "user",
                                   content=quote, timestamp=1793836800)], "RECENT_CONTEXT_REFERENCE_ONLY": []}
    db = SimpleNamespace(get_resolution_candidates=lambda *a, **k: [deepcopy(target)],
                         get_unresolved_memories=lambda *a, **k: [deepcopy(pending)])
    choice = {"db_id": str(uuid4()) if defect == "unoffered_target" else target["id"]}
    source = "foreign" if defect == "foreign_source" else "u2"
    evidence = "Unsupported source quotation" if defect == "unsupported_quote" else quote
    if route == "contextual":
        action = dict(action_type="CLARIFICATION", reference_hint="Identifies the painted car.",
            evidence_quote=evidence, source_chat_ids=[source], target=choice,
            target_fact_id=target["payload"]["facts"][0]["fact_id"])
        ops = workflow({"actions": [action]}, db)._resolve_contextual_actions("tenant", "user", data, [1., 0.])
    else:
        binding = dict(unresolved_id=pending["id"], target=choice, source_chat_id=source, evidence_quote=evidence)
        ops, retained = prepare_unresolved_bindings(workflow({"bindings": [binding]}, db),
            "tenant", "user", data, [1., 0.], [])
        assert retained[pending["id"]]["payload"] == pending["payload"]
    assert ops == []


def test_truncated_candidates_do_not_veto_explicit_selected_group():
    from test_resolution import decision, entity, record as car
    # Use the production preparer directly; the group has more search hits than its two named members.
    records = [car(name) for name in ("Tesla", "Honda")]
    for row in records:
        row["identity_matches_total"] = 6
    quote = "I sold the Tesla and Honda."
    result = decision("MATCH_MANY", records, equal=True)
    result.update(group_source_chat_id="current-source", group_evidence_quote=quote)
    w = workflow({"resolutions": [result]})
    ops = w._reconcile_with_golden_records("tenant", "user", [entity()], records,
        current_batch=[dict(item_id="current-source", role="user", content=quote)])
    assert {op["entity_id"] for op in ops} == {row["entity_id"] for row in records}
