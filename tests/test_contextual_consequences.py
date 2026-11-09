"""Routing mechanics only; live probes evaluate the model's semantic choices."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from memory_fixtures import simple_entity, simple_fact, field_entries, fact_text
from tomte.engine.core.contextual import ContextualActionResponse
from tomte.engine.workflows.consolidation import ConsolidationWorkflow


@pytest.mark.parametrize("route_concern", [True, False])
def test_related_fact_reaches_mutation_only_when_model_routes_it(route_concern):
    job = simple_entity("College job", facts=[simple_fact("employment", "Teaching at college")])
    concern = simple_entity("Balancing work", facts=[simple_fact("current_priority", "Balancing two active jobs")])
    untouched = simple_fact("preference", "Keep Sundays free")
    concern["facts"].append(untouched)
    records = [{"id": e["entity_id"], **e, "payload": e} for e in (job, concern)]
    before = deepcopy(records)
    data = dict(CURRENT_BATCH=[dict(item_id="u1", role="user", timestamp=1793664000,
        content="The college job ended. The hospital job continues.")], RECENT_CONTEXT_REFERENCE_ONLY=[])
    incoming = dict(attribute_name="employment", value={"detail": dict(text="College job ended", fact_confidence="high")},
                    source_chat_id="u1", incoming_fact_key="fact_0")
    prepared = [ConsolidationWorkflow._prepare_operation("tenant", "user", {**job, "facts": [incoming]}, records[0])]
    calls = []
    def candidates(tenant, user, *args):
        assert (tenant, user) == ("tenant", "user")
        return records
    def model(system_prompt, user_text, schema):
        payload = json.loads(user_text)
        calls.append(schema.__name__)
        if schema is ContextualActionResponse:
            assert payload["ALREADY_RESOLVED_PROPOSALS"][0]["db_id"] == job["entity_id"]
            assert any(e["db_id"] == concern["entity_id"] for e in payload["CANDIDATES"])
            return schema.model_validate({"actions": [] if not route_concern else [dict(
                action_type="CLARIFICATION", reference_hint="Qualify the former two-job situation; feelings unknown.",
                evidence_quote="The college job ended.", source_chat_ids=["u1"],
                target=dict(db_id=concern["entity_id"], resolution_confidence=.98, evidence="RESOLVED_REFERENCE"),
                target_fact_id=concern["facts"][0]["fact_id"])]})
        plans = []
        for target in payload["TARGETS"]:
            is_concern = target["identity"]["entity_id"] == concern["entity_id"]
            plans.append(dict(target_key=target["target_key"], operations=[dict(operation="UPDATE",
                target_fact_id=target["facts"][0]["fact_id"], change_type="real_change",
                fields=field_entries("Former two-job situation; college ended, hospital continues. Present feelings unknown."
                                     if is_concern else "College job ended"), source_chat_ids=["u1"],
                incoming_fact_key=None if is_concern else "fact_0",
                contextual_action_key=target["contextual_actions"][0]["contextual_action_key"] if is_concern else None)]))
        return schema.model_validate(dict(plans=plans))
    workflow = object.__new__(ConsolidationWorkflow)
    workflow.db = SimpleNamespace(get_resolution_candidates=candidates)
    workflow.llm = SimpleNamespace(parse_structured=model)
    routed = workflow._resolve_contextual_actions("tenant", "user", data, [1.0], prepared=prepared)
    result = workflow._plan_fact_mutations(prepared + routed, data, "batch")
    assert len(result) == (2 if route_concern else 1)
    assert calls == ["ContextualActionResponse", "MutationPlanResponse"]
    assert records == before
    assert fact_text(result[0]["payload"]["facts"][0]) == "College job ended"
    if route_concern:
        stored = result[1]["payload"]
        assert stored["entity_id"] == concern["entity_id"]
        assert stored["facts"][0]["fact_id"] == concern["facts"][0]["fact_id"]
        assert stored["facts"][0]["history_values"][0]["value"] == concern["facts"][0]["value"]
        assert stored["facts"][1] == untouched
