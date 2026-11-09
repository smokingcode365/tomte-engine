"""Group application follows explicit proposals, not equality of human wording."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from memory_fixtures import simple_entity, simple_fact
from tomte.engine.core.assertions import FactMutation, MutationPlanResponse
from tomte.engine.workflows.consolidation import ConsolidationWorkflow


CHAT = [dict(item_id="source", role="user", timestamp=1793664000,
             content="I enjoy seeing both brothers. Both used to work in York. David works on Mondays."),
        dict(item_id="other", role="user", timestamp=1793664001, content="I also enjoy cycling."),
        dict(item_id="assistant", role="assistant", timestamp=1793664002, content="Untrusted context.")]


def fields(text, name="preference"):
    return [dict(field_name=name, text=text, fact_confidence="high")]


def setup_group():
    entities = [simple_entity(name) for name in ("David", "Daniel")]
    ids = [e["entity_id"] for e in entities]
    operations, plans = [], []
    for index, entity in enumerate(entities):
        entity["facts"] = [simple_fact("employment", "Currently works in Leeds")]
        proposal = dict(attribute_name="family_time", source_chat_id="source",
                        incoming_fact_key="fact_0", _group_key="shared", _group_targets=ids,
                        value={"preference": dict(text="Enjoys seeing both brothers.", fact_confidence="high")})
        operations.append(ConsolidationWorkflow._prepare_operation("tenant", "user",
            {**entity, "facts": [proposal]}, {"id": entity["entity_id"], "payload": entity}))
        plans.append(dict(target_key=f"target_{index}", operations=[dict(operation="ADD",
            attribute_name="family_time", incoming_fact_key="fact_0", source_chat_ids=["source"],
            fields=fields("The user likes spending time with David and Daniel."))]))
    return operations, dict(plans=plans)


def run(operations, response):
    workflow = object.__new__(ConsolidationWorkflow)
    seen = []
    def parse(**kwargs):
        seen.append(kwargs["user_text"])
        return MutationPlanResponse.model_validate(deepcopy(response))
    workflow.llm = SimpleNamespace(parse_structured=parse)
    result = workflow._plan_fact_mutations(operations,
        dict(CURRENT_BATCH=CHAT, RECENT_CONTEXT_REFERENCE_ONLY=[]), "batch")
    return result, workflow.last_mutation_recovery, seen


def test_group_paraphrase_is_saved_for_both_participants():
    operations, response = setup_group()
    before = deepcopy(operations)
    result, recovery, _ = run(operations, response)
    assert len(result) == 2 and recovery["skipped_groups"] == 0
    assert all(o["payload"]["facts"][-1]["value"]["preference"]["text"] ==
               "The user likes spending time with David and Daniel." for o in result)
    assert all(len(o["payload"]["facts"]) == 2 for o in result)
    assert operations == before


@pytest.mark.parametrize("mode", ["UPDATE", "APPEND_HISTORY", "MENTION"])
def test_group_correlation_supports_each_existing_fact_operation(mode):
    operations, response = setup_group()
    for op, p in zip(operations, response["plans"]):
        fact = op["payload"]["facts"][0]
        fact["attribute_name"] = "employment"
        incoming = op["extracted_facts"][0]
        incoming.update(attribute_name="work", value={"place": dict(text="Previously worked in York.", fact_confidence="high")})
        if mode == "MENTION":
            # LLM determines that the same history is already known, even though
            # its spelling and attribute differ from the incoming interpretation.
            fact["history_values"] = [dict(_archived_at="2026-10-01T00:00:00Z",
                value={"place": dict(text="Former workplace: York.", fact_confidence="high",
                    _recorded_at="2026-10-01T00:00:00Z", _updated_at="2026-10-01T00:00:00Z")})]
        p["operations"] = [dict(operation=mode, target_fact_id=fact["fact_id"],
            incoming_fact_key="fact_0", source_chat_ids=["source"],
            fields=[] if mode == "MENTION" else fields("Former workplace: York.", "place"))]
    before = deepcopy(operations)
    result, recovery, _ = run(operations, response)
    assert len(result) == 2 and recovery["skipped_groups"] == 0
    for old, op in zip(before, result):
        original, fact = old["payload"]["facts"][0], op["payload"]["facts"][0]
        assert fact["fact_id"] == original["fact_id"] and fact["attribute_name"] == "employment"
        if mode == "APPEND_HISTORY":
            assert fact["value"] == original["value"]
            assert fact["history_values"][-1]["value"]["place"]["text"] == "Former workplace: York."
        elif mode == "MENTION":
            assert fact["value"] == original["value"] and fact["history_values"] == original["history_values"]
        else:
            assert fact["value"]["place"]["text"] == "Former workplace: York."
        assert fact["_mention_count"] == original["_mention_count"] + 1


@pytest.mark.parametrize("defect", ["abstain", "unknown_key", "missing_key", "wrong_source", "assistant_source", "foreign_fact"])
def test_invalid_participant_blocks_group_but_keeps_independent_fact(defect):
    operations, response = setup_group()
    bad = response["plans"][1]["operations"][0]
    if defect == "abstain":
        response["plans"][1]["operations"] = []
    elif defect == "unknown_key":
        bad["incoming_fact_key"] = "not_offered"
    elif defect == "missing_key":
        bad["incoming_fact_key"] = None
    elif defect in ("wrong_source", "assistant_source"):
        bad["source_chat_ids"] = ["other" if defect == "wrong_source" else "assistant"]
    else:
        bad.update(operation="UPDATE", target_fact_id=operations[0]["payload"]["facts"][0]["fact_id"], attribute_name=None)
    operations[0]["extracted_facts"].append(dict(attribute_name="hobby", incoming_fact_key="fact_1",
        source_chat_id="other", value={"activity": dict(text="Cycling", fact_confidence="high")}))
    response["plans"][0]["operations"].append(dict(operation="ADD", attribute_name="hobby",
        incoming_fact_key="fact_1", source_chat_ids=["other"], fields=fields("Cycling", "activity")))
    result, recovery, _ = run(operations, response)
    assert len(result) == 1 and result[0]["entity_id"] == operations[0]["entity_id"]
    assert [f["attribute_name"] for f in result[0]["payload"]["facts"]] == ["employment", "hobby"]
    assert result[0]["payload"]["facts"][0] == operations[0]["payload"]["facts"][0]
    assert recovery["skipped_groups"] == 1


def contextual_setup():
    operations, response = setup_group()
    fact = operations[0]["payload"]["facts"][0]
    operations[0]["_contextual_actions"] = [dict(action_type="CLARIFICATION", reference_hint="David's schedule",
        evidence_quote="David works on Mondays.", source_chat_ids=["source"], target_fact_id=fact["fact_id"],
        target=dict(db_id=operations[0]["entity_id"], resolution_confidence=1, evidence="EXPLICIT"))]
    mutation = dict(operation="UPDATE", target_fact_id=fact["fact_id"], source_chat_ids=["source"],
        contextual_action_key="action_0", fields=fields("Monday", "schedule"))
    response = dict(plans=[dict(target_key="target_0", operations=[mutation])])
    return operations, response


def test_contextual_update_sharing_group_source_remains_independent():
    import json
    operations, response = contextual_setup()
    result, recovery, seen = run(operations, response)
    assert len(result) == 1 and recovery["skipped_units"] == 0
    fact = result[0]["payload"]["facts"][0]
    assert fact["value"]["schedule"]["text"] == "Monday"
    assert not any(f["attribute_name"] == "family_time" for f in result[0]["payload"]["facts"])
    assert json.loads(seen[0])["TARGETS"][0]["contextual_actions"][0]["contextual_action_key"] == "action_0"
    assert "contextual_action_key" not in json.dumps(result[0]["payload"])


@pytest.mark.parametrize("defect", ["key", "fact", "source", "add", "missing", "both"])
def test_contextual_reference_cannot_bypass_group_authorization(defect):
    operations, response = contextual_setup()
    mutation = response["plans"][0]["operations"][0]
    if defect == "key":
        mutation["contextual_action_key"] = "unknown"
    elif defect == "fact":
        other = simple_fact("hobby", "Cycling")
        operations[0]["payload"]["facts"].append(other)
        mutation["target_fact_id"] = other["fact_id"]
    elif defect == "source":
        mutation["source_chat_ids"] = ["other"]
    elif defect == "add":
        mutation.update(operation="ADD", target_fact_id=None, attribute_name="invented")
    elif defect == "missing":
        mutation.pop("contextual_action_key")
    else:
        mutation["incoming_fact_key"] = "fact_0"
    result, _, _ = run(operations, response)
    assert result == []


def test_repair_cannot_retarget_a_contextual_action():
    operations, response = contextual_setup()
    raw = response["plans"][0]["operations"][0]
    del raw["fields"][0]["fact_confidence"]
    original = MutationPlanResponse.model_validate(response)
    repaired = deepcopy(response)
    repaired["plans"][0]["operations"][0]["fields"][0]["fact_confidence"] = "high"
    repaired["plans"][0]["operations"][0]["contextual_action_key"] = "action_1"
    merged = original.merge_repair(MutationPlanResponse.model_validate(repaired))
    assert merged.plans[0].operations == []


def test_group_meaning_contract_is_explicit_in_prompt():
    from tomte.engine.llm.prompts.fact_mutation import MUTATION_SYSTEM_PROMPT
    assert "semantic acknowledgement" in MUTATION_SYSTEM_PROMPT
    assert "contextual_action_key" in MUTATION_SYSTEM_PROMPT
    assert "verbatim" in MUTATION_SYSTEM_PROMPT
